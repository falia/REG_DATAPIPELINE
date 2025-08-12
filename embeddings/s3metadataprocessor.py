import json
import boto3
import hashlib
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from embeddings.embedding_provider.embedding_provider import EmbeddingService
from embeddings.chunker.document_chunker import DocumentChunker
from embeddings.parsers.parser import EurlexHTMLParser, CSSFHTMLParser, PDFParserPipeline, DocumentProcessor


# --- only the fields you said you want (plus vector/text/doc_id used by the store) ---
ALLOWED_FIELDS = {
    "text", "vector", "doc_id",
    "url", "title", "subtitle", "document_type", "document_number",
    "publication_date", "update_date", "content_hash", "crawl_timestamp",
    "file_size", "lang", "super_category", "crawl_session",
    "page_number", "top_related", "bottom_related", "themes", "entities", "keywords",
}


class S3MetadataProcessor:
    def __init__(self, s3_bucket: str, session_id: Optional[str] = None, milvus_config: Optional[Dict] = None):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.s3 = boto3.client("s3")

        self.processor = DocumentProcessor(parsers=[EurlexHTMLParser(), CSSFHTMLParser(), PDFParserPipeline()])
        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)
        self.seen_hashes: set[str] = set()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if milvus_config is None:
            milvus_config = {
                "host": "34.241.177.15",
                "port": "19530",
                "collection_name": "cssf_documents_final",
                "connection_args": {"host": "34.241.177.15", "port": "19530"},
            }

        self.embedding_service = EmbeddingService(
            use_tei=True,
            milvus_config=milvus_config,
            endpoint_name="embedding-endpoint",
            region_name="eu-west-1",
        )

    # ---------------- util ----------------

    def list_sessions(self) -> List[str]:
        try:
            resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix="", Delimiter="/")
            sessions = [p["Prefix"].rstrip("/") for p in resp.get("CommonPrefixes", [])]
            sessions.sort(reverse=True)
            return sessions
        except Exception as e:
            self.logger.error(f"Error listing sessions: {e}")
            return []

    def get_most_recent_session(self) -> Optional[str]:
        sessions = self.list_sessions()
        if not sessions:
            self.logger.warning("No sessions found")
            return None
        most_recent = sessions[0]
        self.logger.info(f"Most recent session: {most_recent}")
        return most_recent

    def get_session_metadata_files(self, session_id: str) -> List[str]:
        try:
            paginator = self.s3.get_paginator("list_objects_v2")
            it = paginator.paginate(Bucket=self.s3_bucket, Prefix=f"{session_id}/")
            out: List[str] = []
            for page in it:
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith("metadata.json"):
                        out.append(obj["Key"])
            return out
        except Exception as e:
            self.logger.error(f"Error listing metadata files for session {session_id}: {e}", exc_info=True)
            return []

    def read_metadata_from_s3(self, s3_key: str) -> Dict[str, Any]:
        try:
            resp = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except Exception as e:
            self.logger.error(f"Error reading metadata {s3_key}: {e}")
            return {}

    def download_document_from_s3(self, s3_uri: str) -> bytes:
        try:
            s3_key = s3_uri.replace(f"s3://{self.s3_bucket}/", "")
            resp = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
            return resp["Body"].read()
        except Exception as e:
            self.logger.error(f"Error downloading document from {s3_uri}: {e}")
            return b""

    # ---------------- metadata shaping (ONLY your fields + page_number) ----------------

    @staticmethod
    def _clamp(s: Any, n: int) -> str:
        s = "" if s is None else str(s)
        return s[:n]

    @staticmethod
    def _as_json(val: Any, default: Any):
        if isinstance(val, (list, dict)):
            return val
        if isinstance(val, str):
            try:
                loaded = json.loads(val)
                return loaded if isinstance(loaded, (list, dict)) else default
            except Exception:
                return default
        return default

    def flatten_metadata_for_search(self, metadata: dict, page_number: int | None = None) -> dict:
        """Return only the fields you want; arrays kept as JSON; include page_number if provided."""
        md = {
            "url": self._clamp(metadata.get("url", ""), 1000),
            "title": self._clamp(metadata.get("title", ""), 1000),
            "subtitle": self._clamp(metadata.get("subtitle", ""), 500),
            "document_type": self._clamp(metadata.get("document_type", ""), 100),
            "document_number": self._clamp(metadata.get("document_number", ""), 100),
            "publication_date": self._clamp(metadata.get("publication_date") or "", 50),
            "update_date": self._clamp(metadata.get("update_date") or "", 50),
            "content_hash": self._clamp(metadata.get("content_hash", ""), 100),
            "crawl_timestamp": self._clamp(metadata.get("crawl_timestamp", ""), 50),
            "file_size": int(metadata.get("file_size") or 0),
            "lang": self._clamp(metadata.get("lang", ""), 10),
            "super_category": self._clamp(metadata.get("super_category", ""), 100),
            "crawl_session": self._clamp(metadata.get("crawl_session", ""), 50),
            # JSON columns
            "top_related": self._as_json(metadata.get("top_related", []), []),
            "bottom_related": self._as_json(metadata.get("bottom_related", []), []),
            "themes": self._as_json(metadata.get("themes", []), []),
            "entities": self._as_json(metadata.get("entities", []), []),
            "keywords": self._as_json(metadata.get("keywords", []), []),
        }
        if page_number is not None:
            try:
                md["page_number"] = int(page_number)
            except Exception:
                md["page_number"] = 0
        return md

    @staticmethod
    def _filter_to_schema(meta: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in meta.items() if k in ALLOWED_FIELDS}

    @staticmethod
    def _ensure_required_fields(meta: Dict[str, Any], text: str) -> Dict[str, Any]:
        # doc_id must exist
        if not meta.get("doc_id"):
            base = (meta.get("url", "") + str(meta.get("page_number", 0)) + text).encode("utf-8")
            meta["doc_id"] = hashlib.sha256(base).hexdigest()
        # page_number default
        try:
            meta["page_number"] = int(meta.get("page_number", 0))
        except Exception:
            meta["page_number"] = 0
        return meta

    @staticmethod
    def _extract_page_number(doc: Document) -> int:
        # If the chunker preserved page_number, use it; else 0
        try:
            pn = doc.metadata.get("page_number", 0) if isinstance(doc.metadata, dict) else 0
            return int(pn) if pn is not None else 0
        except Exception:
            return 0

    # ---------------- pipeline ----------------

    def hash_document(self, doc: Document) -> str:
        base = (doc.page_content + str(doc.metadata.get("url", "")) + str(doc.metadata.get("page_number", 0))).encode("utf-8")
        return hashlib.sha256(base).hexdigest()

    def convert_elements_to_documents(self, elements, base_meta: Dict[str, Any]) -> List[Document]:
        """(Unused when chunking; kept for completeness)."""
        docs: List[Document] = []
        for i, el in enumerate(elements):
            try:
                txt = (getattr(el, "text", None) or getattr(el, "page_content", None) or str(el)).strip()
                if not txt:
                    continue
                pn = getattr(getattr(el, "metadata", None), "page_number", None)
                meta = self.flatten_metadata_for_search(base_meta, page_number=pn if isinstance(pn, int) else 0)
                docs.append(Document(page_content=txt, metadata=meta))
            except Exception as e:
                self.logger.error(f"Error converting element {i}: {e}")
        return docs

    def process_document(self, metadata: Dict[str, Any]) -> List[Document]:
        """Download & parse every referenced file, chunk, and attach only your fields."""
        all_docs: List[Document] = []

        for file_info in metadata.get("top_related", []):
            try:
                s3_uri = file_info.get("s3_uri")
                original_url = file_info.get("url")
                content_type = file_info.get("content_type", "application/octet-stream")
                if not s3_uri:
                    continue

                self.logger.info(f"Processing document: {original_url}")
                content = self.download_document_from_s3(s3_uri)
                if not content:
                    continue

                elements = self.processor.process(content, original_url, content_type)

                # chunk
                chunked_docs = self.chunker.chunk_document(elements, original_url)

                # replace each chunk's metadata with ONLY your flattened fields (+ page_number if present)
                for d in chunked_docs:
                    pn = self._extract_page_number(d)
                    d.metadata = self.flatten_metadata_for_search(metadata, page_number=pn)
                all_docs.extend(chunked_docs)

            except Exception as e:
                self.logger.error(f"Error processing document {file_info.get('url','unknown')}: {e}")

        return all_docs

    def store_documents_in_milvus(self, documents: List[Document]) -> Dict[str, Any]:
        if not documents:
            return {"count": 0, "milvus_ids": []}

        new_docs, texts, metas = [], [], []

        for d in documents:
            # make sure required fields exist & metadata only contains schema fields
            meta = self._ensure_required_fields(self._filter_to_schema(dict(d.metadata or {})), d.page_content)

            # dedupe
            doc_hash = self.hash_document(Document(page_content=d.page_content, metadata=meta))
            if doc_hash in self.seen_hashes:
                continue
            self.seen_hashes.add(doc_hash)
            meta["doc_id"] = doc_hash  # use this as the doc_id you store

            new_docs.append(d)
            texts.append(d.page_content)
            metas.append(meta)

        if not new_docs:
            return {"count": 0, "milvus_ids": []}

        try:
            result = self.embedding_service.add_texts_to_store(texts=texts, metadatas=metas)
            self.logger.info(f"Stored {result['count']} new documents in Milvus")
            return result
        except Exception as e:
            self.logger.error(f"Failed to store documents in Milvus: {e}")
            return {"count": 0, "milvus_ids": []}

    def process_session(self, session_id: str) -> Dict[str, Any]:
        self.logger.info(f"Processing session: {session_id}")
        keys = self.get_session_metadata_files(session_id)
        print("Meta data files to process ---------------------------------------------------------", len(keys))

        if not keys:
            self.logger.warning(f"No metadata files found for session {session_id}")
            return {"processed": 0, "stored": 0, "errors": 0}

        total_processed = total_stored = total_errors = 0

        for k in keys:
            try:
                md = self.read_metadata_from_s3(k)
                if not md:
                    continue
                md["crawl_session"] = session_id

                docs = self.process_document(md)
                total_processed += len(docs)

                res = self.store_documents_in_milvus(docs)
                total_stored += res["count"]

                self.logger.info(f"Processed {k}: {len(docs)} docs, {res['count']} stored")
            except Exception as e:
                self.logger.error(f"Error processing {k}: {e}")
                total_errors += 1

        summary = {"session_id": session_id, "processed": total_processed, "stored": total_stored, "errors": total_errors}
        self.logger.info(f"Session {session_id} complete: {summary}")
        return summary

    def process_latest_session(self) -> Dict[str, Any]:
        latest = self.get_most_recent_session()
        if not latest:
            self.logger.warning("No sessions found")
            return {"processed": 0, "stored": 0, "errors": 0}
        return self.process_session(latest)


def main():
    S3_BUCKET = "cssf-crawl"
    SESSION_ID = None
    MILVUS_CONFIG = {
        "host": "34.241.177.15",
        "port": "19530",
        "collection_name": "cssf_documents_final_final_CGDEMO4",
        "connection_args": {"host": "34.241.177.15", "port": "19530"},
    }

    processor = S3MetadataProcessor(
        s3_bucket=S3_BUCKET,
        session_id=SESSION_ID,
        milvus_config=MILVUS_CONFIG,
    )

    sessions = processor.list_sessions()
    print(f"Available sessions: {sessions}")

    most_recent = processor.get_most_recent_session()
    if most_recent:
        print(f"Most recent session: {most_recent}")
        result = processor.process_session(most_recent)
        print(f"Processing complete: {result}")
    else:
        print("No sessions found to process")


if __name__ == "__main__":
    main()
