import os
import io
import re
import sys
import json
import time
import boto3
import hashlib
import logging
import threading
from typing import List, Dict, Any, Optional, Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from collections import defaultdict

from boto3.s3.transfer import TransferConfig

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


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ---------- Windows/path safety helpers ----------
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')

def _safe_segment(text: str, maxlen: int = 120) -> str:
    """Sanitize a single path segment for Windows compatibility and cap length."""
    if text is None:
        text = ""
    s = str(text)
    s = _INVALID_CHARS.sub("_", s)          # replace invalid chars
    s = re.sub(r"\s+", " ", s)              # collapse whitespace
    s = s.strip(" .")                       # no trailing dots/spaces
    if not s:
        s = "_"
    if len(s) > maxlen:
        # keep a recognizable prefix; suffix with short hash to avoid collisions
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
        s = f"{s[:maxlen-13]}_{h}"
    # avoid Windows reserved device names (CON, PRN, AUX, NUL, COM1.., LPT1..)
    reserved = {"CON","PRN","AUX","NUL"} | {f"COM{i}" for i in range(1,10)} | {f"LPT{i}" for i in range(1,10)}
    if s.upper() in reserved:
        s = f"_{s}_"
    return s

def _safe_hash_token(data: str, length: int = 32) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:length]

def _win_long_path(path: str) -> str:
    """Prefix with \\?\ on Windows when needed to bypass MAX_PATH issues."""
    if os.name == "nt":
        if not path.startswith("\\\\?\\"):
            # Only use for absolute paths
            abs_path = os.path.abspath(path)
            if len(abs_path) >= 240:
                return "\\\\?\\" + abs_path
            return abs_path
    return path

def _safe_local_rel_from_key(s3_key: str) -> str:
    """
    Create a Windows-safe, not-too-deep relative path from an S3 key.
    We keep the first two segments (usually: session + slug) as sanitized folders
    and collapse the rest (often a long base64/URL-ish tail) into a short hash filename.
    """
    parts = [p for p in s3_key.split("/") if p not in ("", ".", "..")]

    if not parts:
        return _safe_segment("file")

    # Take up to first 2 segments as folders (session / slug)
    head = parts[:2]
    tail = "/".join(parts[2:]) if len(parts) > 2 else parts[-1]

    safe_head = [_safe_segment(p) for p in head]
    # File leaf becomes a stable hash, with a tiny human hint if possible
    hint = _safe_segment(head[1] if len(head) > 1 else parts[0], 40)
    leaf = f"{hint}__{_safe_hash_token(tail, 40)}"

    rel_path = os.path.join(*safe_head, leaf) if safe_head else leaf
    return rel_path


def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    # returns (bucket, key)
    if s3_uri.startswith("s3://"):
        u = urlparse(s3_uri)
        return u.netloc, u.path.lstrip("/")
    # fallback: allow raw keys
    return "", s3_uri


class S3MetadataProcessor:
    def __init__(
        self,
        s3_bucket: str,
        session_id: Optional[str] = None,
        milvus_config: Optional[Dict] = None,
        local_cache_dir: str = "./_cache/cssf_session",
        max_workers: int = 12,
        download_max_workers: int = 16,
        multipart_chunksize_mb: int = 64,
        region_name: Optional[str] = None,
        batch_store_size: int = 256,
    ):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.local_cache_dir = os.path.abspath(local_cache_dir)
        self.max_workers = max_workers
        self.download_max_workers = download_max_workers
        self.batch_store_size = batch_store_size

        session = boto3.session.Session(region_name=region_name)
        self.s3 = session.client("s3")

        self.transfer_config = TransferConfig(
            multipart_threshold=multipart_chunksize_mb * 1024 * 1024,
            multipart_chunksize=multipart_chunksize_mb * 1024 * 1024,
            max_concurrency=self.download_max_workers,
            use_threads=True,
        )

        self.processor = DocumentProcessor(parsers=[EurlexHTMLParser(), CSSFHTMLParser(), PDFParserPipeline()])
        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)

        # dedupe structures (thread-safe)
        self.seen_hashes: set[str] = set()
        self._seen_lock = threading.Lock()

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
            region_name=region_name or "eu-west-1",
        )

    # ---------------- util ----------------

    def list_sessions(self) -> List[str]:
        try:
            resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix="", Delimiter="/")
            sessions = [p["Prefix"].rstrip("/") for p in resp.get("CommonPrefixes", [])]
            sessions.sort(reverse=True)
            return sessions
        except Exception as e:
            self.logger.error(f"Error listing sessions: {e}", exc_info=True)
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
            self.logger.error(f"Error reading metadata {s3_key}: {e}", exc_info=True)
            return {}

    # ---------------- local cache (pre-download) ----------------

    def _download_one(self, bucket: str, key: str, dst_path: str) -> Tuple[str, bool]:
        _ensure_dir(os.path.dirname(dst_path))
        # quick skip if present (simple heuristic; for correctness you could check ETag in future)
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            return (dst_path, False)
        try:
            dst_for_api = _win_long_path(dst_path)
            self.s3.download_file(
                Bucket=bucket,
                Key=key,
                Filename=dst_for_api,
                Config=self.transfer_config,
            )
            return (dst_path, True)
        except Exception as e:
            self.logger.error(f"Download failed: s3://{bucket}/{key} -> {dst_path}: {e}")
            return (dst_path, False)

    def collect_all_s3_uris_for_session(self, session_id: str) -> List[str]:
        """Reads all metadata.json files (in parallel) and extracts every 'top_related[].s3_uri'."""
        keys = self.get_session_metadata_files(session_id)
        if not keys:
            return []
        self.logger.info(f"Found {len(keys)} metadata files for session {session_id}")

        uris: set[str] = set()
        with ThreadPoolExecutor(max_workers=min(32, self.max_workers * 2)) as ex:
            futures = {ex.submit(self.read_metadata_from_s3, k): k for k in keys}
            for fut in as_completed(futures):
                md = fut.result() or {}
                for fi in md.get("top_related", []):
                    s3_uri = fi.get("s3_uri")
                    if s3_uri:
                        uris.add(s3_uri)

        all_uris = sorted(uris)
        self.logger.info(f"Collected {len(all_uris)} files to download for session {session_id}")
        return all_uris

    def pre_download_session_files(self, session_id: str) -> Dict[str, str]:
        """
        Download ALL referenced S3 objects for the session into the local cache directory.
        Returns a mapping { s3_uri -> local_path }.
        """
        t0 = time.time()
        uris = self.collect_all_s3_uris_for_session(session_id)
        if not uris:
            return {}

        # some s3_uri may not include bucket; assume current bucket if missing
        plan: List[Tuple[str, str, str, str]] = []  # (s3_uri, bucket, key, local_path)
        for u in uris:
            b, k = _parse_s3_uri(u)
            if not b:
                b = self.s3_bucket

            # Build a Windows-safe relative path that doesn't explode depth/length
            local_rel = _safe_local_rel_from_key(k)
            local_path = os.path.join(self.local_cache_dir, "objects", local_rel)
            plan.append((u, b, k, local_path))

        # parallel download
        downloaded = 0
        with ThreadPoolExecutor(max_workers=self.download_max_workers) as ex:
            futures = [ex.submit(self._download_one, b, k, lp) for (_u, b, k, lp) in plan]
            for fut in as_completed(futures):
                _path, did = fut.result()
                downloaded += 1 if did else 0

        elapsed = time.time() - t0
        self.logger.info(f"Pre-download complete: {downloaded}/{len(plan)} fetched in {elapsed:.1f}s (~{len(plan)} total)")

        return {u: lp for (u, _b, _k, lp) in plan}

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
        if not meta.get("doc_id"):
            base = (meta.get("url", "") + str(meta.get("page_number", 0)) + text).encode("utf-8")
            meta["doc_id"] = hashlib.sha256(base).hexdigest()
        try:
            meta["page_number"] = int(meta.get("page_number", 0))
        except Exception:
            meta["page_number"] = 0
        return meta

    @staticmethod
    def _extract_page_number(doc: Document) -> int:
        try:
            pn = doc.metadata.get("page_number", 0) if isinstance(doc.metadata, dict) else 0
            return int(pn) if pn is not None else 0
        except Exception:
            return 0

    # ---------------- pipeline ----------------

    def hash_document(self, doc: Document) -> str:
        base = (doc.page_content + str(doc.metadata.get("url", "")) + str(doc.metadata.get("page_number", 0))).encode("utf-8")
        return hashlib.sha256(base).hexdigest()

    def _read_local_bytes(self, local_path: str) -> bytes:
        try:
            with open(_win_long_path(local_path), "rb") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading local file {local_path}: {e}")
            return b""

    def process_document(self, metadata: Dict[str, Any], local_map: Dict[str, str]) -> List[Document]:
        """Parse every referenced *local* file, chunk, and attach only your fields."""
        all_docs: List[Document] = []

        for file_info in metadata.get("top_related", []):
            try:
                s3_uri = file_info.get("s3_uri")
                original_url = file_info.get("url")
                content_type = file_info.get("content_type", "application/octet-stream")
                if not s3_uri:
                    continue

                local_path = local_map.get(s3_uri)
                if not local_path or not os.path.exists(local_path):
                    # also try with implicit bucket (if s3_uri lacked it)
                    if not local_path:
                        bkt, key = _parse_s3_uri(s3_uri)
                        if not bkt:
                            # build implicit-URI using default bucket
                            implicit = f"s3://{self.s3_bucket}/{key}"
                            local_path = local_map.get(implicit)
                if not local_path or not os.path.exists(local_path):
                    self.logger.warning(f"Local copy missing for {s3_uri} (skipping)")
                    continue

                self.logger.info(f"Processing document: {original_url}")
                content = self._read_local_bytes(local_path)
                if not content:
                    continue

                elements = self.processor.process(content, original_url, content_type)
                chunked_docs = self.chunker.chunk_document(elements, original_url)

                for d in chunked_docs:
                    pn = self._extract_page_number(d)
                    d.metadata = self.flatten_metadata_for_search(metadata, page_number=pn)
                all_docs.extend(chunked_docs)

            except Exception as e:
                self.logger.error(f"Error processing document {file_info.get('url','unknown')}: {e}", exc_info=True)

        return all_docs

    def store_documents_in_milvus(self, documents: List[Document]) -> Dict[str, Any]:
        if not documents:
            return {"count": 0, "milvus_ids": []}

        total_count = 0
        all_ids: List[Any] = []

        batch: List[Document] = []
        for d in documents:
            meta = self._ensure_required_fields(self._filter_to_schema(dict(d.metadata or {})), d.page_content)

            # dedupe across threads
            doc_hash = self.hash_document(Document(page_content=d.page_content, metadata=meta))
            with self._seen_lock:
                if doc_hash in self.seen_hashes:
                    continue
                self.seen_hashes.add(doc_hash)
            meta["doc_id"] = doc_hash

            nd = Document(page_content=d.page_content, metadata=meta)
            batch.append(nd)

            if len(batch) >= self.batch_store_size:
                cnt, ids = self._flush_batch(batch)
                total_count += cnt
                all_ids.extend(ids)
                batch = []

        if batch:
            cnt, ids = self._flush_batch(batch)
            total_count += cnt
            all_ids.extend(ids)

        self.logger.info(f"Stored {total_count} new documents in Milvus")
        return {"count": total_count, "milvus_ids": all_ids}

    def _flush_batch(self, docs: List[Document]) -> Tuple[int, List[Any]]:
        try:
            texts = [d.page_content for d in docs]
            metas = [d.metadata for d in docs]
            result = self.embedding_service.add_texts_to_store(texts=texts, metadatas=metas)
            return result.get("count", len(docs)), result.get("milvus_ids", [])
        except Exception as e:
            self.logger.error(f"Failed to store batch in Milvus: {e}", exc_info=True)
            return (0, [])

    def process_session(self, session_id: str) -> Dict[str, Any]:
        self.logger.info(f"Processing session: {session_id}")

        # 1) Pre-download everything referenced for this session
        local_map = self.pre_download_session_files(session_id)
        if not local_map:
            self.logger.warning(f"No files downloaded for session {session_id}")
        else:
            self.logger.info(f"Local cache at {self.local_cache_dir}")

        # 2) Process all metadata files in parallel
        keys = self.get_session_metadata_files(session_id)
        self.logger.info(f"Metadata files to process: {len(keys)}")

        if not keys:
            return {"session_id": session_id, "processed": 0, "stored": 0, "errors": 0}

        total_processed = 0
        total_errors = 0
        all_docs: List[Document] = []

        def _work_one(k: str) -> Tuple[str, int, List[Document], Optional[str]]:
            try:
                md = self.read_metadata_from_s3(k)
                if not md:
                    return (k, 0, [], "empty-metadata")
                md["crawl_session"] = session_id
                docs = self.process_document(md, local_map)
                return (k, len(docs), docs, None)
            except Exception as e:
                return (k, 0, [], str(e))

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(_work_one, k): k for k in keys}
            for fut in as_completed(futures):
                k, n, docs, err = fut.result()
                if err:
                    self.logger.error(f"Error processing {k}: {err}")
                    total_errors += 1
                else:
                    self.logger.info(f"Processed {k}: {n} chunks")
                    total_processed += n
                    # append docs (safe: we only extend in main thread)
                    all_docs.extend(docs)

        # 3) Store (batched) — single-threaded to keep Milvus/embeddings happy
        res = self.store_documents_in_milvus(all_docs)
        summary = {
            "session_id": session_id,
            "processed": total_processed,
            "stored": res.get("count", 0),
            "errors": total_errors,
        }
        self.logger.info(f"Session {session_id} complete: {summary}")
        return summary

    def process_latest_session(self) -> Dict[str, Any]:
        latest = self.get_most_recent_session()
        if not latest:
            self.logger.warning("No sessions found")
            return {"processed": 0, "stored": 0, "errors": 0}
        return self.process_session(latest)


def main():
    S3_BUCKET = os.getenv("CSSF_S3_BUCKET", "cssf-crawl")
    SESSION_ID = os.getenv("CSSF_SESSION_ID")  # optional; when None we pick most recent
    REGION = os.getenv("AWS_REGION", "eu-west-1")

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
        local_cache_dir=os.getenv("CSSF_LOCAL_CACHE", "./_cache/cssf_session"),
        max_workers=int(os.getenv("CSSF_MAX_WORKERS", "12")),
        download_max_workers=int(os.getenv("CSSF_DL_WORKERS", "16")),
        multipart_chunksize_mb=int(os.getenv("CSSF_CHUNK_MB", "64")),
        region_name=REGION,
        batch_store_size=int(os.getenv("CSSF_BATCH_STORE", "256")),
    )

    sessions = processor.list_sessions()
    print(f"Available sessions: {sessions}")

    target_session = SESSION_ID or processor.get_most_recent_session()
    if target_session:
        print(f"Processing session: {target_session}")
        result = processor.process_session(target_session)
        print(f"Processing complete: {result}")
    else:
        print("No sessions found to process")


if __name__ == "__main__":
    main()
