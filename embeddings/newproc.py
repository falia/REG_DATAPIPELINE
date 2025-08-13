import json
import boto3
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial
import queue
import time

from embeddings.embedding_provider.embedding_provider import EmbeddingService
from embeddings.chunker.document_chunker import DocumentChunker
from embeddings.parsers.parser import EurlexHTMLParser, CSSFHTMLParser, PDFParserPipeline, DocumentProcessor


ALLOWED_FIELDS = {
    "text", "vector", "doc_id",
    "url", "title", "subtitle", "document_type", "document_number",
    "publication_date", "update_date", "content_hash", "crawl_timestamp",
    "file_size", "lang", "super_category", "crawl_session",
    "page_number", "top_related", "bottom_related", "themes", "entities", "keywords",
}


class DocumentLevelProcessor:
    """Process individual documents in parallel instead of files."""
    
    def __init__(self, s3_bucket: str, session_id: Optional[str] = None, 
                 milvus_config: Optional[Dict] = None, max_concurrent_documents: int = 15):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.max_concurrent_documents = max_concurrent_documents
        
        self.s3 = boto3.client("s3")
        
        # Thread-local storage for parsers (each thread gets its own)
        self._local = threading.local()
        
        # Shared chunker and embedding service
        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)
        
        # Thread-safe deduplication
        self.seen_hashes: set[str] = set()
        self._seen_hashes_lock = threading.Lock()
        
        # ONNX threading optimization - limit threads per session
        # 15 concurrent docs × 4 ONNX threads = 60 total (reasonable for 64 vCPUs)
        self.onnx_threads_per_document = 4
        
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
        
        self.logger.info(f"Configured for {max_concurrent_documents} concurrent documents, "
                        f"{self.onnx_threads_per_document} ONNX threads per document")

    def _get_processor(self):
        """Get thread-local processor instance with your existing ONNX code."""
        if not hasattr(self._local, 'processor'):
            # Import your existing ONNX parser
            from embeddings.parsers.ONNXPDFParser import ONNXPDFParser
            from embeddings.parsers.PDFRemoveHeaderFooter import HeaderFooterRedactor
            
            # Create optimized PDF pipeline using your existing code
            class OptimizedPDFPipeline:
                def __init__(self, onnx_threads: int):
                    self.onnx_threads = onnx_threads
                    self.redactor = HeaderFooterRedactor(
                        top_k=5, bottom_k=3, win=8,
                        header_th=0.65, footer_th=0.65, rank_th=0.55,
                        pad=2.0, black=True,
                    )
                    # Use your existing ONNXPDFParser
                    self.pdf_parser = ONNXPDFParser()

                def can_process(self, url: str, content_type: str = None) -> bool:
                    url_match = url.lower().endswith(".pdf")
                    content_type_match = content_type and "application/pdf" in content_type
                    return url_match or content_type_match

                def parse(self, content: bytes, url: str, content_type: str):
                    """Parse with controlled ONNX threading using your existing code."""
                    import os
                    import tempfile
                    
                    # Set ONNX threading for this document
                    original_omp = os.environ.get('OMP_NUM_THREADS')
                    os.environ['OMP_NUM_THREADS'] = str(self.onnx_threads)
                    
                    try:
                        # Write original bytes to temp file
                        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as tmp:
                            tmp.write(content)
                            orig_path = tmp.name

                        sanitized_path = None
                        try:
                            # Step 1: redact headers/footers (your existing code)
                            sanitized_path = self.redactor.redact_to_temp(orig_path)
                            # Step 2: parse with your existing ONNXPDFParser
                            return self.pdf_parser.parse_file(sanitized_path)
                        finally:
                            # Cleanup
                            for p in (orig_path, sanitized_path):
                                if p and os.path.exists(p):
                                    try:
                                        os.unlink(p)
                                    except Exception:
                                        pass
                    finally:
                        # Restore environment
                        if original_omp is not None:
                            os.environ['OMP_NUM_THREADS'] = original_omp
                        elif 'OMP_NUM_THREADS' in os.environ:
                            del os.environ['OMP_NUM_THREADS']
            
            # Create thread-local processor with your existing parsers
            self._local.processor = DocumentProcessor(parsers=[
                EurlexHTMLParser(), 
                CSSFHTMLParser(), 
                OptimizedPDFPipeline(self.onnx_threads_per_document)
            ])
        return self._local.processor

    # ===== Utility Methods (same as before) =====
    
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
            self.logger.error(f"Error listing metadata files for session {session_id}: {e}")
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

    # ===== Metadata Processing (same as before) =====
    
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

    def hash_document(self, doc: Document) -> str:
        base = (doc.page_content + str(doc.metadata.get("url", "")) + str(doc.metadata.get("page_number", 0))).encode("utf-8")
        return hashlib.sha256(base).hexdigest()

    # ===== NEW: Document-Level Processing =====
    
    def process_single_document(self, document_info: Tuple[dict, Dict[str, Any]]) -> List[Document]:
        """Process a single document. This is what runs in parallel."""
        file_info, base_metadata = document_info
        
        try:
            s3_uri = file_info.get("s3_uri")
            original_url = file_info.get("url")
            content_type = file_info.get("content_type", "application/octet-stream")
            
            if not s3_uri:
                return []

            self.logger.info(f"Processing document: {original_url}")
            
            # Download document
            content = self.download_document_from_s3(s3_uri)
            if not content:
                return []

            # Process with thread-local processor (includes ONNX threading control)
            processor = self._get_processor()
            elements = processor.process(content, original_url, content_type)

            # Chunk
            chunked_docs = self.chunker.chunk_document(elements, original_url)

            # Apply metadata to each chunk
            processed_docs = []
            for doc in chunked_docs:
                pn = self._extract_page_number(doc)
                doc.metadata = self.flatten_metadata_for_search(base_metadata, page_number=pn)
                processed_docs.append(doc)
                
            self.logger.info(f"Completed document: {original_url} - {len(processed_docs)} chunks")
            return processed_docs

        except Exception as e:
            self.logger.error(f"Error processing document {file_info.get('url','unknown')}: {e}")
            return []

    def collect_all_documents_from_metadata_files(self, metadata_files: List[str], session_id: str) -> List[Tuple[dict, Dict[str, Any]]]:
        """Collect all individual documents from all metadata files."""
        all_documents = []
        
        for metadata_file in metadata_files:
            try:
                metadata = self.read_metadata_from_s3(metadata_file)
                if not metadata:
                    continue
                    
                metadata["crawl_session"] = session_id
                
                # Extract all documents from this metadata file
                for file_info in metadata.get("top_related", []):
                    all_documents.append((file_info, metadata))
                    
            except Exception as e:
                self.logger.error(f"Error reading metadata {metadata_file}: {e}")
                
        return all_documents

    def process_documents_in_parallel(self, all_documents: List[Tuple[dict, Dict[str, Any]]]) -> List[Document]:
        """Process documents in parallel - this is the key change!"""
        
        all_processed_docs = []
        total_documents = len(all_documents)
        
        self.logger.info(f"Processing {total_documents} documents with {self.max_concurrent_documents} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_documents) as executor:
            # Submit all document processing tasks
            future_to_doc = {
                executor.submit(self.process_single_document, doc_info): doc_info
                for doc_info in all_documents
            }
            
            completed_count = 0
            
            # Collect results as they complete
            for future in as_completed(future_to_doc):
                doc_info = future_to_doc[future]
                try:
                    docs = future.result()
                    all_processed_docs.extend(docs)
                    completed_count += 1
                    
                    if completed_count % 10 == 0:  # Progress logging
                        self.logger.info(f"Completed {completed_count}/{total_documents} documents")
                        
                except Exception as e:
                    file_info, _ = doc_info
                    self.logger.error(f"Failed to process document {file_info.get('url', 'unknown')}: {e}")

        self.logger.info(f"Completed all {total_documents} documents, generated {len(all_processed_docs)} chunks")
        return all_processed_docs

    def store_documents_in_milvus(self, documents: List[Document]) -> Dict[str, Any]:
        """Store documents in Milvus with thread-safe deduplication."""
        if not documents:
            return {"count": 0, "milvus_ids": []}

        new_docs, texts, metas = [], [], []

        for d in documents:
            meta = self._ensure_required_fields(self._filter_to_schema(dict(d.metadata or {})), d.page_content)

            # Thread-safe deduplication
            doc_hash = self.hash_document(Document(page_content=d.page_content, metadata=meta))
            with self._seen_hashes_lock:
                if doc_hash in self.seen_hashes:
                    continue
                self.seen_hashes.add(doc_hash)
                
            meta["doc_id"] = doc_hash

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
        """Main processing method - now with document-level parallelism."""
        self.logger.info(f"Processing session: {session_id}")
        
        # Get all metadata files
        metadata_files = self.get_session_metadata_files(session_id)
        self.logger.info(f"Found {len(metadata_files)} metadata files")

        if not metadata_files:
            self.logger.warning(f"No metadata files found for session {session_id}")
            return {"processed": 0, "stored": 0, "errors": 0}

        # Collect all individual documents from all metadata files
        all_documents = self.collect_all_documents_from_metadata_files(metadata_files, session_id)
        self.logger.info(f"Collected {len(all_documents)} total documents to process")

        # Process all documents in parallel (THIS IS THE KEY CHANGE!)
        start_time = time.time()
        processed_docs = self.process_documents_in_parallel(all_documents)
        processing_time = time.time() - start_time
        
        # Store all results
        result = self.store_documents_in_milvus(processed_docs)
        
        summary = {
            "session_id": session_id,
            "metadata_files": len(metadata_files),
            "total_documents": len(all_documents),
            "processed_chunks": len(processed_docs),
            "stored": result["count"],
            "processing_time_seconds": processing_time,
            "documents_per_minute": (len(all_documents) / processing_time) * 60 if processing_time > 0 else 0
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
    S3_BUCKET = "cssf-crawl"
    SESSION_ID = None
    MILVUS_CONFIG = {
        "host": "54.217.166.223",
        "port": "19530",
        "collection_name": "cssf_documents_final_final_CGDEMO4",
        "connection_args": {"host": "54.217.166.223", "port": "19530"},
    }

    # Create processor with document-level parallelism
    processor = DocumentLevelProcessor(
        s3_bucket=S3_BUCKET,
        session_id=SESSION_ID,
        milvus_config=MILVUS_CONFIG,
        max_concurrent_documents=15,  # 15 documents in parallel!
    )

    sessions = processor.list_sessions()
    print(f"Available sessions: {sessions}")

    most_recent = processor.get_most_recent_session()
    if most_recent:
        print(f"Processing session: {most_recent}")
        result = processor.process_session(most_recent)
        print(f"Processing complete: {result}")
        print(f"Throughput: {result.get('documents_per_minute', 0):.1f} documents/minute")
    else:
        print("No sessions found to process")


if __name__ == "__main__":
    main()