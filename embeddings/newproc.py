import json
import boto3
import hashlib
import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import os

# DISABLE ALL NOISE LOGGING
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('s3transfer').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('unstructured').setLevel(logging.CRITICAL)
logging.getLogger('unstructured_inference').setLevel(logging.CRITICAL)
logging.getLogger('embeddings').setLevel(logging.CRITICAL)

from embeddings.embedding_provider.embedding_provider import EmbeddingService
from embeddings.chunker.document_chunker import DocumentChunker
from embeddings.parsers.parser import EurlexHTMLParser, CSSFHTMLParser, DocumentProcessor


class ParallelismDebugger:
    """Debug tool to track actual parallelism - CLEAN OUTPUT ONLY."""
    
    def __init__(self):
        self.active_threads = set()
        self.active_lock = threading.Lock()
        self.start_times = {}
        self.completion_times = {}
        self.max_concurrent = 0
        
    def start_processing(self, doc_url: str):
        thread_id = threading.current_thread().ident
        with self.active_lock:
            self.active_threads.add(thread_id)
            self.start_times[doc_url] = time.time()
            current_active = len(self.active_threads)
            self.max_concurrent = max(self.max_concurrent, current_active)
            
            print(f"🟢 START  [{current_active:2d}] {doc_url[-50:]:50s} (T-{str(thread_id)[-4:]})")
    
    def finish_processing(self, doc_url: str):
        thread_id = threading.current_thread().ident
        with self.active_lock:
            self.active_threads.discard(thread_id)
            self.completion_times[doc_url] = time.time()
            duration = self.completion_times[doc_url] - self.start_times.get(doc_url, 0)
            current_active = len(self.active_threads)
            
            print(f"✅ FINISH [{current_active:2d}] {doc_url[-50:]:50s} ({duration:4.1f}s)")
    
    def get_stats(self):
        return {
            "max_concurrent": self.max_concurrent,
            "total_processed": len(self.completion_times),
            "avg_duration": sum(self.completion_times[url] - self.start_times[url] 
                               for url in self.completion_times) / len(self.completion_times) if self.completion_times else 0
        }


class FixedDocumentProcessor:
    """Your processor with CLEAN parallelism logging only."""
    
    def __init__(self, s3_bucket: str, session_id: Optional[str] = None, 
                 milvus_config: Optional[Dict] = None, max_concurrent_documents: int = 15):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.max_concurrent_documents = max_concurrent_documents
        
        self.s3 = boto3.client("s3")
        
        # ONLY parallelism debugging
        self.debugger = ParallelismDebugger()
        
        # Thread-local storage for parsers
        self._local = threading.local()
        
        # Shared services
        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)
        
        # Thread-safe deduplication
        self.seen_hashes: set[str] = set()
        self._seen_hashes_lock = threading.Lock()
        
        # ONNX optimization: 4 threads per document
        self.onnx_threads_per_document = 4
        
        # Set global environment for ONNX
        os.environ['OMP_NUM_THREADS'] = str(self.onnx_threads_per_document)
        
        # DISABLE regular logging - only errors
        logging.basicConfig(level=logging.ERROR)

        if milvus_config is None:
            milvus_config = {
                "host": "54.217.166.223",
                "port": "19530",
                "collection_name": "cssf_documents_final",
                "connection_args": {"host": "54.217.166.223", "port": "19530"},
            }

        self.embedding_service = EmbeddingService(
            use_tei=True,
            milvus_config=milvus_config,
            endpoint_name="embedding-endpoint",
            region_name="eu-west-1",
        )
        
        # Only this print
        print(f"🚀 Configured for {max_concurrent_documents} concurrent documents")

    def _get_processor(self):
        """Get thread-local processor with your existing ONNX code - SILENT."""
        if not hasattr(self._local, 'processor'):
            # Import your existing components
            from embeddings.parsers.ONNXPDFParser import ONNXPDFParser
            from embeddings.parsers.PDFRemoveHeaderFooter import HeaderFooterRedactor
            
            class ThreadSafePDFPipeline:
                def __init__(self):
                    self.redactor = HeaderFooterRedactor(
                        top_k=5, bottom_k=3, win=8,
                        header_th=0.65, footer_th=0.65, rank_th=0.55,
                        pad=2.0, black=True,
                    )
                    self.pdf_parser = ONNXPDFParser()

                def can_process(self, url: str, content_type: str = None) -> bool:
                    return (url.lower().endswith(".pdf") or 
                           (content_type and "application/pdf" in content_type))

                def parse(self, content: bytes, url: str, content_type: str):
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as tmp:
                        tmp.write(content)
                        orig_path = tmp.name

                    sanitized_path = None
                    try:
                        sanitized_path = self.redactor.redact_to_temp(orig_path)
                        return self.pdf_parser.parse_file(sanitized_path)
                    finally:
                        for p in (orig_path, sanitized_path):
                            if p and os.path.exists(p):
                                try:
                                    os.unlink(p)
                                except:
                                    pass
            
            self._local.processor = DocumentProcessor(parsers=[
                EurlexHTMLParser(), 
                CSSFHTMLParser(), 
                ThreadSafePDFPipeline()
            ])
            
        return self._local.processor

    # Utility methods - SILENT
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
            print(f"❌ Error listing metadata files: {e}")
            return []

    def read_metadata_from_s3(self, s3_key: str) -> Dict[str, Any]:
        try:
            resp = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except Exception as e:
            print(f"❌ Error reading metadata {s3_key}: {e}")
            return {}

    def download_document_from_s3(self, s3_uri: str) -> bytes:
        try:
            s3_key = s3_uri.replace(f"s3://{self.s3_bucket}/", "")
            resp = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
            return resp["Body"].read()
        except Exception as e:
            print(f"❌ Error downloading {s3_uri}: {e}")
            return b""

    # Metadata processing methods - SILENT
    def flatten_metadata_for_search(self, metadata: dict, page_number: int = None) -> dict:
        return {
            "url": metadata.get("url", ""),
            "title": metadata.get("title", ""),
            "crawl_session": metadata.get("crawl_session", ""),
            "page_number": page_number or 0
        }

    def _extract_page_number(self, doc: Document) -> int:
        try:
            return int(doc.metadata.get("page_number", 0)) if doc.metadata else 0
        except:
            return 0

    def process_single_document(self, document_info: Tuple[dict, Dict[str, Any]]) -> List[Document]:
        """Process a single document with ONLY parallelism debugging."""
        file_info, base_metadata = document_info
        
        s3_uri = file_info.get("s3_uri")
        original_url = file_info.get("url", "unknown")
        content_type = file_info.get("content_type", "application/octet-stream")
        
        # ONLY parallelism tracking
        self.debugger.start_processing(original_url)
        
        try:
            if not s3_uri:
                return []

            # Download document - SILENT
            content = self.download_document_from_s3(s3_uri)
            if not content:
                return []

            # Process with thread-local processor - SILENT
            processor = self._get_processor()
            elements = processor.process(content, original_url, content_type)

            # Chunk - SILENT
            chunked_docs = self.chunker.chunk_document(elements, original_url)

            # Apply metadata - SILENT
            processed_docs = []
            for doc in chunked_docs:
                pn = self._extract_page_number(doc)
                doc.metadata = self.flatten_metadata_for_search(base_metadata, page_number=pn)
                processed_docs.append(doc)
                
            return processed_docs

        except Exception as e:
            print(f"❌ Error processing {original_url}: {e}")
            return []
        finally:
            # ONLY parallelism tracking
            self.debugger.finish_processing(original_url)

    def collect_all_documents(self, metadata_files: List[str], session_id: str) -> List[Tuple[dict, Dict[str, Any]]]:
        """Collect all documents - CLEAN OUTPUT."""
        all_documents = []
        pdf_count = 0
        
        print(f"📂 Processing {len(metadata_files)} metadata files...")
        
        for metadata_file in metadata_files:
            try:
                metadata = self.read_metadata_from_s3(metadata_file)
                if not metadata:
                    continue
                    
                metadata["crawl_session"] = session_id
                
                for file_info in metadata.get("top_related", []):
                    url = file_info.get("url", "").lower()
                    content_type = file_info.get("content_type", "")
                    
                    # Count PDFs
                    if url.endswith(".pdf") or "pdf" in content_type.lower():
                        pdf_count += 1
                    
                    all_documents.append((file_info, metadata))
                        
            except Exception as e:
                print(f"❌ Error reading metadata {metadata_file}: {e}")
                
        print(f"📊 Collected {len(all_documents)} documents ({pdf_count} PDFs)")
        return all_documents

    def process_documents_in_parallel(self, all_documents: List[Tuple[dict, Dict[str, Any]]]) -> List[Document]:
        """Process documents with ONLY parallelism tracking."""
        
        print(f"\n🎯 Starting parallel processing:")
        print(f"   📋 Documents: {len(all_documents)}")
        print(f"   🔧 Workers: {self.max_concurrent_documents}")
        print(f"   {'='*60}")
        
        all_processed_docs = []
        
        with ThreadPoolExecutor(
            max_workers=self.max_concurrent_documents,
            thread_name_prefix="Worker"
        ) as executor:
            
            # Submit ALL jobs at once
            future_to_doc = {}
            for i, doc_info in enumerate(all_documents):
                future = executor.submit(self.process_single_document, doc_info)
                future_to_doc[future] = (i, doc_info)
            
            # Wait for completion - SILENT except parallelism tracking
            completed_count = 0
            start_time = time.time()
            
            for future in as_completed(future_to_doc):
                i, doc_info = future_to_doc[future]
                file_info, _ = doc_info
                url = file_info.get("url", "unknown")
                
                try:
                    docs = future.result()
                    all_processed_docs.extend(docs)
                    completed_count += 1
                        
                except Exception as e:
                    print(f"❌ Failed document {i}: {url[:50]}... - {e}")

        # Print ONLY summary stats
        stats = self.debugger.get_stats()
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"📊 PARALLELISM RESULTS:")
        print(f"   Max concurrent: {stats['max_concurrent']}")
        print(f"   Documents: {len(all_documents)}")
        print(f"   Chunks: {len(all_processed_docs)}")
        print(f"   Time: {total_time:.1f}s")
        print(f"   Rate: {len(all_documents)/total_time:.1f} docs/sec")
        print(f"   Rate: {(len(all_documents)/total_time)*60:.1f} docs/minute")
        print(f"   Success: {'✅ YES' if stats['max_concurrent'] >= 12 else '❌ NO'}")
        print(f"{'='*60}")
        
        return all_processed_docs

    def test_parallelism(self, session_id: str, limit_documents: int = 20):
        """Test parallelism - CLEAN OUTPUT."""
        print(f"🧪 TESTING PARALLELISM")
        print(f"Target: {self.max_concurrent_documents} concurrent documents")
        
        # Get metadata files
        metadata_files = self.get_session_metadata_files(session_id)
        print(f"Found {len(metadata_files)} metadata files")

        # Collect documents (limited for testing)
        all_documents = self.collect_all_documents(metadata_files, session_id)
        
        # Limit for testing
        test_documents = all_documents[:limit_documents]
        print(f"Testing with {len(test_documents)} documents")
        
        # Process with clean tracking
        start_time = time.time()
        processed_docs = self.process_documents_in_parallel(test_documents)
        total_time = time.time() - start_time
        
        return {
            "documents_processed": len(test_documents),
            "chunks_generated": len(processed_docs),
            "total_time": total_time,
            "docs_per_second": len(test_documents) / total_time,
            "docs_per_minute": (len(test_documents) / total_time) * 60
        }


def main():
    S3_BUCKET = "cssf-crawl"
    MILVUS_CONFIG = {
        "host": "54.217.166.223",
        "port": "19530", 
        "collection_name": "cssf_documents_final_final_CGDEMO4",
        "connection_args": {"host": "54.217.166.223", "port": "19530"},
    }

    # Create processor 
    processor = FixedDocumentProcessor(
        s3_bucket=S3_BUCKET,
        milvus_config=MILVUS_CONFIG,
        max_concurrent_documents=15
    )

    # Test parallelism with your session
    result = processor.test_parallelism("20250702_020822", limit_documents=30)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Achieved: {result['docs_per_minute']:.1f} documents/minute")
    print(f"Target was: 30 documents/minute")
    print(f"Success: {'✅ YES' if result['docs_per_minute'] >= 25 else '❌ NO'}")


if __name__ == "__main__":
    main()