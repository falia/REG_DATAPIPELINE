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


class ParallelismTracker:
    """Track parallelism across multiple batches."""
    
    def __init__(self):
        self.active_threads = set()
        self.lock = threading.Lock()
        self.start_times = {}
        self.completion_times = {}
        self.max_concurrent = 0
        self.total_processed = 0
        
    def start_doc(self, doc_url: str):
        thread_id = threading.current_thread().ident
        with self.lock:
            self.active_threads.add(thread_id)
            self.start_times[doc_url] = time.time()
            current_active = len(self.active_threads)
            self.max_concurrent = max(self.max_concurrent, current_active)
            
            print(f"🟢 START  [{current_active:2d}] {doc_url[-50:]:50s} (T-{str(thread_id)[-4:]})")
    
    def finish_doc(self, doc_url: str, chunk_count: int = 0):
        thread_id = threading.current_thread().ident
        with self.lock:
            self.active_threads.discard(thread_id)
            self.completion_times[doc_url] = time.time()
            duration = time.time() - self.start_times.get(doc_url, 0)
            current_active = len(self.active_threads)
            self.total_processed += 1
            
            print(f"✅ FINISH [{current_active:2d}] {doc_url[-50:]:50s} ({duration:4.1f}s, {chunk_count} chunks) [Total: {self.total_processed}]")
    
    def get_stats(self):
        return {
            "max_concurrent": self.max_concurrent,
            "total_processed": self.total_processed,
        }


class StreamingBatchProcessor:
    """Process documents in streaming batches - don't wait for all collection!"""
    
    def __init__(self, s3_bucket: str, session_id: Optional[str] = None, 
                 milvus_config: Optional[Dict] = None, max_concurrent_documents: int = 15,
                 batch_size: int = 30):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.max_concurrent_documents = max_concurrent_documents
        self.batch_size = batch_size  # Process when we have this many documents
        
        self.s3 = boto3.client("s3")
        self.tracker = ParallelismTracker()
        
        # Thread-local storage
        self._local = threading.local()
        
        # Shared services
        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)
        
        # Thread-safe deduplication
        self.seen_hashes: set[str] = set()
        self._seen_hashes_lock = threading.Lock()
        
        # ONNX optimization
        self.onnx_threads_per_document = 4
        os.environ['OMP_NUM_THREADS'] = str(self.onnx_threads_per_document)
        
        # Only error logging
        logging.basicConfig(level=logging.ERROR)

        if milvus_config is None:
            milvus_config = {
                "host": "54.217.166.223",
                "port": "19530",
                "collection_name": "cssf_documents_final_final_CGDEMO4",
                "connection_args": {"host": "54.217.166.223", "port": "19530"},
            }

        self.embedding_service = EmbeddingService(
            use_tei=True,
            milvus_config=milvus_config,
            endpoint_name="embedding-endpoint",
            region_name="eu-west-1",
        )
        
        print(f"🚀 Streaming processor: {max_concurrent_documents} workers, {batch_size} batch size")

    def _get_processor(self):
        """Get thread-local processor - SILENT."""
        if not hasattr(self._local, 'processor'):
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
        except:
            return {}

    def download_document_from_s3(self, s3_uri: str) -> bytes:
        try:
            s3_key = s3_uri.replace(f"s3://{self.s3_bucket}/", "")
            resp = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
            return resp["Body"].read()
        except:
            return b""

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
        """Process single document with parallelism tracking."""
        file_info, base_metadata = document_info
        
        s3_uri = file_info.get("s3_uri")
        original_url = file_info.get("url", "unknown")
        content_type = file_info.get("content_type", "application/octet-stream")
        
        self.tracker.start_doc(original_url)
        
        try:
            if not s3_uri:
                return []

            content = self.download_document_from_s3(s3_uri)
            if not content:
                return []

            processor = self._get_processor()
            elements = processor.process(content, original_url, content_type)
            chunked_docs = self.chunker.chunk_document(elements, original_url)

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
            self.tracker.finish_doc(original_url, len(processed_docs) if 'processed_docs' in locals() else 0)

    def process_batch(self, batch: List[Tuple[dict, Dict[str, Any]]], batch_num: int) -> List[Document]:
        """Process a batch of documents in parallel."""
        print(f"\n🎯 BATCH {batch_num}: Processing {len(batch)} documents")
        
        all_docs = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_documents, thread_name_prefix=f"Batch{batch_num}") as executor:
            # Submit all jobs in this batch
            futures = [executor.submit(self.process_single_document, doc_info) for doc_info in batch]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"❌ Batch {batch_num} job failed: {e}")
        
        print(f"✅ BATCH {batch_num} COMPLETE: {len(all_docs)} chunks generated")
        return all_docs

    def store_documents_in_milvus(self, documents: List[Document]) -> Dict[str, Any]:
        """Store documents with thread-safe deduplication."""
        if not documents:
            return {"count": 0, "milvus_ids": []}

        new_docs, texts, metas = [], [], []

        for d in documents:
            # Simple metadata for testing
            meta = {
                "url": d.metadata.get("url", ""),
                "title": d.metadata.get("title", ""),
                "page_number": d.metadata.get("page_number", 0),
                "crawl_session": d.metadata.get("crawl_session", ""),
                "doc_id": hashlib.sha256((d.page_content + d.metadata.get("url", "")).encode()).hexdigest()
            }

            # Thread-safe deduplication
            with self._seen_hashes_lock:
                if meta["doc_id"] in self.seen_hashes:
                    continue
                self.seen_hashes.add(meta["doc_id"])

            new_docs.append(d)
            texts.append(d.page_content)
            metas.append(meta)

        if not new_docs:
            return {"count": 0, "milvus_ids": []}

        try:
            result = self.embedding_service.add_texts_to_store(texts=texts, metadatas=metas)
            print(f"💾 Stored {result['count']} documents in Milvus")
            return result
        except Exception as e:
            print(f"❌ Failed to store in Milvus: {e}")
            return {"count": 0, "milvus_ids": []}

    def stream_process_session(self, session_id: str) -> Dict[str, Any]:
        """STREAMING: Process documents in batches as we collect them."""
        print(f"\n🌊 STREAMING BATCH PROCESSING")
        print(f"Session: {session_id}")
        print(f"Batch size: {self.batch_size}")
        print(f"Workers per batch: {self.max_concurrent_documents}")
        print(f"{'='*60}")
        
        # Get metadata files
        metadata_files = self.get_session_metadata_files(session_id)
        print(f"📂 Found {len(metadata_files)} metadata files")
        
        # Streaming variables
        current_batch = []
        batch_num = 1
        total_chunks_stored = 0
        start_time = time.time()
        
        print(f"\n📋 Starting to collect and process...")
        
        for i, metadata_file in enumerate(metadata_files):
            # Progress every 100 files
            if i % 100 == 0 and i > 0:
                print(f"   📄 Processed {i}/{len(metadata_files)} metadata files...")
            
            try:
                metadata = self.read_metadata_from_s3(metadata_file)
                if not metadata:
                    continue
                    
                metadata["crawl_session"] = session_id
                
                # Add all documents from this metadata file to current batch
                for file_info in metadata.get("top_related", []):
                    current_batch.append((file_info, metadata))
                    
                    # When batch is full, process it!
                    if len(current_batch) >= self.batch_size:
                        print(f"\n🔄 Batch {batch_num} ready ({len(current_batch)} documents)")
                        
                        # Process this batch
                        batch_docs = self.process_batch(current_batch, batch_num)
                        
                        # Store results
                        store_result = self.store_documents_in_milvus(batch_docs)
                        total_chunks_stored += store_result["count"]
                        
                        # Stats
                        elapsed = time.time() - start_time
                        rate = self.tracker.total_processed / elapsed if elapsed > 0 else 0
                        
                        print(f"📊 BATCH {batch_num} STATS:")
                        print(f"   Processed: {self.tracker.total_processed} docs total")
                        print(f"   Stored: {total_chunks_stored} chunks total")
                        print(f"   Rate: {rate:.1f} docs/sec ({rate*60:.1f} docs/min)")
                        print(f"   Max concurrent: {self.tracker.max_concurrent}")
                        
                        # Reset for next batch
                        current_batch = []
                        batch_num += 1
                        
            except Exception as e:
                print(f"❌ Error with metadata file {metadata_file}: {e}")
        
        # Process final partial batch
        if current_batch:
            print(f"\n🔄 Final batch {batch_num} ({len(current_batch)} documents)")
            batch_docs = self.process_batch(current_batch, batch_num)
            store_result = self.store_documents_in_milvus(batch_docs)
            total_chunks_stored += store_result["count"]
        
        # Final results
        total_time = time.time() - start_time
        final_stats = self.tracker.get_stats()
        
        print(f"\n{'='*60}")
        print(f"🎯 STREAMING PROCESSING COMPLETE!")
        print(f"   Total documents: {final_stats['total_processed']}")
        print(f"   Total chunks: {total_chunks_stored}")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Max concurrent: {final_stats['max_concurrent']}")
        print(f"   Final rate: {final_stats['total_processed']/total_time:.1f} docs/sec")
        print(f"   Final rate: {(final_stats['total_processed']/total_time)*60:.1f} docs/min")
        print(f"   Success: {'✅ YES' if final_stats['max_concurrent'] >= 12 else '❌ NO'}")
        print(f"{'='*60}")
        
        return {
            "total_processed": final_stats['total_processed'],
            "total_stored": total_chunks_stored,
            "total_time": total_time,
            "max_concurrent": final_stats['max_concurrent'],
            "docs_per_minute": (final_stats['total_processed']/total_time)*60,
            "batches_processed": batch_num
        }


def main():
    S3_BUCKET = "cssf-crawl"
    MILVUS_CONFIG = {
        "host": "54.217.166.223",
        "port": "19530", 
        "collection_name": "cssf_documents_final_final_CGDEMO4",
        "connection_args": {"host": "54.217.166.223", "port": "19530"},
    }

    # Create streaming processor
    processor = StreamingBatchProcessor(
        s3_bucket=S3_BUCKET,
        milvus_config=MILVUS_CONFIG,
        max_concurrent_documents=15,
        batch_size=30  # Process every 30 documents
    )

    # Stream process the entire session
    result = processor.stream_process_session("20250702_020822")
    
    print(f"\n🎉 FINAL RESULTS:")
    print(f"Processed: {result['total_processed']} documents")
    print(f"Rate: {result['docs_per_minute']:.1f} docs/minute")
    print(f"Parallelism: {result['max_concurrent']} max concurrent")


if __name__ == "__main__":
    main()