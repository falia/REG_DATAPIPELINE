import json
import boto3
import hashlib
import logging
import threading
import time
import os
import psutil
import multiprocessing as mp
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# DISABLE ALL NOISE LOGGING
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('unstructured').setLevel(logging.CRITICAL)
logging.getLogger('unstructured_inference').setLevel(logging.CRITICAL)
logging.getLogger('embeddings').setLevel(logging.CRITICAL)

from embeddings.embedding_provider.embedding_provider import EmbeddingService
from embeddings.chunker.document_chunker import DocumentChunker
from embeddings.parsers.parser import EurlexHTMLParser, CSSFHTMLParser, DocumentProcessor


# ISOLATED ONNX WORKER FUNCTION (runs in separate process)
def isolated_onnx_worker_init(worker_id: int, cpu_cores: List[int]):
    """Initialize isolated ONNX worker process with dedicated CPU cores."""
    global worker_id_global, onnx_parser, redactor, chunker
    
    # Set CPU affinity for this worker
    p = psutil.Process()
    p.cpu_affinity(cpu_cores)
    
    # Set isolated environment
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['ONNX_WORKER_ID'] = str(worker_id)
    
    # Create worker-specific temp directory
    worker_temp = f"/tmp/onnx_worker_{worker_id}"
    os.makedirs(worker_temp, exist_ok=True)
    os.environ['TMPDIR'] = worker_temp
    
    # Initialize ONNX components in this process
    from embeddings.parsers.ONNXPDFParser import ONNXPDFParser
    from embeddings.parsers.PDFRemoveHeaderFooter import HeaderFooterRedactor
    from embeddings.chunker.document_chunker import DocumentChunker
    
    worker_id_global = worker_id
    onnx_parser = ONNXPDFParser()
    redactor = HeaderFooterRedactor(
        top_k=5, bottom_k=3, win=8,
        header_th=0.65, footer_th=0.65, rank_th=0.55,
        pad=2.0, black=True,
    )
    chunker = DocumentChunker(max_chunk_size=1800, overlap=200)
    
    print(f"🔧 ONNX Worker {worker_id} initialized on cores {cpu_cores} (PID: {os.getpid()})")


def process_pdf_in_isolated_worker(args):
    """Process PDF in isolated worker process - this runs in separate process."""
    pdf_content, url, metadata = args
    
    try:
        # Use global variables initialized in worker
        global worker_id_global, onnx_parser, redactor, chunker
        
        print(f"🟢 Worker {worker_id_global}: Processing {url[-40:]} (PID: {os.getpid()})")
        
        # Process PDF with isolated ONNX
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_content)
            temp_path = tmp.name
        
        try:
            # Redact and parse with isolated ONNX session
            sanitized_path = redactor.redact_to_temp(temp_path)
            elements = onnx_parser.parse_file(sanitized_path)
            
            # Chunk in the same process
            docs = chunker.chunk_document(elements, url)
            
            # Convert to serializable format with proper metadata extraction
            result = []
            for doc in docs:
                # Extract page number properly
                pn = 0
                try:
                    if hasattr(doc, 'metadata') and doc.metadata:
                        pn = int(doc.metadata.get("page_number", 0))
                except:
                    pn = 0
                
                result.append({
                    'content': doc.page_content,
                    'metadata': dict(doc.metadata) if hasattr(doc, 'metadata') and doc.metadata else {},
                    'page_number': pn  # Pass page number separately for proper handling
                })
            
            print(f"✅ Worker {worker_id_global}: Completed {url[-40:]} -> {len(result)} chunks")
            return result
            
        finally:
            # Cleanup temp files
            for path in [temp_path, sanitized_path if 'sanitized_path' in locals() else None]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
                        
    except Exception as e:
        print(f"❌ Worker {worker_id_global}: Error processing {url}: {e}")
        return []
    finally:
        # Cleanup worker temp directory periodically
        import random
        if random.randint(1, 10) == 1:  # 10% chance to cleanup
            worker_temp = f"/tmp/onnx_worker_{worker_id_global}"
            if os.path.exists(worker_temp):
                import shutil
                for f in os.listdir(worker_temp):
                    try:
                        os.unlink(os.path.join(worker_temp, f))
                    except:
                        pass


class ParallelismTracker:
    """Track parallelism across processes."""
    
    def __init__(self):
        self.active_threads = set()
        self.lock = threading.Lock()
        self.start_times = {}
        self.completion_times = {}
        self.max_concurrent = 0
        self.total_processed = 0
        
    def start_doc(self, doc_url: str, worker_id: int = None):
        thread_id = threading.current_thread().ident
        with self.lock:
            self.active_threads.add(thread_id)
            self.start_times[doc_url] = time.time()
            current_active = len(self.active_threads)
            self.max_concurrent = max(self.max_concurrent, current_active)
            
            worker_info = f"W{worker_id}" if worker_id is not None else ""
            print(f"🎯 SUBMIT [{current_active:2d}] {doc_url[-45:]:45s} ({worker_info})")
    
    def finish_doc(self, doc_url: str, chunk_count: int = 0):
        thread_id = threading.current_thread().ident
        with self.lock:
            self.active_threads.discard(thread_id)
            self.completion_times[doc_url] = time.time()
            duration = time.time() - self.start_times.get(doc_url, 0)
            current_active = len(self.active_threads)
            self.total_processed += 1
            
            print(f"✅ RESULT [{current_active:2d}] {doc_url[-45:]:45s} ({duration:4.1f}s, {chunk_count} chunks) [Total: {self.total_processed}]")


class ProcessIsolatedONNXProcessor:
    """Your processor modified to use separate processes for ONNX."""
    
    def __init__(self, s3_bucket: str, session_id: Optional[str] = None, 
                 milvus_config: Optional[Dict] = None, max_concurrent_documents: int = 15,
                 onnx_workers: int = 8, batch_size: int = 30):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.max_concurrent_documents = max_concurrent_documents
        self.batch_size = batch_size
        self.onnx_workers = onnx_workers
        
        self.s3 = boto3.client("s3")
        self.tracker = ParallelismTracker()
        
        # Create isolated ONNX worker pool
        self.onnx_executor = self._create_isolated_worker_pool()
        
        # Thread-local storage for non-PDF parsers (runs in main process)
        self._local = threading.local()
        
        # Shared services (main process)
        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)
        
        # Thread-safe deduplication
        self.seen_hashes: set[str] = set()
        self._seen_hashes_lock = threading.Lock()
        
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
        
        print(f"🚀 Process-isolated processor: {max_concurrent_documents} workers, {onnx_workers} ONNX processes, {batch_size} batch size")

    def _create_isolated_worker_pool(self):
        """Create pool of isolated ONNX worker processes with dedicated CPU cores."""
        
        # Allocate CPU cores to workers (m7i.16xlarge has 64 vCPUs)
        # Reserve first 16 cores for main process, use remaining 48 for ONNX workers
        cores_per_worker = 48 // self.onnx_workers  # 6 cores per worker if 8 workers
        
        worker_configs = []
        for i in range(self.onnx_workers):
            start_core = 16 + (i * cores_per_worker)  # Start from core 16
            end_core = start_core + cores_per_worker - 1
            cpu_cores = list(range(start_core, end_core + 1))
            worker_configs.append((i, cpu_cores))
        
        print(f"🏗️  Creating {self.onnx_workers} isolated ONNX workers:")
        for worker_id, cores in worker_configs:
            print(f"   Worker {worker_id}: CPU cores {cores[0]}-{cores[-1]}")
        
        # Create process pool with initialization
        def init_worker_wrapper(config):
            worker_id, cpu_cores = config
            isolated_onnx_worker_init(worker_id, cpu_cores)
        
        executor = ProcessPoolExecutor(
            max_workers=self.onnx_workers,
            initializer=init_worker_wrapper,
            initargs=worker_configs
        )
        
        return executor

    def _get_non_pdf_processor(self):
        """Get thread-local processor for non-PDF documents (main process)."""
        if not hasattr(self._local, 'processor'):
            self._local.processor = DocumentProcessor(parsers=[
                EurlexHTMLParser(), 
                CSSFHTMLParser()
            ])
        return self._local.processor

    # Utility methods (same as your original)
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

    def process_single_document(self, document_info: Tuple[dict, Dict[str, Any]]) -> List[Document]:
        """Process single document using isolated ONNX workers."""
        file_info, base_metadata = document_info
        
        s3_uri = file_info.get("s3_uri")
        original_url = file_info.get("url", "unknown")
        content_type = file_info.get("content_type", "application/octet-stream")
        
        # Check if it's a PDF
        is_pdf = (original_url.lower().endswith(".pdf") or 
                 (content_type and "application/pdf" in content_type.lower()))
        
        try:
            if not s3_uri:
                return []

            content = self.download_document_from_s3(s3_uri)
            if not content:
                return []

            if is_pdf:
                # Use isolated ONNX worker process for PDFs
                self.tracker.start_doc(original_url)
                
                # Submit to isolated worker process
                future = self.onnx_executor.submit(
                    process_pdf_in_isolated_worker, 
                    (content, original_url, base_metadata)
                )
                
                # Get result from isolated process
                serialized_docs = future.result()
                
                # Convert back to Document objects with proper metadata
                processed_docs = []
                for doc_data in serialized_docs:
                    # Apply proper metadata flattening
                    raw_metadata = doc_data['metadata']
                    pn = doc_data.get('page_number', 0)
                    flattened_metadata = self.flatten_metadata_for_search(base_metadata, page_number=pn)
                    
                    doc = Document(
                        page_content=doc_data['content'],
                        metadata=flattened_metadata
                    )
                    processed_docs.append(doc)
                
                return processed_docs
                
            else:
                # Use regular processor for non-PDFs (main process)
                self.tracker.start_doc(original_url)
                processor = self._get_non_pdf_processor()
                elements = processor.process(content, original_url, content_type)
                
                # Chunk in main process
                chunked_docs = self.chunker.chunk_document(elements, original_url)
                
                # Apply metadata in main process with proper flattening
                processed_docs = []
                for doc in chunked_docs:
                    pn = self._extract_page_number(doc)
                    flattened_metadata = self.flatten_metadata_for_search(base_metadata, page_number=pn)
                    doc.metadata = flattened_metadata
                    processed_docs.append(doc)
                
                return processed_docs

        except Exception as e:
            print(f"❌ Error processing {original_url}: {e}")
            return []
        finally:
            self.tracker.finish_doc(original_url, len(processed_docs) if 'processed_docs' in locals() else 0)

    def process_batch(self, batch: List[Tuple[dict, Dict[str, Any]]], batch_num: int) -> List[Document]:
        """Process batch using isolated ONNX workers."""
        print(f"\n🎯 BATCH {batch_num}: Processing {len(batch)} documents")
        print(f"   🏭 {self.onnx_workers} isolated ONNX worker processes available")
        
        all_docs = []
        
        # Use thread pool for coordination (actual ONNX work happens in separate processes)
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
        """Store documents with proper metadata filtering and deduplication."""
        if not documents:
            return {"count": 0, "milvus_ids": []}

        new_docs, texts, metas = [], [], []

        for d in documents:
            # Ensure proper metadata structure and filtering
            meta = self._ensure_required_fields(
                self._filter_to_schema(dict(d.metadata or {})), 
                d.page_content
            )

            # Thread-safe deduplication using proper hash
            doc_hash = self.hash_document(Document(page_content=d.page_content, metadata=meta))
            with self._seen_hashes_lock:
                if doc_hash in self.seen_hashes:
                    continue
                self.seen_hashes.add(doc_hash)

            # Use the hash as doc_id
            meta["doc_id"] = doc_hash

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
        """Stream process with isolated ONNX workers."""
        print(f"\n🌊 STREAMING WITH ISOLATED ONNX PROCESSES")
        print(f"Session: {session_id}")
        print(f"ONNX worker processes: {self.onnx_workers}")
        print(f"Max workers: {self.max_concurrent_documents}")
        print(f"Batch size: {self.batch_size}")
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
            if i % 100 == 0 and i > 0:
                print(f"   📄 Processed {i}/{len(metadata_files)} metadata files...")
            
            try:
                metadata = self.read_metadata_from_s3(metadata_file)
                if not metadata:
                    continue
                    
                metadata["crawl_session"] = session_id
                
                for file_info in metadata.get("top_related", []):
                    current_batch.append((file_info, metadata))
                    
                    if len(current_batch) >= self.batch_size:
                        print(f"\n🔄 Batch {batch_num} ready ({len(current_batch)} documents)")
                        
                        batch_docs = self.process_batch(current_batch, batch_num)
                        
                        # Use batch storage for efficiency
                        store_result = self.store_documents_in_milvus_batch(batch_docs, batch_size=256)
                        total_chunks_stored += store_result["count"]
                        
                        elapsed = time.time() - start_time
                        rate = self.tracker.total_processed / elapsed if elapsed > 0 else 0
                        
                        print(f"📊 BATCH {batch_num} STATS:")
                        print(f"   Processed: {self.tracker.total_processed} docs total")
                        print(f"   Rate: {rate:.1f} docs/sec ({rate*60:.1f} docs/min)")
                        print(f"   Max concurrent: {self.tracker.max_concurrent}")
                        
                        current_batch = []
                        batch_num += 1
                        
            except Exception as e:
                print(f"❌ Error with metadata file {metadata_file}: {e}")
        
        # Process final batch
        if current_batch:
            print(f"\n🔄 Final batch {batch_num} ({len(current_batch)} documents)")
            batch_docs = self.process_batch(current_batch, batch_num)
            store_result = self.store_documents_in_milvus_batch(batch_docs, batch_size=256)
            total_chunks_stored += store_result["count"]
        
        # Final results
        total_time = time.time() - start_time
        final_stats = self.tracker.get_stats()
        
        print(f"\n{'='*60}")
        print(f"🎯 ISOLATED PROCESS PROCESSING COMPLETE!")
        print(f"   ONNX worker processes: {self.onnx_workers}")
        print(f"   Total documents: {final_stats['total_processed']}")
        print(f"   Max concurrent: {final_stats['max_concurrent']}")
        print(f"   Final rate: {(final_stats['total_processed']/total_time)*60:.1f} docs/min")
        print(f"{'='*60}")
        
        return {
            "total_processed": final_stats['total_processed'],
            "total_stored": total_chunks_stored,
            "max_concurrent": final_stats['max_concurrent'],
            "docs_per_minute": (final_stats['total_processed']/total_time)*60,
        }
        
        # Final results
        total_time = time.time() - start_time
        final_stats = self.tracker.get_stats()
        
        print(f"\n{'='*60}")
        print(f"🎯 ISOLATED PROCESS PROCESSING COMPLETE!")
        print(f"   ONNX worker processes: {self.onnx_workers}")
        print(f"   Total documents: {final_stats['total_processed']}")
        print(f"   Max concurrent: {final_stats['max_concurrent']}")
        print(f"   Final rate: {(final_stats['total_processed']/total_time)*60:.1f} docs/min")
        print(f"{'='*60}")
        
    def store_documents_in_milvus_batch(self, documents: List[Document], batch_size: int = 256) -> Dict[str, Any]:
        """Store documents in batches with proper metadata filtering."""
        if not documents:
            return {"count": 0, "milvus_ids": []}

        total_stored = 0
        all_ids = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            result = self.store_documents_in_milvus(batch)
            total_stored += result.get("count", 0)
            all_ids.extend(result.get("milvus_ids", []))
            
            if i % (batch_size * 4) == 0:  # Progress every 4 batches
                print(f"📊 Stored {total_stored} documents so far...")
        
        return {"count": total_stored, "milvus_ids": all_ids}

    def cleanup(self):
        """Cleanup isolated worker processes."""
        if hasattr(self, 'onnx_executor'):
            self.onnx_executor.shutdown(wait=True)
            print("🧹 Cleaned up isolated ONNX worker processes")


def main():
    # IMPORTANT: This is required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    
    S3_BUCKET = "cssf-crawl"
    MILVUS_CONFIG = {
        "host": "54.217.166.223",
        "port": "19530", 
        "collection_name": "cssf_documents_final_final_CGDEMO4",
        "connection_args": {"host": "54.217.166.223", "port": "19530"},
    }

    # Create processor with isolated ONNX workers
    processor = ProcessIsolatedONNXProcessor(
        s3_bucket=S3_BUCKET,
        milvus_config=MILVUS_CONFIG,
        max_concurrent_documents=15,  # Main process coordination threads
        onnx_workers=8,              # Isolated ONNX worker processes
        batch_size=30
    )

    try:
        # Stream process with isolated workers
        result = processor.stream_process_session("20250702_020822")
        
        print(f"\n🎉 FINAL RESULTS:")
        print(f"Rate: {result['docs_per_minute']:.1f} docs/minute")
        print(f"Max concurrent: {result['max_concurrent']}")
        print(f"Success: {'✅ YES' if result['docs_per_minute'] >= 25 else '❌ NO'}")
        
    finally:
        # Always cleanup
        processor.cleanup()


if __name__ == "__main__":
    main()