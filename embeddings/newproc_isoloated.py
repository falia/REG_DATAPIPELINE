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

# Global worker assignment tracking
_worker_assignments = {}
_assignment_lock = threading.Lock()

# DISABLE ALL NOISE LOGGING
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('unstructured').setLevel(logging.CRITICAL)
logging.getLogger('unstructured_inference').setLevel(logging.CRITICAL)
logging.getLogger('embeddings').setLevel(logging.CRITICAL)

from embeddings.embedding_provider.embedding_provider import EmbeddingService
from embeddings.chunker.document_chunker import DocumentChunker
from embeddings.parsers.parser import EurlexHTMLParser, CSSFHTMLParser, DocumentProcessor


# =================== MODULE-LEVEL FUNCTIONS FOR MULTIPROCESSING ===================

def isolated_onnx_worker_init(worker_id: int, cpu_cores: List[int]):
    """Enhanced worker initialization with better resource isolation."""
    global worker_id_global, onnx_parser, redactor, chunker
    
    try:
        # Set CPU affinity for this worker
        p = psutil.Process()
        p.cpu_affinity(cpu_cores)
        print(f"🔧 Worker {worker_id}: Set CPU affinity to cores {cpu_cores}")
        
        # More aggressive resource limits per worker
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1' 
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['ONNX_WORKER_ID'] = str(worker_id)
        
        # Set memory limits (3GB per worker to be safe)
        import resource
        memory_limit = 3 * 1024 * 1024 * 1024  # 3GB
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        
        # Worker-specific temp directory
        worker_temp = f"/tmp/onnx_worker_{worker_id}_{os.getpid()}"
        os.makedirs(worker_temp, exist_ok=True)
        os.environ['TMPDIR'] = worker_temp
        
        # Initialize ONNX components
        from embeddings.parsers.ONNXPDFParser import ONNXPDFParser
        from embeddings.parsers.PDFRemoveHeaderFooter import HeaderFooterRedactor
        from embeddings.chunker.document_chunker import DocumentChunker
        
        worker_id_global = worker_id
        
        # More conservative ONNX settings to reduce processing time
        onnx_parser = ONNXPDFParser()
        redactor = HeaderFooterRedactor(
            top_k=2, bottom_k=1, win=4,  # Very conservative settings
            header_th=0.7, footer_th=0.7, rank_th=0.6,  # Higher thresholds
            pad=1.0, black=True,  # Reduced padding
        )
        chunker = DocumentChunker(max_chunk_size=1800, overlap=200)
        
        print(f"🔧 ONNX Worker {worker_id} initialized on cores {cpu_cores} (PID: {os.getpid()})")
        
    except Exception as e:
        print(f"❌ Worker {worker_id} initialization failed: {e}")
        raise


def get_worker_config_for_process(all_configs):
    """Assign worker config based on process ID to ensure uniqueness."""
    pid = os.getpid()
    
    with _assignment_lock:
        # If this process already has an assignment, return it
        if pid in _worker_assignments:
            return _worker_assignments[pid]
        
        # Find the next available worker config
        assigned_worker_ids = {config[0] for config in _worker_assignments.values()}
        
        for worker_id, cpu_cores in all_configs:
            if worker_id not in assigned_worker_ids:
                _worker_assignments[pid] = (worker_id, cpu_cores)
                print(f"🔧 Assigned Worker {worker_id} to Process {pid}")
                return (worker_id, cpu_cores)
        
        # If all workers assigned, cycle through (shouldn't happen with proper pool size)
        fallback_config = all_configs[len(_worker_assignments) % len(all_configs)]
        _worker_assignments[pid] = fallback_config
        return fallback_config


def process_pdf_with_fixed_worker_assignment(args):
    """Process PDF with proper per-process worker assignment."""
    pdf_content, url, metadata, all_worker_configs = args
    
    # Get or assign worker config for this specific process
    if not hasattr(process_pdf_with_fixed_worker_assignment, '_process_initialized'):
        try:
            # Get unique worker config for this process
            worker_id, cpu_cores = get_worker_config_for_process(all_worker_configs)
            
            print(f"🔧 Process {os.getpid()} initializing as Worker {worker_id} with cores {cpu_cores}")
            
            # Initialize this process with the assigned config
            isolated_onnx_worker_init(worker_id, cpu_cores)
            
            # Mark this process as initialized
            process_pdf_with_fixed_worker_assignment._process_initialized = True
            process_pdf_with_fixed_worker_assignment._worker_id = worker_id
            
            print(f"✅ Process {os.getpid()} successfully initialized as Worker {worker_id}")
            
        except Exception as e:
            print(f"❌ Failed to initialize process {os.getpid()}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # Process the PDF
    try:
        global worker_id_global, onnx_parser, redactor, chunker
        
        if not all([onnx_parser, redactor, chunker]):
            print(f"❌ Process {os.getpid()}: ONNX components not available")
            return []
        
        print(f"🟢 Worker {worker_id_global}: Processing {url[-40:]} (PID: {os.getpid()})")
        
        # File size check
        if len(pdf_content) > 50 * 1024 * 1024:  # 50MB limit
            print(f"⚠️  Worker {worker_id_global}: File too large ({len(pdf_content)/1024/1024:.1f}MB), skipping")
            return []
        
        start_time = time.time()
        
        # Process PDF with better error handling
        import tempfile
        import gc
        
        temp_path = None
        sanitized_path = None
        
        try:
            # Write PDF to temp file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_content)
                temp_path = tmp.name
            
            # Clear content from memory immediately
            del pdf_content
            gc.collect()
            
            # Process with ONNX - suppress PDF warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                sanitized_path = redactor.redact_to_temp(temp_path)
                elements = onnx_parser.parse_file(sanitized_path)
            
            # Chunk the document
            docs = chunker.chunk_document(elements, url)
            
            # Clear intermediate data
            del elements
            gc.collect()
            
            # Convert to serializable format
            result = []
            for doc in docs:
                pn = 0
                try:
                    if hasattr(doc, 'metadata') and doc.metadata:
                        pn = int(doc.metadata.get("page_number", 0))
                except:
                    pn = 0
                
                result.append({
                    'content': doc.page_content,
                    'metadata': dict(doc.metadata) if hasattr(doc, 'metadata') and doc.metadata else {},
                    'page_number': pn
                })
            
            processing_time = time.time() - start_time
            print(f"✅ Worker {worker_id_global}: Completed {url[-40:]} -> {len(result)} chunks ({processing_time:.1f}s)")
            return result
            
        finally:
            # Aggressive cleanup
            for path in [temp_path, sanitized_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
            gc.collect()
                        
    except Exception as e:
        print(f"❌ Worker {worker_id_global if 'worker_id_global' in globals() else 'Unknown'}: Error processing {url}: {e}")
        # Don't print full traceback for PDF parsing errors
        if "Cannot set gray non-stroke color" not in str(e):
            import traceback
            traceback.print_exc()
        return []


# =================== MAIN CLASSES ===================

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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get final statistics."""
        return {
            "total_processed": self.total_processed,
            "max_concurrent": self.max_concurrent,
            "start_times": self.start_times,
            "completion_times": self.completion_times
        }


class ProcessIsolatedONNXProcessor:
    """Fixed multiprocessing processor for ONNX isolation."""
    
    def __init__(self, s3_bucket: str, session_id: Optional[str] = None, 
                 milvus_config: Optional[Dict] = None, max_concurrent_documents: int = 12,
                 onnx_workers: int = 4, batch_size: int = 20):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.max_concurrent_documents = max_concurrent_documents
        self.batch_size = batch_size
        self.onnx_workers = onnx_workers
        
        self.s3 = boto3.client("s3")
        self.tracker = ParallelismTracker()
        
        # Create worker configs
        self.worker_configs = self._create_worker_configs()
        
        # Create simple process pool WITHOUT initializer
        self.onnx_executor = ProcessPoolExecutor(
            max_workers=self.onnx_workers
            # No initializer, no initargs - let each process initialize itself
        )
        
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
        
        print(f"🚀 Process-isolated processor: {max_concurrent_documents} coordinators, {onnx_workers} ONNX processes, {batch_size} batch size")

    def _create_worker_configs(self):
        """Create well-separated worker configurations."""
        # Give more cores per worker for better isolation
        cores_per_worker = 48 // self.onnx_workers  # 12 cores per worker if 4 workers
        
        worker_configs = []
        for i in range(self.onnx_workers):
            start_core = 16 + (i * cores_per_worker)
            end_core = start_core + cores_per_worker - 1
            cpu_cores = list(range(start_core, min(end_core + 1, 64)))
            worker_configs.append((i, cpu_cores))
        
        print(f"🏗️  Configured {self.onnx_workers} well-isolated ONNX workers:")
        for worker_id, cores in worker_configs:
            print(f"   Worker {worker_id}: CPU cores {cores[0]}-{cores[-1]} ({len(cores)} cores)")
        
        return worker_configs

    def _get_non_pdf_processor(self):
        """Get thread-local processor for non-PDF documents (main process)."""
        if not hasattr(self._local, 'processor'):
            self._local.processor = DocumentProcessor(parsers=[
                EurlexHTMLParser(), 
                CSSFHTMLParser()
            ])
        return self._local.processor

    # Utility methods
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
        """Process document with fixed worker assignment."""
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
                
                try:
                    # Submit with fixed worker assignment
                    future = self.onnx_executor.submit(
                        process_pdf_with_fixed_worker_assignment,
                        (content, original_url, base_metadata, self.worker_configs)
                    )
                    
                    # Shorter timeout to prevent hanging
                    serialized_docs = future.result(timeout=120)  # 2 minutes max per PDF
                    
                except Exception as e:
                    print(f"❌ ONNX processing failed for {original_url}: {e}")
                    return []
                
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

    @staticmethod
    def _extract_page_number(doc: Document) -> int:
        # If the chunker preserved page_number, use it; else 0
        try:
            pn = doc.metadata.get("page_number", 0) if isinstance(doc.metadata, dict) else 0
            return int(pn) if pn is not None else 0
        except Exception:
            return 0

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

    def cleanup(self):
        """Cleanup isolated worker processes."""
        if hasattr(self, 'onnx_executor'):
            self.onnx_executor.shutdown(wait=True)
            print("🧹 Cleaned up isolated ONNX worker processes")

    def flatten_metadata_for_search(self, metadata: dict, page_number: int | None = None) -> dict:
        """Flatten metadata for search with proper field handling."""
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

    def _clamp(self, value: str, max_length: int) -> str:
        """Clamp string to maximum length."""
        if not isinstance(value, str):
            value = str(value) if value is not None else ""
        return value[:max_length] if len(value) > max_length else value

    def _as_json(self, value: Any, default: Any) -> str:
        """Convert value to JSON string safely."""
        try:
            if value is None:
                return json.dumps(default)
            return json.dumps(value)
        except Exception:
            return json.dumps(default)

    def _filter_to_schema(self, metadata: dict) -> dict:
        """Filter metadata to match Milvus schema."""
        # Define allowed fields based on your Milvus collection schema
        allowed_fields = {
            "url", "title", "subtitle", "document_type", "document_number",
            "publication_date", "update_date", "content_hash", "crawl_timestamp",
            "file_size", "lang", "super_category", "crawl_session", "page_number",
            "top_related", "bottom_related", "themes", "entities", "keywords", "doc_id"
        }
        
        return {k: v for k, v in metadata.items() if k in allowed_fields}

    def _ensure_required_fields(self, metadata: dict, content: str) -> dict:
        """Ensure all required fields are present in metadata."""
        # Set defaults for required fields
        metadata.setdefault("url", "")
        metadata.setdefault("title", "")
        metadata.setdefault("page_number", 0)
        metadata.setdefault("crawl_session", self.session_id or "")
        
        # Generate doc_id if not present
        if "doc_id" not in metadata:
            metadata["doc_id"] = hashlib.sha256(
                (content + metadata.get("url", "")).encode()
            ).hexdigest()
        
        return metadata

    def hash_document(self, doc: Document) -> str:
        """Generate hash for document deduplication."""
        content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
        url_hash = hashlib.sha256(doc.metadata.get("url", "").encode()).hexdigest()
        return hashlib.sha256((content_hash + url_hash).encode()).hexdigest()


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

    # Create processor with conservative settings for stability
    processor = ProcessIsolatedONNXProcessor(
        s3_bucket=S3_BUCKET,
        milvus_config=MILVUS_CONFIG,
        max_concurrent_documents=12,  # Reduced from 15
        onnx_workers=4,              # Reduced from 8
        batch_size=20                # Reduced from 30
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