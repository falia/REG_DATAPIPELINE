import json
import boto3
import hashlib
import logging
import time
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp

from langchain_core.documents import Document

# ========== QUIET NOISE ==========
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('unstructured').setLevel(logging.ERROR)
logging.getLogger('unstructured_inference').setLevel(logging.ERROR)
logging.getLogger('embeddings').setLevel(logging.ERROR)
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

# ========== YOUR MODULES ==========
from embeddings.embedding_provider.embedding_provider import EmbeddingService
from embeddings.chunker.document_chunker import DocumentChunker
from embeddings.parsers.parser import EurlexHTMLParser, CSSFHTMLParser, DocumentProcessor


# ==========================================================
# Worker process init: cap intra-op threads, keep things lean
# ==========================================================
def _worker_init(omp_threads: int = 1):
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("ORT_DISABLE_MEMORY_ARENA", "0")


# ==============================
# Per-worker singletons & caches
# ==============================
_PDF_CHUNKER = None
_HTML_PROCESSOR = None
_S3_CLIENT = None


def _ensure_worker_tools(chunker_cfg: Optional[Dict[str, Any]] = None):
    """Instantiate per-process tools exactly once."""
    global _PDF_CHUNKER, _HTML_PROCESSOR
    if _PDF_CHUNKER is None:
        max_chunk = int((chunker_cfg or {}).get("max_chunk_size", 1800))
        overlap = int((chunker_cfg or {}).get("overlap", 200))
        _PDF_CHUNKER = DocumentChunker(max_chunk_size=max_chunk, overlap=overlap)
    if _HTML_PROCESSOR is None:
        _HTML_PROCESSOR = DocumentProcessor(parsers=[EurlexHTMLParser(), CSSFHTMLParser()])


def _s3_read_bytes(bucket: str, s3_uri: str) -> bytes:
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client("s3")
    key = s3_uri.replace(f"s3://{bucket}/", "")
    resp = _S3_CLIENT.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def _is_pdf(url: str, content_type: str) -> bool:
    return (url.lower().endswith(".pdf") or (content_type and "application/pdf" in content_type.lower()))


def _extract_page_number_from_doc(doc: Document) -> int:
    try:
        return int(doc.metadata.get("page_number", 0)) if doc.metadata else 0
    except Exception:
        return 0


# ======================================
# Worker: parse + chunk a single document
# ======================================
def parse_and_chunk_worker(
    bucket: str,
    file_info: Dict[str, Any],
    base_metadata: Dict[str, Any],
    model_name: str = "detectron2_onnx",  # try "yolox_quantized" for more speed on CPU
    omp_threads: int = 1,
    chunker_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Runs inside a separate process. Each process owns its ONNX Runtime session(s)
    and threadpools. Returns serializable payload: {ok, url, n_chunks, docs:[{text, page_number}]}
    """
    try:
        _ensure_worker_tools(chunker_cfg)

        s3_uri = file_info.get("s3_uri")
        original_url = file_info.get("url", "unknown")
        content_type = file_info.get("content_type", "application/octet-stream")
        if not s3_uri:
            return {"ok": False, "url": original_url, "error": "missing s3_uri"}

        content = _s3_read_bytes(bucket, s3_uri)
        if not content:
            return {"ok": False, "url": original_url, "error": "empty content"}

        elements = None
        if _is_pdf(original_url, content_type):
            # Prefer Unstructured hi_res with ONNX backend
            from unstructured.partition.pdf import partition_pdf

            redacted_path = None
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                pdf_path = tmp.name
            try:
                # Optional header/footer redaction if available
                try:
                    from embeddings.parsers.PDFRemoveHeaderFooter import HeaderFooterRedactor
                    redactor = HeaderFooterRedactor(
                        top_k=5, bottom_k=3, win=8,
                        header_th=0.65, footer_th=0.65, rank_th=0.55,
                        pad=2.0, black=True,
                    )
                    redacted_path = redactor.redact_to_temp(pdf_path)
                except Exception:
                    redacted_path = pdf_path

                elements = partition_pdf(
                    filename=redacted_path,
                    strategy="hi_res",
                    hi_res_model_name=model_name,
                    extract_images_in_pdf=False,
                    languages=["en", "fr"],     # set expected languages to skip detection
                )
            finally:
                # Cleanup the original tmp file; keep redacted_path if same as pdf_path
                if pdf_path and os.path.exists(pdf_path) and pdf_path != redacted_path:
                    try:
                        os.unlink(pdf_path)
                    except Exception:
                        pass
        else:
            # HTML and other content types
            elements = _HTML_PROCESSOR.process(content, original_url, content_type)

        # Chunk into lightweight dicts
        chunked_docs: List[Document] = _PDF_CHUNKER.chunk_document(elements, original_url)
        packed_docs = [{"text": d.page_content, "page_number": _extract_page_number_from_doc(d)} for d in chunked_docs]

        return {"ok": True, "url": original_url, "docs": packed_docs, "n_chunks": len(packed_docs)}
    except Exception as e:
        return {"ok": False, "url": file_info.get("url", "unknown"), "error": str(e)}


# ==================
# Tracking utilities
# ==================
class ParallelismTracker:
    """Track parallelism and throughput from the main process."""
    def __init__(self):
        self.active = 0
        self.max_concurrent = 0
        self.total_processed = 0
        self.start_times: Dict[str, float] = {}

    def start(self, url: str):
        self.active += 1
        self.max_concurrent = max(self.max_concurrent, self.active)
        self.start_times[url] = time.time()
        print(f"🟢 START  [{self.active:2d}] {url[-60:]:60s}")

    def finish(self, url: str, n_chunks: int):
        self.active = max(0, self.active - 1)
        self.total_processed += 1
        dur = time.time() - self.start_times.get(url, time.time())
        print(f"✅ FINISH [{self.active:2d}] {url[-60:]:60s} ({dur:4.1f}s, {n_chunks} chunks) [Total: {self.total_processed}]")

    def get_stats(self):
        return {"max_concurrent": self.max_concurrent, "total_processed": self.total_processed}


# =====================
# Main processing class
# =====================
class ProcessIsolatedONNXPipeline:
    def __init__(
        self,
        s3_bucket: str,
        session_id: Optional[str] = None,
        milvus_config: Optional[Dict[str, Any]] = None,
        max_workers: int = 8,
        omp_threads: int = 1,
        hi_res_model_name: str = "detectron2_onnx",
    ):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.max_workers = max_workers
        self.omp_threads = omp_threads
        self.hi_res_model_name = hi_res_model_name

        self.s3 = boto3.client("s3")
        self.tracker = ParallelismTracker()

        self._seen_hashes: set[str] = set()
        self._seen_hashes_lock = mp.RLock()

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

        # Persistent warm pool (continuous scheduling)
        self._ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=self._ctx,
            initializer=_worker_init,
            initargs=(self.omp_threads,),
            # max_tasks_per_child=50,  # optional (Python 3.11+)
        )

        print(f"🚀 Continuous pipeline: {max_workers} worker processes, model={hi_res_model_name}, OMP={omp_threads}")

    # ------------
    # S3 utilities
    # ------------
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
        except Exception:
            return {}

    # ---------------
    # Metadata helper
    # ---------------
    def flatten_metadata_for_search(self, metadata: dict, page_number: Optional[int] = None) -> dict:
        def _clamp(x: Any, n: int) -> Any:
            if isinstance(x, str):
                return x[:n]
            return x

        def _as_json(x: Any, default: Any) -> str:
            try:
                return json.dumps(x) if x is not None else json.dumps(default)
            except Exception:
                return json.dumps(default)

        md = {
            "url": _clamp(metadata.get("url", ""), 2500),
            "title": _clamp(metadata.get("title", ""), 2500),
            "subtitle": _clamp(metadata.get("subtitle", ""), 2500),
            "document_type": _clamp(metadata.get("document_type", ""), 100),
            "document_number": _clamp(metadata.get("document_number", ""), 100),
            "publication_date": _clamp(metadata.get("publication_date") or "", 50),
            "update_date": _clamp(metadata.get("update_date") or "", 50),
            "content_hash": _clamp(metadata.get("content_hash", ""), 100),
            "crawl_timestamp": _clamp(metadata.get("crawl_timestamp", ""), 50),
            "file_size": int(metadata.get("file_size") or 0),
            "lang": _clamp(metadata.get("lang", ""), 10),
            "super_category": _clamp(metadata.get("super_category", ""), 100),
            "crawl_session": _clamp(metadata.get("crawl_session", ""), 50),
            "top_related": _as_json(metadata.get("top_related", []), []),
            "bottom_related": _as_json(metadata.get("bottom_related", []), []),
            "themes": _as_json(metadata.get("themes", []), []),
            "entities": _as_json(metadata.get("entities", []), []),
            "keywords": _as_json(metadata.get("keywords", []), []),
        }
        if page_number is not None:
            try:
                md["page_number"] = int(page_number)
            except Exception:
                md["page_number"] = 0
        return md

    # -----------------------
    # Store in Milvus (schema)
    # -----------------------
    def store_documents_in_milvus(self, documents: List[Document]) -> Dict[str, Any]:
        """Store with FULL flattened metadata & normalization (fixes 'subtitle' required errors)."""
        if not documents:
            return {"count": 0, "milvus_ids": []}

        REQUIRED_FIELDS = [
            "url", "title", "subtitle", "document_type", "document_number",
            "publication_date", "update_date", "content_hash", "crawl_timestamp",
            "file_size", "lang", "super_category", "crawl_session",
            "top_related", "bottom_related", "themes", "entities", "keywords",
            "page_number",
        ]
        INT_FIELDS = {"file_size", "page_number"}
        JSON_STRING_FIELDS = {"top_related", "bottom_related", "themes", "entities", "keywords"}

        def _normalize_meta(md: Dict[str, Any]) -> Dict[str, Any]:
            m = dict(md or {})
            for f in REQUIRED_FIELDS:
                if f not in m or m[f] is None:
                    m[f] = 0 if f in INT_FIELDS else ""
            for f in INT_FIELDS:
                try:
                    m[f] = int(m.get(f, 0) or 0)
                except Exception:
                    m[f] = 0
            for f in JSON_STRING_FIELDS:
                v = m.get(f, [])
                if not isinstance(v, str):
                    try:
                        m[f] = json.dumps(v if v is not None else [])
                    except Exception:
                        m[f] = json.dumps([])
            return m

        new_docs, texts, metas = [], [], []
        for d in documents:
            base_meta = _normalize_meta(d.metadata)
            base_meta["doc_id"] = hashlib.sha256((d.page_content + base_meta.get("url", "")).encode()).hexdigest()

            with self._seen_hashes_lock:
                if base_meta["doc_id"] in self._seen_hashes:
                    continue
                self._seen_hashes.add(base_meta["doc_id"])

            new_docs.append(d)
            texts.append(d.page_content)
            metas.append(base_meta)

        if not new_docs:
            return {"count": 0, "milvus_ids": []}

        try:
            result = self.embedding_service.add_texts_to_store(texts=texts, metadatas=metas)
            print(f"💾 Stored {result['count']} documents in Milvus")
            return result
        except Exception as e:
            print(f"❌ Failed to store in Milvus: {e}")
            return {"count": 0, "milvus_ids": []}

    # ---------------
    # File enumeration
    # ---------------
    def _iter_session_files(self, session_id: str):
        """Yield (file_info, base_metadata) pairs for the whole session."""
        metadata_files = self.get_session_metadata_files(session_id)
        print(f"📂 Found {len(metadata_files)} metadata files")
        for i, s3_key in enumerate(metadata_files):
            if i % 100 == 0 and i > 0:
                print(f"   📄 Scanned {i}/{len(metadata_files)} metadata files...")
            try:
                md = self.read_metadata_from_s3(s3_key)
                if not md:
                    continue
                md["crawl_session"] = session_id
                for file_info in md.get("top_related", []):
                    yield (file_info, md)
            except Exception as e:
                print(f"❌ Error with metadata file {s3_key}: {e}")

    # --------------------------
    # Continuous streaming runner
    # --------------------------
    def stream_process_session(self, session_id: str) -> Dict[str, Any]:
        print(f"\n🌊 CONTINUOUS MODE (no batches)")
        print(f"Session: {session_id}")
        print(f"Workers: {self.max_workers} | Model: {self.hi_res_model_name} | OMP: {self.omp_threads}")
        print(f"{'='*60}")

        start_time = time.time()
        total_chunks_stored = 0

        task_iter = self._iter_session_files(session_id)
        inflight: dict = {}  # fut -> (file_info, base_md)

        # Prime the pool
        for _ in range(self.max_workers):
            try:
                file_info, base_md = next(task_iter)
            except StopIteration:
                break
            url = file_info.get("url", "unknown")
            self.tracker.start(url)
            fut = self._executor.submit(
                parse_and_chunk_worker,
                self.s3_bucket, file_info, base_md,
                self.hi_res_model_name, self.omp_threads,
                {"max_chunk_size": 1800, "overlap": 200},
            )
            inflight[fut] = (file_info, base_md)

        # Drain/Refill loop: when one finishes, submit the next file immediately
        while inflight:
            done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                file_info, base_md = inflight.pop(fut)
                url = file_info.get("url", "unknown")

                try:
                    res = fut.result()
                    if not res.get("ok"):
                        self.tracker.finish(url, 0)
                        print(f"❌ Failed: {url} -> {res.get('error')}")
                    else:
                        docs: List[Document] = []
                        for d in res.get("docs", []):
                            pn = d.get("page_number", 0)
                            md_flat = self.flatten_metadata_for_search(base_md, page_number=pn)
                            docs.append(Document(page_content=d["text"], metadata=md_flat))
                        self.tracker.finish(url, len(docs))
                        if docs:
                            store_result = self.store_documents_in_milvus(docs)
                            total_chunks_stored += store_result.get("count", 0)
                except Exception as e:
                    self.tracker.finish(url, 0)
                    print(f"❌ Error processing {url}: {e}")

                # Refill one slot
                try:
                    file_info, base_md = next(task_iter)
                    url2 = file_info.get("url", "unknown")
                    self.tracker.start(url2)
                    fut2 = self._executor.submit(
                        parse_and_chunk_worker,
                        self.s3_bucket, file_info, base_md,
                        self.hi_res_model_name, self.omp_threads,
                        {"max_chunk_size": 1800, "overlap": 200},
                    )
                    inflight[fut2] = (file_info, base_md)
                except StopIteration:
                    pass

        # Wrap up
        self._executor.shutdown(wait=True)

        total_time = time.time() - start_time
        final_stats = self.tracker.get_stats()
        dpm = (final_stats["total_processed"] / total_time) * 60 if total_time > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"   CONTINUOUS PIPELINE COMPLETE!")
        print(f"   Worker processes: {self.max_workers}")
        print(f"   Total documents: {final_stats['total_processed']}")
        print(f"   Max concurrent: {final_stats['max_concurrent']}")
        print(f"   Final rate: {dpm:.1f} docs/min")
        print(f"{'='*60}")

        return {
            "total_processed": final_stats["total_processed"],
            "total_stored": total_chunks_stored,
            "max_concurrent": final_stats["max_concurrent"],
            "docs_per_minute": dpm,
        }


# =====
# main
# =====
def main():
    S3_BUCKET = "cssf-crawl"
    MILVUS_CONFIG = {
        "host": "54.217.166.223",
        "port": "19530",
        "collection_name": "CG_DEMO",
        "connection_args": {"host": "54.217.166.223", "port": "19530"},
    }

    pipeline = ProcessIsolatedONNXPipeline(
        s3_bucket=S3_BUCKET,
        milvus_config=MILVUS_CONFIG,
        max_workers=8,             # number of ONNX-isolated processes
        omp_threads=1,             # per-process intra-op threads
        hi_res_model_name="detectron2_onnx",  # or "yolox_quantized" for more speed
    )

    result = pipeline.stream_process_session("20250702_020822")

    print(f"\n🎉 FINAL RESULTS:")
    print(f"Rate: {result['docs_per_minute']:.1f} docs/minute")
    print(f"Max concurrent: {result['max_concurrent']}")


if __name__ == "__main__":
    main()
