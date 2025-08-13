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
import shutil
import errno
import mimetypes
from typing import List, Dict, Any, Optional, Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from collections import defaultdict
from queue import Queue

from boto3.s3.transfer import TransferConfig

from langchain_core.documents import Document
from embeddings.embedding_provider.embedding_provider import EmbeddingService
from embeddings.chunker.document_chunker import DocumentChunker
from embeddings.parsers.parser import EurlexHTMLParser, CSSFHTMLParser, PDFParserPipeline, DocumentProcessor

# ✅ external tracker
from embeddings.tracker import Progress

# Allowed fields
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
    if text is None:
        text = ""
    s = str(text)
    s = _INVALID_CHARS.sub("_", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .")
    if not s:
        s = "_"
    if len(s) > maxlen:
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
        s = f"{s[:maxlen-13]}_{h}"
    reserved = {"CON","PRN","AUX","NUL"} | {f"COM{i}" for i in range(1,10)} | {f"LPT{i}" for i in range(1,10)}
    if s.upper() in reserved:
        s = f"_{s}_"
    return s

def _safe_hash_token(data: str, length: int = 40) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:length]

def _win_long_path(path: str) -> str:
    if os.name == "nt":
        if not path.startswith("\\\\?\\"):
            abs_path = os.path.abspath(path)
            if len(abs_path) >= 240:
                return "\\\\?\\" + abs_path
            return abs_path
    return path

def _safe_local_rel_from_key(s3_key: str) -> str:
    parts = [p for p in s3_key.split("/") if p not in ("", ".", "..")]
    if not parts:
        return _safe_segment("file")
    head = parts[:2]
    tail = "/".join(parts[2:]) if len(parts) > 2 else parts[-1]
    safe_head = [_safe_segment(p) for p in head]
    hint = _safe_segment(head[1] if len(head) > 1 else parts[0], 40)
    leaf = f"{hint}__{_safe_hash_token(tail, 40)}"
    rel_path = os.path.join(*safe_head, leaf) if safe_head else leaf
    return rel_path

def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if s3_uri.startswith("s3://"):
        u = urlparse(s3_uri)
        return u.netloc, u.path.lstrip("/")
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
        transfer_threads_per_download: int = 8,  # threads per object for multipart
    ):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.local_cache_dir = os.path.abspath(local_cache_dir)
        self.max_workers = max_workers                       # processing workers
        self.download_max_workers = download_max_workers     # download workers
        self.batch_store_size = batch_store_size
        self.transfer_threads = transfer_threads_per_download

        session = boto3.session.Session(region_name=region_name)
        self.s3 = session.client("s3")

        self.transfer_config = TransferConfig(
            multipart_threshold=multipart_chunksize_mb * 1024 * 1024,
            multipart_chunksize=multipart_chunksize_mb * 1024 * 1024,
            max_concurrency=self.transfer_threads,
            use_threads=True,
        )

        # ---- Parser wiring ----
        # Use a shared, lightweight HTML processor
        self.html_processor = DocumentProcessor(parsers=[EurlexHTMLParser(), CSSFHTMLParser()])
        # PDF parser is NOT shared; created per-thread (thread-local) to avoid cross-thread state.
        self._tls = threading.local()
        self.pdf_max_concurrency = int(os.getenv("CSSF_PDF_CONCURRENCY", "16"))
        self._pdf_slots = threading.Semaphore(self.pdf_max_concurrency)

        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)

        # dedupe structures (thread-safe)
        self.seen_hashes: set[str] = set()
        self._seen_lock = threading.Lock()

        logging.basicConfig(level=logging.CRITICAL, format="%(asctime)s %(levelname)s %(message)s")
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

        # Disk space policy (env-overridable)
        self.min_free_bytes = int(float(os.getenv("CSSF_MIN_FREE_MB", "8192")) * 1024 * 1024)     # 8 GB floor
        self.max_cache_bytes = int(float(os.getenv("CSSF_MAX_CACHE_GB", "200")) * 1024 * 1024 * 1024)  # 200 GB

        # progress handle (set in stream_session)
        self._progress: Optional[Progress] = None

    # ---------------- util ----------------

    def list_sessions(self) -> List[str]:
        try:
            resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix="", Delimiter="/")
            sessions = [p["Prefix"].rstrip("/") for p in resp.get("CommonPrefixes", [])]
            sessions.sort(reverse=True)
            return sessions
        except Exception:
            return []

    def get_most_recent_session(self) -> Optional[str]:
        sessions = self.list_sessions()
        if not sessions:
            return None
        return sessions[0]

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
        except Exception:
            return []

    def read_metadata_from_s3(self, s3_key: str) -> Dict[str, Any]:
        try:
            resp = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except Exception:
            return {}

    # ---------------- disk space helpers ----------------

    def _cache_dir(self) -> str:
        return os.path.join(self.local_cache_dir, "objects")

    def _bytes_free(self, path: str) -> int:
        try:
            total, used, free = shutil.disk_usage(os.path.abspath(path))
            return free
        except Exception:
            return 0

    def _cache_size(self) -> int:
        root = self._cache_dir()
        total = 0
        for base, _, files in os.walk(root):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(base, f))
                except Exception:
                    pass
        return total

    def _evict_cache(self, target_free_bytes: int) -> None:
        root = self._cache_dir()
        entries = []
        for base, _, files in os.walk(root):
            for f in files:
                p = os.path.join(base, f)
                try:
                    st = os.stat(p)
                    entries.append((st.st_mtime, st.st_size, p))
                except Exception:
                    pass
        entries.sort()
        for _, _sz, p in entries:
            try:
                os.remove(p)
                try:
                    os.removedirs(os.path.dirname(p))
                except Exception:
                    pass
                if self._bytes_free(root) >= target_free_bytes:
                    break
            except Exception:
                pass

    def _ensure_space_for(self, expected_bytes: int) -> None:
        root = self._cache_dir()
        _ensure_dir(root)
        free_now = self._bytes_free(root)
        cache_sz = self._cache_size()
        if cache_sz > self.max_cache_bytes:
            self._evict_cache(self.min_free_bytes + expected_bytes)
            return
        need = max(self.min_free_bytes + expected_bytes - free_now, 0)
        if need > 0:
            self._evict_cache(self.min_free_bytes + expected_bytes)

    # ---------------- file I/O helpers ----------------

    def _read_local_bytes(self, local_path: str) -> bytes:
        try:
            with open(_win_long_path(local_path), "rb") as f:
                return f.read()
        except Exception:
            return b""

    def _filename_from_url_or_key(self, original_url: Optional[str], key: str, local_path: Optional[str] = None) -> str:
        if original_url:
            try:
                u = urlparse(original_url)
                name = os.path.basename(u.path.rstrip("/"))
                if name:
                    return name
                parts = [p for p in u.path.split("/") if p]
                if parts:
                    return parts[-1]
                if u.netloc:
                    return u.netloc
            except Exception:
                pass
        base = os.path.basename(key) or (os.path.basename(local_path) if local_path else "")
        return base or "document"

    # ---------------- parsing helpers (HTML vs PDF) ----------------

    def _is_pdf_type(self, content_type: str, original_url: str, local_path: str) -> bool:
        ct = (content_type or "").lower()
        if "pdf" in ct:
            return True
        p = (original_url or local_path or "").lower()
        return p.endswith(".pdf")

    def _get_thread_pdf_parser(self) -> PDFParserPipeline:
        """Create/reuse a PDF parser per-thread to avoid cross-thread sharing."""
        if not hasattr(self._tls, "pdf_parser"):
            self._tls.pdf_parser = PDFParserPipeline()
        return self._tls.pdf_parser

    def _parse_to_elements(self, content: bytes, original_url: str, content_type: str, is_pdf: bool):
        if is_pdf:
            parser = self._get_thread_pdf_parser()
            return parser.process(content, original_url, "application/pdf")
        # Non-PDF: shared HTML-capable processor
        return self.html_processor.process(content, original_url, content_type)

    # ---------------- streaming pipeline helpers ----------------

    def _process_one_file(self, metadata: Dict[str, Any], original_url: str, content_type: str, local_path: str) -> List[Document]:
        """Parse one local file and return chunked Documents with flattened metadata. Prints content type."""
        fname = self._filename_from_url_or_key(original_url, key=os.path.basename(local_path), local_path=local_path)
        ct = (content_type or "application/octet-stream").split(";")[0].strip()

        content = self._read_local_bytes(local_path)
        if not content:
            print(f"Processed: {fname} [{ct}]")
            return []

        is_pdf = self._is_pdf_type(ct, original_url, local_path)

        # PDF parsing is throttled to avoid choking other workers
        if is_pdf:
            with self._pdf_slots:
                elements = self._parse_to_elements(content, original_url, ct, is_pdf=True)
        else:
            elements = self._parse_to_elements(content, original_url, ct, is_pdf=False)

        chunked_docs = self.chunker.chunk_document(elements, original_url)

        out_docs: List[Document] = []
        for d in chunked_docs:
            pn = self._extract_page_number(d)
            d.metadata = self.flatten_metadata_for_search(metadata, page_number=pn)
            out_docs.append(d)

        print(f"Processed: {fname} [{ct}]")
        return out_docs

    # ---------------- metadata shaping ----------------

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

    # ---------------- hashing & storage ----------------

    def hash_document(self, doc: Document) -> str:
        base = (doc.page_content + str(doc.metadata.get("url", "")) + str(doc.metadata.get("page_number", 0))).encode("utf-8")
        return hashlib.sha256(base).hexdigest()


    def store_documents_in_milvus_streaming(self, docs_q: Queue) -> Dict[str, Any]:
        """
        Consume docs from a queue and flush to Milvus in smaller batches, with a time-based flush too.
        This prevents the first big batch from blocking the entire pipeline.
        """
        total_count = 0
        all_ids: List[Any] = []
        batch: List[Document] = []

        # tunables (env)
        batch_target = self.batch_store_size                           # e.g., 128–256
        flush_sec = float(os.getenv("CSSF_EMBED_FLUSH_SEC", "2.0"))     # flush every N seconds even if batch small
        last_flush = time.time()

        def _flush_now():
            nonlocal batch, total_count, all_ids, last_flush
            if not batch:
                last_flush = time.time()
                return
            try:
                texts = [d.page_content for d in batch]
                metas = [d.metadata for d in batch]
                # optional: quick progress hint
                # print(f"[EMBED] flushing {len(batch)} docs...")
                result = self.embedding_service.add_texts_to_store(texts=texts, metadatas=metas)
                cnt = result.get("count", len(batch))
                ids = result.get("milvus_ids", [])
                total_count += cnt
                all_ids.extend(ids)
                if self._progress:
                    self._progress.inc_stored(cnt)
            except Exception as e:
                # don't kill the pipeline on embed failure; log & drop this batch
                print(f"[EMBED][WARN] flush failed for {len(batch)} docs: {e}", file=sys.stderr)
            finally:
                batch = []
                last_flush = time.time()

        while True:
            item = docs_q.get()
            if item is None:
                break

            docs: List[Document] = item
            for d in docs:
                meta = self._ensure_required_fields(self._filter_to_schema(dict(d.metadata or {})), d.page_content)

                doc_hash = self.hash_document(Document(page_content=d.page_content, metadata=meta))
                with self._seen_lock:
                    if doc_hash in self.seen_hashes:
                        continue
                    self.seen_hashes.add(doc_hash)
                meta["doc_id"] = doc_hash

                batch.append(Document(page_content=d.page_content, metadata=meta))

                # size-based flush
                if len(batch) >= batch_target:
                    _flush_now()

                # time-based flush
                elif (time.time() - last_flush) >= flush_sec:
                    _flush_now()

        # final flush
        _flush_now()

        return {"count": total_count, "milvus_ids": all_ids}





    def _flush_batch(self, docs: List[Document]) -> Tuple[int, List[Any]]:
        try:
            texts = [d.page_content for d in docs]
            metas = [d.metadata for d in docs]
            result = self.embedding_service.add_texts_to_store(texts=texts, metadatas=metas)
            cnt = result.get("count", len(docs))
            return cnt, result.get("milvus_ids", [])
        except Exception:
            return (0, [])

    # ---------------- STREAMING SESSION ORCHESTRATION ----------------

    def stream_session(self, session_id: str) -> Dict[str, Any]:
        """Concurrent download → process → store pipeline with proper backpressure + progress tracker."""
        # 1) Read metadata files and build download plan
        keys = self.get_session_metadata_files(session_id)
        if not keys:
            return {"session_id": session_id, "processed": 0, "stored": 0, "errors": 0}

        metadatas: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(32, self.max_workers * 2)) as ex:
            futures = [ex.submit(self.read_metadata_from_s3, k) for k in keys]
            for fut in as_completed(futures):
                md = fut.result() or {}
                if md:
                    md["crawl_session"] = session_id
                    metadatas.append(md)

        download_jobs: List[Tuple[str, str, str, Dict[str, Any], Dict[str, Any]]] = []
        for md in metadatas:
            for fi in md.get("top_related", []):
                s3_uri = fi.get("s3_uri")
                if not s3_uri:
                    continue
                b, k = _parse_s3_uri(s3_uri)
                if not b:
                    b = self.s3_bucket
                local_rel = _safe_local_rel_from_key(k)
                local_path = os.path.join(self.local_cache_dir, "objects", local_rel)
                download_jobs.append((b, k, local_path, md, fi))

        if not download_jobs:
            return {"session_id": session_id, "processed": 0, "stored": 0, "errors": 0}

        # 2) Progress + heartbeat
        progress = Progress(total_files=len(download_jobs))
        self._progress = progress

        hb_stop = threading.Event()
        hb_interval = float(os.getenv("CSSF_PROGRESS_EVERY", "5"))

        def heartbeat():
            if os.getenv("CSSF_PROGRESS", "1") == "0":
                return
            while not hb_stop.wait(hb_interval):
                snap = progress.snapshot()
                print(
                    "[PROG] dl {dl}/{tot} ({pdl:.1f}%) | proc {pr}/{tot} ({ppr:.1f}%) | stored_docs {sd} | rate {r:.2f} f/s | ETA ~ {eta:.0f}s"
                    .format(dl=snap["downloaded"], pr=snap["processed"], sd=snap["stored_docs"], tot=snap["total"],
                            pdl=snap["pct_dl"], ppr=snap["pct_pr"], r=snap["rate_fps"], eta=snap["eta_sec"])
                )

        hb = threading.Thread(target=heartbeat, name="PROG", daemon=True)
        hb.start()

        # 3) Queues
        proc_q: Queue = Queue(maxsize=int(os.getenv("CSSF_PIPELINE_QUEUE", "100")))  # download -> processors
        docs_q: Queue = Queue(maxsize=int(os.getenv("CSSF_DOCS_QUEUE", "200")))      # processors -> writer

        # 4) Start writer FIRST
        writer_result = {}
        def writer_worker():
            writer_result["res"] = self.store_documents_in_milvus_streaming(docs_q)

        writer = threading.Thread(target=writer_worker, daemon=True)
        writer.start()

        # 5) Start processor workers
        def processor_worker():
            while True:
                item = proc_q.get()
                if item is None:
                    proc_q.task_done()
                    break
                md, fi, local_path = item
                original_url = (fi.get("url") or "").strip()
                ct = (fi.get("content_type") or "").strip()
                if not ct or ct == "application/octet-stream":
                    guess, _ = mimetypes.guess_type(original_url or local_path)
                    if not guess and (original_url.lower().endswith(".pdf") or local_path.lower().endswith(".pdf")):
                        guess = "application/pdf"
                    ct = guess or "application/octet-stream"
                try:
                    # Route to PDF/non-PDF paths; PDF path is semaphore-limited.
                    is_pdf = self._is_pdf_type(ct, original_url, local_path)
                    if is_pdf:
                        with self._pdf_slots:
                            docs = self._process_one_file(md, original_url, "application/pdf", local_path)
                    else:
                        docs = self._process_one_file(md, original_url, ct, local_path)

                    if docs:
                        docs_q.put(docs)
                except Exception:
                    pass
                finally:
                    progress.inc_processed(1)  # ✅ file processed (even if 0 chunks)
                    proc_q.task_done()

        processors = []
        for i in range(self.max_workers):
            t = threading.Thread(target=processor_worker, name=f"PROC-{i+1:02d}", daemon=True)
            t.start()
            processors.append(t)

        # 6) Downloader workers (produce into proc_q)
        def downloader_task(job_tuple):
            b, k, lp, md, fi = job_tuple
            original_url = fi.get("url")
            try:
                self._download_one(b, k, lp, original_url)
            except Exception:
                pass
            if os.path.exists(lp) and os.path.getsize(lp) > 0:
                proc_q.put((md, fi, lp))
                progress.inc_downloaded(1)  # ✅ ready for processing

        with ThreadPoolExecutor(max_workers=self.download_max_workers, thread_name_prefix="DL") as dl_ex:
            futures = [dl_ex.submit(downloader_task, job) for job in download_jobs]
            for _ in as_completed(futures):
                pass

        # 7) Wait until all proc_q tasks are handled, then stop processors
        proc_q.join()
        for _ in range(self.max_workers):
            proc_q.put(None)
        for t in processors:
            t.join()

        # 8) Signal writer to finish AFTER processors are done enqueuing docs
        docs_q.put(None)
        writer.join()

        # 9) stop heartbeat
        hb_stop.set()
        hb.join(timeout=1)

        res = writer_result.get("res", {"count": 0, "milvus_ids": []})
        return {
            "session_id": session_id,
            "processed": res.get("count", 0),
            "stored": res.get("count", 0),
            "errors": 0,
        }

    # Backwards-compat wrappers
    def process_latest_session(self) -> Dict[str, Any]:
        latest = self.get_most_recent_session()
        if not latest:
            return {"processed": 0, "stored": 0, "errors": 0}
        return self.stream_session(latest)

    def process_session(self, session_id: str) -> Dict[str, Any]:
        return self.stream_session(session_id)

    # ---------------- download helper ----------------

    def _download_one(self, bucket: str, key: str, dst_path: str, original_url: Optional[str]) -> Tuple[str, bool]:
        """Download a single S3 object to dst_path. Returns (dst_path, did_download). Prints only when downloaded."""
        _ensure_dir(os.path.dirname(dst_path))
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            return (dst_path, False)

        expected = 64 * 1024 * 1024
        try:
            head = self.s3.head_object(Bucket=bucket, Key=key)
            expected = int(head.get("ContentLength") or expected)
        except Exception:
            pass

        self._ensure_space_for(expected)

        try:
            dst_for_api = _win_long_path(dst_path)
            self.s3.download_file(
                Bucket=bucket,
                Key=key,
                Filename=dst_for_api,
                Config=self.transfer_config,
            )
            return (dst_path, True)
        except OSError as oe:
            if getattr(oe, "errno", None) == errno.ENOSPC:
                self._evict_cache(self.min_free_bytes + expected)
                try:
                    dst_for_api = _win_long_path(dst_path)
                    self.s3.download_file(
                        Bucket=bucket,
                        Key=key,
                        Filename=dst_for_api,
                        Config=self.transfer_config,
                    )
                    print(f"Downloaded: {self._filename_from_url_or_key(original_url, key, dst_path)}")
                    return (dst_path, True)
                except Exception:
                    return (dst_path, False)
            else:
                return (dst_path, False)
        except Exception:
            return (dst_path, False)

def main():
    S3_BUCKET = os.getenv("CSSF_S3_BUCKET", "cssf-crawl")
    SESSION_ID = os.getenv("CSSF_SESSION_ID")
    REGION = os.getenv("AWS_REGION", "eu-west-1")

    # m7i.16xlarge tuned defaults (env-overridable)
    DL_WORKERS = int(os.getenv("CSSF_DL_WORKERS", "192"))            # files in parallel (downloader)
    MAX_WORKERS = int(os.getenv("CSSF_MAX_WORKERS", "48"))           # processing workers
    CHUNK_MB = int(os.getenv("CSSF_CHUNK_MB", "64"))                 # multipart chunk size
    TRANSFER_THREADS = int(os.getenv("CSSF_TRANSFER_THREADS", "8"))  # threads per file (multipart)
    BATCH_STORE = int(os.getenv("CSSF_BATCH_STORE", "1024"))         # embeddings batch
    LOCAL_CACHE = os.getenv("CSSF_LOCAL_CACHE", "./_cache/cssf_session")

    # queue sizes
    os.environ.setdefault("CSSF_PIPELINE_QUEUE", "100")
    os.environ.setdefault("CSSF_DOCS_QUEUE", "200")

    # progress heartbeat controls
    os.environ.setdefault("CSSF_PROGRESS_EVERY", "5")  # seconds
    os.environ.setdefault("CSSF_PROGRESS", "1")        # 1=on, 0=off

    # cache policy defaults for big machine
    os.environ.setdefault("CSSF_MIN_FREE_MB", "8192")
    os.environ.setdefault("CSSF_MAX_CACHE_GB", "200")

    # NEW: cap PDF concurrency (defaults to 4)
    os.environ.setdefault("CSSF_PDF_CONCURRENCY", "4")

    MILVUS_CONFIG = {
        "host": "54.217.166.223",
        "port": "19530",
        "collection_name": "cssf_documents_final_final_CGDEMO41",
        "connection_args": {"host": "54.217.166.223", "port": "19530"},
    }

    processor = S3MetadataProcessor(
        s3_bucket=S3_BUCKET,
        session_id=SESSION_ID,
        milvus_config=MILVUS_CONFIG,
        local_cache_dir=LOCAL_CACHE,
        max_workers=MAX_WORKERS,
        download_max_workers=DL_WORKERS,
        multipart_chunksize_mb=CHUNK_MB,
        region_name=REGION,
        batch_store_size=BATCH_STORE,
        transfer_threads_per_download=TRANSFER_THREADS,
    )

    target_session = SESSION_ID or processor.get_most_recent_session()
    if target_session:
        processor.stream_session(target_session)

if __name__ == "__main__":
    main()