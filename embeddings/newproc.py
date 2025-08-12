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
from typing import List, Dict, Any, Optional, Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from collections import defaultdict

from boto3.s3.transfer import TransferConfig

from langchain_core.documents import Document
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
        transfer_threads_per_download: int = 8,
    ):
        self.s3_bucket = s3_bucket
        self.session_id = session_id
        self.local_cache_dir = os.path.abspath(local_cache_dir)
        self.max_workers = max_workers
        self.download_max_workers = download_max_workers
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

        self.processor = DocumentProcessor(parsers=[EurlexHTMLParser(), CSSFHTMLParser(), PDFParserPipeline()])
        self.chunker = DocumentChunker(max_chunk_size=1800, overlap=200)

        self.seen_hashes: set[str] = set()
        self._seen_lock = threading.Lock()

        # Silence all logs by default (only our explicit prints will show)
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
        self.min_free_bytes = int(float(os.getenv("CSSF_MIN_FREE_MB", "8192")) * 1024 * 1024)
        self.max_cache_bytes = int(float(os.getenv("CSSF_MAX_CACHE_GB", "200")) * 1024 * 1024 * 1024)

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
        freed = 0
        for _, sz, p in entries:
            try:
                os.remove(p)
                freed += sz
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

    # ---------------- local cache (pre-download) ----------------

    def _download_one(self, bucket: str, key: str, dst_path: str) -> Tuple[str, bool]:
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
                    return (dst_path, True)
                except Exception:
                    return (dst_path, False)
            else:
                return (dst_path, False)
        except Exception:
            return (dst_path, False)

    def collect_all_s3_uris_for_session(self, session_id: str) -> List[str]:
        keys = self.get_session_metadata_files(session_id)
        if not keys:
            return []
        uris: set[str] = set()
        with ThreadPoolExecutor(max_workers=min(32, self.max_workers * 2)) as ex:
            futures = {ex.submit(self.read_metadata_from_s3, k): k for k in keys}
            for fut in as_completed(futures):
                md = fut.result() or {}
                for fi in md.get("top_related", []):
                    s3_uri = fi.get("s3_uri")
                    if s3_uri:
                        uris.add(s3_uri)
        return sorted(uris)

    def pre_download_session_files(self, session_id: str) -> Dict[str, str]:
        uris = self.collect_all_s3_uris_for_session(session_id)
        if not uris:
            return {}
        plan: List[Tuple[str, str, str, str]] = []
        for u in uris:
            b, k = _parse_s3_uri(u)
            if not b:
                b = self.s3_bucket
            local_rel = _safe_local_rel_from_key(k)
            local_path = os.path.join(self.local_cache_dir, "objects", local_rel)
            plan.append((u, b, k, local_path))

        with ThreadPoolExecutor(max_workers=self.download_max_workers) as ex:
            futures = [ex.submit(self._download_one, b, k, lp) for (_u, b, k, lp) in plan]
            for _ in as_completed(futures):
                pass

        return {u: lp for (u, _b, _k, lp) in plan}

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

    # ---------------- pipeline ----------------

    def hash_document(self, doc: Document) -> str:
        base = (doc.page_content + str(doc.metadata.get("url", "")) + str(doc.metadata.get("page_number", 0))).encode("utf-8")
        return hashlib.sha256(base).hexdigest()

    def _read_local_bytes(self, local_path: str) -> bytes:
        try:
            with open(_win_long_path(local_path), "rb") as f:
                return f.read()
        except Exception:
            return b""

    def _filename_from_url_or_path(self, original_url: Optional[str], local_path: str) -> str:
        # Try URL path
        if original_url:
            try:
                u = urlparse(original_url)
                name = os.path.basename(u.path.rstrip("/"))
                if name:
                    return name
                # fallback to last non-empty segment
                parts = [p for p in u.path.split("/") if p]
                if parts:
                    return parts[-1]
                if u.netloc:
                    return u.netloc
            except Exception:
                pass
        # Fallback to local filename
        base = os.path.basename(local_path)
        return base or "document"

    def process_document(self, metadata: Dict[str, Any], local_map: Dict[str, str]) -> List[Document]:
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
                    if not local_path:
                        bkt, key = _parse_s3_uri(s3_uri)
                        if not bkt:
                            implicit = f"s3://{self.s3_bucket}/{key}"
                            local_path = local_map.get(implicit)
                if not local_path or not os.path.exists(local_path):
                    continue

                # >>> Minimal output: print only the filename being processed
                fname = self._filename_from_url_or_path(original_url, local_path)
                print(f"Processing: {fname}")

                content = self._read_local_bytes(local_path)
                if not content:
                    continue

                elements = self.processor.process(content, original_url, content_type)
                chunked_docs = self.chunker.chunk_document(elements, original_url)

                for d in chunked_docs:
                    pn = self._extract_page_number(d)
                    d.metadata = self.flatten_metadata_for_search(metadata, page_number=pn)
                all_docs.extend(chunked_docs)

            except Exception:
                # fully silent on errors per request
                pass

        return all_docs

    def store_documents_in_milvus(self, documents: List[Document]) -> Dict[str, Any]:
        if not documents:
            return {"count": 0, "milvus_ids": []}

        total_count = 0
        all_ids: List[Any] = []

        batch: List[Document] = []
        for d in documents:
            meta = self._ensure_required_fields(self._filter_to_schema(dict(d.metadata or {})), d.page_content)

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

        return {"count": total_count, "milvus_ids": all_ids}

    def _flush_batch(self, docs: List[Document]) -> Tuple[int, List[Any]]:
        try:
            texts = [d.page_content for d in docs]
            metas = [d.metadata for d in docs]
            result = self.embedding_service.add_texts_to_store(texts=texts, metadatas=metas)
            return result.get("count", len(docs)), result.get("milvus_ids", [])
        except Exception:
            return (0, [])

    def process_session(self, session_id: str) -> Dict[str, Any]:
        local_map = self.pre_download_session_files(session_id)
        keys = self.get_session_metadata_files(session_id)
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
                    total_errors += 1
                else:
                    total_processed += n
                    all_docs.extend(docs)

        res = self.store_documents_in_milvus(all_docs)
        return {
            "session_id": session_id,
            "processed": total_processed,
            "stored": res.get("count", 0),
            "errors": total_errors,
        }

    def process_latest_session(self) -> Dict[str, Any]:
        latest = self.get_most_recent_session()
        if not latest:
            return {"processed": 0, "stored": 0, "errors": 0}
        return self.process_session(latest)


def main():
    S3_BUCKET = os.getenv("CSSF_S3_BUCKET", "cssf-crawl")
    SESSION_ID = os.getenv("CSSF_SESSION_ID")
    REGION = os.getenv("AWS_REGION", "eu-west-1")

    # m7i.16xlarge tuned defaults (env-overridable)
    DL_WORKERS = int(os.getenv("CSSF_DL_WORKERS", "192"))
    MAX_WORKERS = int(os.getenv("CSSF_MAX_WORKERS", "48"))
    CHUNK_MB = int(os.getenv("CSSF_CHUNK_MB", "64"))
    TRANSFER_THREADS = int(os.getenv("CSSF_TRANSFER_THREADS", "8"))
    BATCH_STORE = int(os.getenv("CSSF_BATCH_STORE", "1024"))
    LOCAL_CACHE = os.getenv("CSSF_LOCAL_CACHE", "./_cache/cssf_session")

    os.environ.setdefault("CSSF_MIN_FREE_MB", "8192")
    os.environ.setdefault("CSSF_MAX_CACHE_GB", "200")

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
        local_cache_dir=LOCAL_CACHE,
        max_workers=MAX_WORKERS,
        download_max_workers=DL_WORKERS,
        multipart_chunksize_mb=CHUNK_MB,
        region_name=REGION,
        batch_store_size=BATCH_STORE,
        transfer_threads_per_download=TRANSFER_THREADS,
    )

    # Minimal stdout: no session listing or summaries
    target_session = SESSION_ID or processor.get_most_recent_session()
    if target_session:
        processor.process_session(target_session)


if __name__ == "__main__":
    main()
