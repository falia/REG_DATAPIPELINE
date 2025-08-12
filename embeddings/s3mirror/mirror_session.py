#!/usr/bin/env python3
import os, sys, json, hashlib, re, errno
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1F]')

def safe_segment(text: str, maxlen: int = 120) -> str:
    s = "" if text is None else str(text)
    s = INVALID.sub("_", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    if not s:
        s = "_"
    if len(s) > maxlen:
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
        s = f"{s[:maxlen-13]}_{h}"
    reserved = {"CON","PRN","AUX","NUL"} | {f"COM{i}" for i in range(1,10)} | {f"LPT{i}" for i in range(1,10)}
    if s.upper() in reserved:
        s = f"_{s}_"
    return s

def safe_hash_token(data: str, length: int = 40) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:length]

def safe_local_rel_from_key(s3_key: str) -> str:
    parts = [p for p in s3_key.split("/") if p not in ("", ".", "..")]
    if not parts:
        return safe_segment("file")
    head = parts[:2]
    tail = "/".join(parts[2:]) if len(parts) > 2 else parts[-1]
    safe_head = [safe_segment(p) for p in head]
    hint = safe_segment(head[1] if len(head) > 1 else parts[0], 40)
    leaf = f"{hint}__{safe_hash_token(tail, 40)}"
    return os.path.join(*safe_head, leaf) if safe_head else leaf

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Mirror a session from S3 to local with safe filenames")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--session", required=True, help="e.g. 20250702_020822")
    ap.add_argument("--out-session", required=True, help="Dir to store metadata.json under original paths (e.g. /data/cssf/20250702_020822)")
    ap.add_argument("--out-cache", required=True, help="Cache root used by your pipeline (will write to <out-cache>/objects/...)")
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "eu-west-1"))
    ap.add_argument("--dl-workers", type=int, default=int(os.getenv("CSSF_DL_WORKERS","160")))
    ap.add_argument("--transfer-threads", type=int, default=int(os.getenv("CSSF_TRANSFER_THREADS","16")))
    ap.add_argument("--chunk-mb", type=int, default=int(os.getenv("CSSF_CHUNK_MB","128")))
    ap.add_argument("--max-pool", type=int, default=int(os.getenv("CSSF_S3_MAX_POOL","2048")))
    args = ap.parse_args()

    sess = boto3.session.Session(region_name=args.region)
    s3 = sess.client("s3", config=Config(
        max_pool_connections=args.max_pool,
        retries={"max_attempts": 10, "mode": "adaptive"},
        connect_timeout=10, read_timeout=60, tcp_keepalive=True,
    ))
    tcfg = TransferConfig(
        multipart_threshold=args.chunk_mb * 1024 * 1024,
        multipart_chunksize=args.chunk_mb * 1024 * 1024,
        max_concurrency=args.transfer_threads,
        use_threads=True,
    )

    prefix = f"{args.session}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=args.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    if not keys:
        print("No objects found under session", args.session)
        return 0

    print(f"Found {len(keys)} objects. Downloading…")
    out_cache_objects = os.path.join(args.out_cache, "objects")
    ensure_dir(out_cache_objects)
    ensure_dir(args.out_session)

    def job(key: str):
        try:
            # metadata.json keep exact path under out-session
            if key.endswith("metadata.json"):
                dst = os.path.join(args.out_session, key)  # may include subdirs
                ensure_dir(os.path.dirname(dst))
            else:
                # all heavy content goes to hashed cache path
                rel = safe_local_rel_from_key(key)
                dst = os.path.join(out_cache_objects, rel)
                ensure_dir(os.path.dirname(dst))

            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                return (key, "skip")

            s3.download_file(args.bucket, key, dst, Config=tcfg)
            return (key, "ok")
        except Exception as e:
            return (key, f"err: {e}")

    ok = err = skip = 0
    with ThreadPoolExecutor(max_workers=args.dl_workers) as ex:
        futures = [ex.submit(job, k) for k in keys]
        for fut in as_completed(futures):
            k, status = fut.result()
            if status == "ok":
                ok += 1
                if ok % 100 == 0:
                    print(f"Downloaded {ok}/{len(keys)} …")
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"[ERR] {k}: {status}")

    print(f"Done. ok={ok} skip={skip} err={err}")
    print(f"- metadata.json under: {args.out_session}/{args.session}/…")
    print(f"- content under:       {args.out_cache}/objects/…  (hashed paths)")
    return 0

if __name__ == "__main__":
    sys.exit(main())