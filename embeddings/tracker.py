class Progress:
    def __init__(self, total_files: int):
        self.total = total_files
        self.downloaded = 0     # files handed off to processors
        self.processed = 0      # files parsed/chunked (regardless of chunk count)
        self.stored_docs = 0    # number of docs actually stored in Milvus
        self.t0 = time()
        self._lock = threading.Lock()

    def inc_downloaded(self, n=1):
        with self._lock:
            self.downloaded += n

    def inc_processed(self, n=1):
        with self._lock:
            self.processed += n

    def inc_stored(self, n=1):
        with self._lock:
            self.stored_docs += n

    def snapshot(self):
        with self._lock:
            dl = self.downloaded
            pr = self.processed
            sd = self.stored_docs
            tot = self.total
        elapsed = max(time() - self.t0, 1e-6)
        rate = pr / elapsed  # files/sec
        pct_dl = (dl / tot * 100) if tot else 0.0
        pct_pr = (pr / tot * 100) if tot else 0.0
        eta_sec = (tot - pr) / rate if rate > 0 and tot else 0
        return {
            "downloaded": dl, "processed": pr, "stored_docs": sd,
            "total": tot, "pct_dl": pct_dl, "pct_pr": pct_pr,
            "elapsed": elapsed, "rate_fps": rate, "eta_sec": eta_sec,
        }
