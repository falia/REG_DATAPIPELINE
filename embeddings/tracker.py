# Tracker.py
import threading
import time
from typing import Dict, Optional


class Progress:
    """
    Thread-safe progress tracker.

    Fields tracked:
      - total: total files planned
      - downloaded: files downloaded and queued for processing
      - processed: files finished processing (even if 0 chunks)
      - stored_docs: number of chunks/docs successfully written to Milvus

    Methods used by your pipeline:
      - inc_downloaded(n=1)
      - inc_processed(n=1)
      - inc_stored(n=1)
      - snapshot() -> Dict[str, float|int]
    """
    def __init__(self, total_files: int):
        self.total: int = int(total_files) if total_files is not None else 0
        self.downloaded: int = 0
        self.processed: int = 0
        self.stored_docs: int = 0

        self._lock = threading.Lock()
        # Use a monotonic clock for elapsed time
        self._t0: float = time.perf_counter()

    # --- incrementers (thread-safe) ---

    def inc_downloaded(self, n: int = 1) -> None:
        with self._lock:
            self.downloaded += int(n)

    def inc_processed(self, n: int = 1) -> None:
        with self._lock:
            self.processed += int(n)

    def inc_stored(self, n: int = 1) -> None:
        with self._lock:
            self.stored_docs += int(n)

    # --- optional: change total if needed at runtime ---
    def set_total(self, total_files: int) -> None:
        with self._lock:
            self.total = int(total_files)

    # --- read-only snapshot for printing/metrics ---
    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            dl = self.downloaded
            pr = self.processed
            sd = self.stored_docs
            tot = self.total

        elapsed = max(time.perf_counter() - self._t0, 1e-6)

        # Files/sec based on processed count
        rate_fps = pr / elapsed

        pct_dl = (dl / tot * 100.0) if tot else 0.0
        pct_pr = (pr / tot * 100.0) if tot else 0.0

        # Simple ETA using average rate; 0 if not computable
        eta_sec = ((tot - pr) / rate_fps) if (rate_fps > 0 and tot and pr <= tot) else 0.0

        return {
            "downloaded": dl,
            "processed": pr,
            "stored_docs": sd,
            "total": tot,
            "pct_dl": pct_dl,
            "pct_pr": pct_pr,
            "elapsed": elapsed,
            "rate_fps": rate_fps,
            "eta_sec": eta_sec,
        }
