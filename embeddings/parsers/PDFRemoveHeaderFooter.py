#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import fitz  # PyMuPDF


# ----------------------
# Utility functions
# ----------------------

#SEE: https://medium.com/@hussainshahbazkhawaja/paper-implementation-header-and-footer-extraction-by-page-association-3a499b2552ae

_DIG = "@"

def norm_text(s: str) -> str:
    """Normalize text: lowercase, collapse spaces, replace digits."""
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\d+", _DIG, s)
    return s.lower()


def levenshtein_sim(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0,1]."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    # DP distance
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1,      # deletion
                          curr[j - 1] + 1,  # insertion
                          prev[j - 1] + cost)  # substitution
        prev, curr = curr, prev
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def overlap_ratio_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap ratio along one axis: intersection / max(lengths)."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-9, max(a1 - a0, b1 - b0))
    return inter / denom


def geom_sim(rect1: fitz.Rect, rect2: fitz.Rect) -> float:
    """
    Geometry similarity in [0,1], combining horizontal & vertical overlap.
    This is more forgiving than IoU for thin text lines.
    """
    h = overlap_ratio_1d(rect1.x0, rect1.x1, rect2.x0, rect2.x1)
    v = overlap_ratio_1d(rect1.y0, rect1.y1, rect2.y0, rect2.y1)
    return 0.5 * (h + v)


# ----------------------
# Data structures
# ----------------------

@dataclass
class LineCand:
    raw: str
    norm: str
    rect: fitz.Rect
    page_index: int   # 0-based
    rank: int         # rank within header/footer group (0 = strongest)


@dataclass
class PageCands:
    headers: List[LineCand]  # top-K (rank 0 = topmost)
    footers: List[LineCand]  # bottom-K (rank 0 = bottommost)


# ----------------------
# Core logic
# ----------------------

def extract_candidates(doc: fitz.Document, top_k: int = 5, bottom_k: int = 3) -> List[PageCands]:
    """Collect top-K header and bottom-K footer candidate lines per page."""
    pages: List[PageCands] = []
    for pi, page in enumerate(doc):  # pi is 0-based
        H = page.rect.height
        data = page.get_text("dict")
        # Gather all lines with rect & text
        lines: List[Tuple[fitz.Rect, str]] = []
        for blk in data.get("blocks", []):
            for line in blk.get("lines", []) or []:
                rect = fitz.Rect(*line["bbox"])
                raw = " ".join(s.get("text", "") for s in line.get("spans", []) if s.get("text")).strip()
                if raw:
                    lines.append((rect, raw))
        if not lines:
            pages.append(PageCands(headers=[], footers=[]))
            continue

        # Sort by y-center ascending (top -> bottom)
        lines.sort(key=lambda t: 0.5 * (t[0].y0 + t[0].y1))

        # Header candidates: top-K
        hdr_lines = lines[:top_k]
        headers = [
            LineCand(raw=l[1], norm=norm_text(l[1]), rect=l[0], page_index=pi, rank=ri)
            for ri, l in enumerate(hdr_lines)
        ]

        # Footer candidates: bottom-K (rank 0 = bottom-most)
        ftr_lines = list(reversed(lines[-bottom_k:])) if bottom_k > 0 else []
        footers = [
            LineCand(raw=l[1], norm=norm_text(l[1]), rect=l[0], page_index=pi, rank=ri)
            for ri, l in enumerate(ftr_lines)
        ]

        pages.append(PageCands(headers=headers, footers=footers))
    return pages


def pair_similarity(a: LineCand, b: LineCand) -> float:
    """Similarity between two line candidates = text × geometry."""
    ts = levenshtein_sim(a.norm, b.norm)
    gs = geom_sim(a.rect, b.rect)
    return ts * gs


def rank_weights_header(k: int) -> List[float]:
    """Higher weight for topmost header line (rank 0)."""
    base = [1.0, 0.8, 0.6, 0.4, 0.2]
    if k <= len(base):
        return base[:k]
    # decay further if more than 5
    tail = [max(0.1, base[-1] - 0.1 * i) for i in range(k - len(base))]
    return base + tail


def rank_weights_footer(k: int) -> List[float]:
    """Higher weight for bottom-most footer line (rank 0)."""
    base = [1.0, 0.8, 0.6]
    if k <= len(base):
        return base[:k]
    tail = [max(0.1, base[-1] - 0.1 * i) for i in range(k - len(base))]
    return base + tail


def compute_confidences(
    pages: List[PageCands],
    win: int = 8,
    header_rank_threshold: float = 0.55,
    footer_rank_threshold: float = 0.55,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """
    For each page p and rank r, compute averaged similarity with same-rank lines
    on neighbor pages within window `win`. Return dicts of per-rank confidences
    keyed by (page_index, rank).
    """
    hdr_conf: Dict[Tuple[int, int], float] = {}
    ftr_conf: Dict[Tuple[int, int], float] = {}

    n = len(pages)

    # Headers
    max_hdr_r = max((len(pc.headers) for pc in pages), default=0)
    for r in range(max_hdr_r):
        for p in range(n):
            if r >= len(pages[p].headers):
                continue
            ref = pages[p].headers[r]
            scores, cnt = 0.0, 0
            lo, hi = max(0, p - win), min(n - 1, p + win)
            for q in range(lo, hi + 1):
                if q == p or r >= len(pages[q].headers):
                    continue
                cand = pages[q].headers[r]
                scores += pair_similarity(ref, cand)
                cnt += 1
            if cnt:
                hdr_conf[(p, r)] = scores / cnt

    # Footers
    max_ftr_r = max((len(pc.footers) for pc in pages), default=0)
    for r in range(max_ftr_r):
        for p in range(n):
            if r >= len(pages[p].footers):
                continue
            ref = pages[p].footers[r]
            scores, cnt = 0.0, 0
            lo, hi = max(0, p - win), min(n - 1, p + win)
            for q in range(lo, hi + 1):
                if q == p or r >= len(pages[q].footers):
                    continue
                cand = pages[q].footers[r]
                scores += pair_similarity(ref, cand)
                cnt += 1
            if cnt:
                ftr_conf[(p, r)] = scores / cnt

    return hdr_conf, ftr_conf


def select_lines_to_redact(
    pages: List[PageCands],
    hdr_conf: Dict[Tuple[int, int], float],
    ftr_conf: Dict[Tuple[int, int], float],
    header_threshold: float = 0.65,
    footer_threshold: float = 0.65,
    header_rank_threshold: float = 0.55,
    footer_rank_threshold: float = 0.55,
) -> Dict[int, List[fitz.Rect]]:
    """
    Decide which lines to redact on each page.
    - Compute weighted page-level confidence for headers/footers.
    - Redact only ranks whose per-rank confidence exceeds rank_threshold.
    """
    to_redact: Dict[int, List[fitz.Rect]] = {}

    n = len(pages)
    for p in range(n):
        # Headers
        h_lines = pages[p].headers
        if h_lines:
            weights = rank_weights_header(len(h_lines))
            w_sum = sum(weights)
            # page-level confidence
            conf = 0.0
            for r, w in enumerate(weights):
                conf += w * hdr_conf.get((p, r), 0.0)
            conf /= (w_sum or 1.0)

            if conf >= header_threshold:
                for r, ln in enumerate(h_lines):
                    if hdr_conf.get((p, r), 0.0) >= header_rank_threshold:
                        to_redact.setdefault(p, []).append(ln.rect)

        # Footers
        f_lines = pages[p].footers
        if f_lines:
            weights = rank_weights_footer(len(f_lines))
            w_sum = sum(weights)
            conf = 0.0
            for r, w in enumerate(weights):
                conf += w * ftr_conf.get((p, r), 0.0)
            conf /= (w_sum or 1.0)

            if conf >= footer_threshold:
                for r, ln in enumerate(f_lines):
                    if ftr_conf.get((p, r), 0.0) >= footer_rank_threshold:
                        to_redact.setdefault(p, []).append(ln.rect)

    return to_redact


def apply_redactions(
    in_path: Path,
    out_path: Path,
    rects_by_page: Dict[int, List[fitz.Rect]],
    pad: float = 2.0,
    black: bool = True,
) -> Tuple[int, int]:
    """Apply redaction rectangles to a copy of the PDF."""
    doc = fitz.open(str(in_path))
    try:
        pages_touched = 0
        boxes_total = 0
        for p, rects in rects_by_page.items():
            if not rects:
                continue
            page = doc[p]
            added = 0
            for r in rects:
                rr = fitz.Rect(r)
                rr.x0 -= pad; rr.y0 -= pad; rr.x1 += pad; rr.y1 += pad
                fill = (0, 0, 0) if black else (1, 1, 1)
                page.add_redact_annot(rr, fill=fill)
                added += 1
            if added:
                page.apply_redactions()
                pages_touched += 1
                boxes_total += added

        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path), deflate=True, garbage=4)
        return pages_touched, boxes_total
    finally:
        doc.close()


def make_copy_path(path: Path) -> Path:
    """Same folder, add _copy before .pdf."""
    if path.suffix.lower() != ".pdf":
        return path.with_suffix(path.suffix + "_copy")
    return path.with_name(f"{path.stem}_copy.pdf")



import tempfile
import os
class HeaderFooterRedactor:
    def __init__(
        self,
        top_k: int = 5,
        bottom_k: int = 3,
        win: int = 8,
        header_th: float = 0.65,
        footer_th: float = 0.65,
        rank_th: float = 0.55,
        pad: float = 2.0,
        black: bool = True,
    ):
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.win = win
        self.header_th = header_th
        self.footer_th = footer_th
        self.rank_th = rank_th
        self.pad = pad
        self.black = black

    def redact_to_temp(self, in_pdf_path: str) -> str:
        """Create a TEMP sanitized copy and return its path."""
        # 1) extract candidates
        doc = fitz.open(in_pdf_path)
        try:
            pages = extract_candidates(doc, top_k=self.top_k, bottom_k=self.bottom_k)
        finally:
            doc.close()

        # 2) score across window
        hdr_conf, ftr_conf = compute_confidences(
            pages,
            win=self.win,
            header_rank_threshold=self.rank_th,
            footer_rank_threshold=self.rank_th,
        )

        # 3) decide what to redact
        rects_by_page = select_lines_to_redact(
            pages,
            hdr_conf,
            ftr_conf,
            header_threshold=self.header_th,
            footer_threshold=self.footer_th,
            header_rank_threshold=self.rank_th,
            footer_rank_threshold=self.rank_th,
        )

        # 4) save to a true temp file
        fd, out_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        pages_touched, boxes = apply_redactions(
            in_path=Path(in_pdf_path),
            out_path=Path(out_path),
            rects_by_page=rects_by_page,
            pad=self.pad,
            black=self.black,
        )
        # optional: log/print pages_touched, boxes
        return out_path

# ----------------------
# CLI
# ----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Header/Footer removal by Page Association (PyMuPDF only, black redaction)."
    )
    ap.add_argument("pdf", type=Path, help="Input PDF")
    ap.add_argument("--top-k", type=int, default=5, help="Top-K header candidate lines per page (default: 5)")
    ap.add_argument("--bottom-k", type=int, default=3, help="Bottom-K footer candidate lines per page (default: 3)")
    ap.add_argument("--win", type=int, default=8, help="Page window size on each side (default: 8)")
    ap.add_argument("--header-th", type=float, default=0.65, help="Page-level header confidence threshold (default: 0.65)")
    ap.add_argument("--footer-th", type=float, default=0.65, help="Page-level footer confidence threshold (default: 0.65)")
    ap.add_argument("--rank-th", type=float, default=0.55, help="Per-rank confidence threshold (default: 0.55)")
    ap.add_argument("--pad", type=float, default=2.0, help="Padding (points) around redaction boxes (default: 2.0)")
    args = ap.parse_args()

    in_path: Path = args.pdf
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    out_path = make_copy_path(in_path)

    # 1) Extract candidates
    doc = fitz.open(str(in_path))
    try:
        pages = extract_candidates(doc, top_k=args.top_k, bottom_k=args.bottom_k)
    finally:
        doc.close()

    # 2) Score across window
    hdr_conf, ftr_conf = compute_confidences(
        pages,
        win=args.win,
        header_rank_threshold=args.rank_th,
        footer_rank_threshold=args.rank_th,
    )

    # 3) Select which lines to redact
    rects_by_page = select_lines_to_redact(
        pages,
        hdr_conf,
        ftr_conf,
        header_threshold=args.header_th,
        footer_threshold=args.footer_th,
        header_rank_threshold=args.rank_th,
        footer_rank_threshold=args.rank_th,
    )

    # 4) Apply redactions & save
    pages_touched, boxes = apply_redactions(
        in_path=in_path,
        out_path=out_path,
        rects_by_page=rects_by_page,
        pad=args.pad,
        black=True,
    )

    print(f"Saved: {out_path}")
    print(f"Pages touched: {pages_touched}, redaction boxes: {boxes}")


if __name__ == "__main__":
    main()
