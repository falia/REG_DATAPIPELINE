import re
import tempfile
import os
import logging
from typing import List, Sequence, Optional
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, Text
from unstructured.chunking.title import chunk_by_title

logger = logging.getLogger(__name__)

class ONNXPDFParser:
    """
    Unstructured hi_res (ONNX) PDF parser with optional cleaning/chunking and fallback.
    Thread/process safe if you call it from isolated workers.
    """

    def __init__(
        self,
        hi_res_model_name: str = "detectron2_onnx",      # or "yolox_quantized" for speed
        languages: Sequence[str] = ("eng", "fra"),       # ISO 639-3 codes for OCR/lang hints
        infer_table_structure: bool = True,
        include_page_breaks: bool = True,
        extract_images_in_pdf: bool = False,
        do_clean: bool = True,
        do_chunk: bool = False,                          # keep False to avoid double-chunking
        max_characters: int = 1500,
        new_after_n_chars: int = 1400,
        combine_under_n: int = 250,
    ):
        self.hi_res_model_name = hi_res_model_name
        self.languages = list(languages)
        self.infer_table_structure = infer_table_structure
        self.include_page_breaks = include_page_breaks
        self.extract_images_in_pdf = extract_images_in_pdf
        self.do_clean = do_clean
        self.do_chunk = do_chunk
        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.combine_under_n = combine_under_n

    # ------------ helpers ------------
    _soft_hyphen = "\u00AD"

    def _clean_text(self, t: str) -> str:
        if not t:
            return t
        # remove soft hyphens
        t = t.replace(self._soft_hyphen, "")
        # join hyphenated line breaks only when it looks like a word wrap: letter-<LF>letter
        t = re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])-\n(?=[A-Za-zÀ-ÖØ-öø-ÿ])", "", t)
        # collapse single newlines to spaces, but keep blank-line paragraph breaks
        # and avoid merging when the next line starts with a bullet/number
        t = re.sub(r"(?<!\n)\n(?!\n|[\-\*\u2022]|\d+\.)", " ", t)
        # collapse runs of spaces
        t = re.sub(r"[ \t]{2,}", " ", t)
        return t.strip()

    def _maybe_clean_elements(self, elements: List[Element]) -> List[Element]:
        if not self.do_clean:
            return elements
        cleaned: List[Element] = []
        for el in elements:
            if isinstance(el, Text) and getattr(el, "text", None):
                txt = self._clean_text(el.text)
                if txt:
                    el.text = txt  # mutate is fine; preserves metadata/page_number
                    cleaned.append(el)
            else:
                cleaned.append(el)
        return cleaned

    def _maybe_chunk(self, elements: List[Element]) -> List[Element]:
        if not self.do_chunk:
            return elements
        try:
            return chunk_by_title(
                elements,
                max_characters=self.max_characters,
                new_after_n_chars=self.new_after_n_chars,
                combine_text_under_n_chars=self.combine_under_n,
            )
        except Exception as e:
            logger.error(f"chunk_by_title failed: {e}")
            return elements

    # ------------ main API ------------
    def parse_file(self, filename: str) -> List[Element]:
        tmp_path: Optional[str] = None
        try:
            # copy to a private tmp path to avoid external file locks
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                with open(filename, "rb") as src:
                    tmp.write(src.read())
                tmp_path = tmp.name

            # primary: hi_res with ONNX
            try:
                elements = partition_pdf(
                    filename=tmp_path,
                    strategy="hi_res",
                    hi_res_model_name=self.hi_res_model_name,
                    infer_table_structure=self.infer_table_structure,
                    include_page_breaks=self.include_page_breaks,
                    extract_images_in_pdf=self.extract_images_in_pdf,
                    languages=list(self.languages),
                )
            except Exception as e:
                logger.warning(f"hi_res failed ({e}); retrying with strategy='fast'")
                elements = []

            # fallback if empty/failed: fast strategy
            if not elements:
                try:
                    elements = partition_pdf(
                        filename=tmp_path,
                        strategy="fast",
                        include_page_breaks=self.include_page_breaks,
                        extract_images_in_pdf=self.extract_images_in_pdf,
                        languages=list(self.languages),
                    )
                except Exception as e:
                    logger.error(f"fast fallback failed: {e}")
                    return []

            elements = self._maybe_clean_elements(elements)
            elements = self._maybe_chunk(elements)
            return elements

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def parse_content(self, pdf_content: bytes) -> List[Element]:
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name
            return self.parse_file(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def create_pdf_parser(config: Optional[dict] = None) -> ONNXPDFParser:
    cfg = config or {}
    return ONNXPDFParser(
        hi_res_model_name=cfg.get("hi_res_model_name", "detectron2_onnx"),
        languages=cfg.get("languages", ("eng", "fra")),
        infer_table_structure=cfg.get("infer_table_structure", True),
        include_page_breaks=cfg.get("include_page_breaks", True),
        extract_images_in_pdf=cfg.get("extract_images_in_pdf", False),
        do_clean=cfg.get("do_clean", True),
        do_chunk=cfg.get("do_chunk", False),  # keep False if you chunk later in the pipeline
        max_characters=cfg.get("max_characters", 1500),
        new_after_n_chars=cfg.get("new_after_n_chars", 1400),
        combine_under_n=cfg.get("combine_under_n", 250),
    )
