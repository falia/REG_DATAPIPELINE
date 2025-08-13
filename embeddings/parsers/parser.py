from abc import ABC, abstractmethod
import os
import tempfile
import logging
import json
from typing import List

from scrapy.http import HtmlResponse
from unstructured.partition.html import partition_html
from unstructured.documents.elements import Element

# PDF pipeline pieces
from .ONNXPDFParser import ONNXPDFParser
from .PDFRemoveHeaderFooter import HeaderFooterRedactor

logger = logging.getLogger(__name__)


class DocumentParser(ABC):
    @abstractmethod
    def can_process(self, url: str, content_type: str = None) -> bool:
        ...

    @abstractmethod
    def parse(self, content: bytes, url: str, content_type: str):
        ...


class EurlexHTMLParser(DocumentParser):
    def can_process(self, url: str, content_type: str = None) -> bool:
        url_match = "eur-lex.europa.eu" in url and not url.lower().endswith(".pdf")
        content_type_match = content_type and "text/html" in content_type
        return url_match and content_type_match

    def parse(self, content: bytes, url: str, content_type: str):
        # Extract main content region
        response = HtmlResponse(url=url, body=content, encoding="utf-8")
        raw_sections = response.css("div.PP4Contents").getall()
        raw_html = "\n\n".join(raw_sections)

        if not raw_html:
            return []

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(raw_html)
            temp_path = temp_file.name

        try:
            elements = partition_html(
                filename=temp_path,
                include_page_breaks=False,      # HTML has no pages
                infer_table_structure=True,     # keep tables as elements
                skip_headers_and_footers=True,  # drop nav/footer chrome
                languages=["eng"],
                chunking_strategy=None,         # <-- NO CHUNKING
            )
            return elements
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


class CSSFHTMLParser(DocumentParser):
    def can_process(self, url: str, content_type: str = None) -> bool:
        url_match = "www.cssf.lu" in url and not url.lower().endswith(".pdf")
        content_type_match = content_type and "text/html" in content_type
        return url_match and content_type_match

    def parse(self, content: bytes, url: str, content_type: str):
        response = HtmlResponse(url=url, body=content, encoding="utf-8")
        raw_sections = response.css("div.content-section").getall()
        raw_html = "\n\n".join(raw_sections)

        if not raw_html:
            return []

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(raw_html)
            temp_path = temp_file.name

        try:
            elements = partition_html(
                filename=temp_path,
                include_page_breaks=False,
                infer_table_structure=True,
                skip_headers_and_footers=True,
                languages=["eng"],
                chunking_strategy=None,         # <-- NO CHUNKING
            )
            return elements
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


class PDFParserPipeline(DocumentParser):

    def __init__(
        self,
        redactor: "HeaderFooterRedactor | None" = None,
        pdf_parser: "ONNXPDFParser | None" = None,
        enable_redaction: bool = False,   # <-- default OFF
    ):
        # Toggle via param or env var
        env_disable = os.getenv("DISABLE_PDF_REDACTION", "1").lower() in ("1", "true", "yes")
        self.enable_redaction = enable_redaction and not env_disable

        # Only build a redactor if we actually plan to use it
        self.redactor = None
        if self.enable_redaction:
            self.redactor = redactor or HeaderFooterRedactor(
                top_k=5, bottom_k=3, win=8,
                header_th=0.65, footer_th=0.65, rank_th=0.55,
                pad=2.0, black=True,
            )

        # Make sure your ONNX parser does not chunk internally
        self.pdf_parser = pdf_parser or ONNXPDFParser()  # should use chunking_strategy=None

    def can_process(self, url: str, content_type: str | None = None) -> bool:
        url_match = url.lower().endswith(".pdf")
        content_type_match = bool(content_type and "application/pdf" in content_type.lower())
        return url_match or content_type_match

    def parse(self, content: bytes, url: str, content_type: str) -> List[Element]:
        # Write original bytes to a temp file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            orig_path = tmp.name

        sanitized_path = None
        try:
            # If redaction is enabled, try it; otherwise parse the original file
            if self.enable_redaction and self.redactor is not None:
                try:
                    sanitized_path = self.redactor.redact_to_temp(orig_path)
                    path_to_parse = sanitized_path
                except Exception as e:
                    # Fallback to original if redaction fails
                    print(f"[PDFParserPipeline] Redaction failed ({e}); using original PDF.")
                    path_to_parse = orig_path
            else:
                # Redaction disabled: work with headers/footers as-is
                path_to_parse = orig_path

            return self.pdf_parser.parse_file(path_to_parse)

        finally:
            # Clean up temps (avoid deleting the same path twice)
            for p in (orig_path, sanitized_path):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass


class GenericHTMLParser(DocumentParser):
    """Fallback parser for HTML content that doesn't match specific parsers."""

    def can_process(self, url: str, content_type: str = None) -> bool:
        return content_type and "text/html" in content_type

    def parse(self, content: bytes, url: str, content_type: str):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".html", delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            elements = partition_html(
                filename=temp_path,
                include_page_breaks=False,
                infer_table_structure=True,
                skip_headers_and_footers=True,
                languages=["eng", "fra"],
                chunking_strategy=None,         # <-- NO CHUNKING
            )
            return elements
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


# --- Parser manager (or factory) ---
class DocumentProcessor:
    def __init__(self, parsers):
        self.parsers = parsers

    def process(self, content: bytes, url: str, content_type: str):
        for parser in self.parsers:
            if parser.can_process(url, content_type):
                logger.info(f"Using {parser.__class__.__name__} for {url}")
                return parser.parse(content, url, content_type)

        logger.warning(f"No parser available for URL: {url}, content-type: {content_type}")
        return []
