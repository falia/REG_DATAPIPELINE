import re
import tempfile
import os
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, Text
from unstructured.chunking.title import chunk_by_title
import logging

logger = logging.getLogger(__name__)


class ONNXPDFParser:
    """Thread-safe Unstructured hi_res (detectron2_onnx) + light cleanup + chunking."""

    # Chunking tunables
    MAX_CHARACTERS = 1500
    NEW_AFTER_N_CHARS = 1400
    COMBINE_UNDER_N = 250

    def __init__(self, 
                 max_characters: int = 1500,
                 new_after_n_chars: int = 1400, 
                 combine_under_n: int = 250):
        """Initialize with configurable chunking parameters."""
        self.MAX_CHARACTERS = max_characters
        self.NEW_AFTER_N_CHARS = new_after_n_chars
        self.COMBINE_UNDER_N = combine_under_n

    def parse_file(self, filename: str) -> List[Element]:
        """Parse a PDF file and return chunked elements. Thread-safe implementation."""
        try:
            # Use a temporary file to avoid any potential file locking issues in parallel processing
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                # Copy the input file to a temporary location
                with open(filename, 'rb') as src:
                    temp_file.write(src.read())
                temp_filename = temp_file.name

            try:
                elements: List[Element] = partition_pdf(
                    filename=temp_filename,
                    strategy="hi_res",
                    hi_res_model_name="detectron2_onnx",
                    include_page_breaks=True,
                    languages=["eng"],            # focusing on English for now
                    infer_table_structure=True,
                    extract_images_in_pdf=False,  # no table image crops
                    chunking_strategy=None,       # clean & chunk after
                )

                if not elements:
                    logger.warning(f"No elements extracted from PDF: {filename}")
                    return []

                # minimal cleanup
                cleaned: List[Element] = []
                for el in elements:
                    if isinstance(el, Text) and getattr(el, "text", None):
                        t = re.sub(r"(\w)-\n(\w)", r"\1\2", el.text)     # join hyphenated breaks
                        t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)           # collapse single newlines
                        t = t.strip()  # remove leading/trailing whitespace
                        
                        if t:  # only add non-empty text
                            el.text = t
                            cleaned.append(el)
                    else:
                        # Keep non-text elements (tables, etc.)
                        cleaned.append(el)

                if not cleaned:
                    logger.warning(f"No content after cleaning PDF: {filename}")
                    return []

                # chunk by title with error handling
                try:
                    chunked = chunk_by_title(
                        cleaned,
                        max_characters=self.MAX_CHARACTERS,
                        new_after_n_chars=self.NEW_AFTER_N_CHARS,
                        combine_text_under_n_chars=self.COMBINE_UNDER_N,
                    )
                    logger.info(f"Successfully processed PDF {filename}: {len(elements)} elements -> {len(cleaned)} cleaned -> {len(chunked)} chunks")
                    return chunked
                    
                except Exception as chunk_error:
                    logger.error(f"Error chunking PDF {filename}: {chunk_error}")
                    # Fallback: return cleaned elements without chunking
                    return cleaned

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_filename)
                except OSError:
                    pass

        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}", exc_info=True)
            return []

    def parse_content(self, pdf_content: bytes, source_url: str = "unknown") -> List[Element]:
        """Parse PDF content from bytes. Useful for processing downloaded content."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_filename = temp_file.name

            try:
                return self.parse_file(temp_filename)
            finally:
                try:
                    os.unlink(temp_filename)
                except OSError:
                    pass

        except Exception as e:
            logger.error(f"Error processing PDF content from {source_url}: {e}", exc_info=True)
            return []


# Optional: Create a factory function for creating parser instances with different configurations
def create_pdf_parser(config: dict = None) -> ONNXPDFParser:
    """Factory function to create PDF parser with custom configuration."""
    if config is None:
        config = {}
    
    return ONNXPDFParser(
        max_characters=config.get('max_characters', 1500),
        new_after_n_chars=config.get('new_after_n_chars', 1400),
        combine_under_n=config.get('combine_under_n', 250)
    )