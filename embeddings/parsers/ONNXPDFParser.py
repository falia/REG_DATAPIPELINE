import re
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, Text
from unstructured.chunking.title import chunk_by_title

class ONNXPDFParser:
    """Unstructured hi_res (detectron2_onnx) + light cleanup + chunking."""

    # Chunking tunables
    MAX_CHARACTERS = 1500
    NEW_AFTER_N_CHARS = 1400
    COMBINE_UNDER_N = 250

    def parse_file(self, filename: str) -> List[Element]:
        elements: List[Element] = partition_pdf(
            filename=filename,
            strategy="hi_res",
            hi_res_model_name="detectron2_onnx",
            include_page_breaks=True,
            languages=["eng"],            # focusing on English for now
            infer_table_structure=True,
            extract_images_in_pdf=False,  # no table image crops
            chunking_strategy=None,       # clean & chunk after
        )

        # minimal cleanup
        cleaned: List[Element] = []
        for el in elements:
            if isinstance(el, Text) and getattr(el, "text", None):
                t = re.sub(r"(\w)-\n(\w)", r"\1\2", el.text)     # join hyphenated breaks
                t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)           # collapse single newlines
                el.text = t
            cleaned.append(el)

        # chunk by title
        return chunk_by_title(
            cleaned,
            max_characters=self.MAX_CHARACTERS,
            new_after_n_chars=self.NEW_AFTER_N_CHARS,
            combine_text_under_n_chars=self.COMBINE_UNDER_N,
        )
