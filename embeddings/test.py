from pathlib import Path
import sys
from parsers.parser import EurlexHTMLParser, CSSFHTMLParser, PDFParserPipeline, DocumentProcessor

def preview_text(txt: str, max_len: int = 300000) -> str:
    if not txt:
        return ""
    txt = " ".join(txt.split())  # collapse whitespace/newlines
    return txt if len(txt) <= max_len else txt[:max_len] + "…"

def main(pdf_path: str):
    p = Path(pdf_path)
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    content = p.read_bytes()

    processor = DocumentProcessor(parsers=[EurlexHTMLParser(), CSSFHTMLParser(), PDFParserPipeline()])
    # content_type helps the parser choose the right branch
    elements = processor.process(content=content, url=str(p), content_type="application/pdf")

    print(f"Parsed {len(elements)} elements from: {p}\n")

    for i, el in enumerate(elements, 1):
        meta = getattr(el, "metadata", None)
        page = getattr(meta, "page_number", None)
        etype = el.__class__.__name__
        category = getattr(el, "category", None)
        text = getattr(el, "text", None)

        print(f"[{i:04d}] page={page} type={etype} category={category}")
        if text:
            print(preview_text(text))
        else:
            # Some elements (e.g., Table/Image) may not have text
            print("(no text)")
        print("-" * 80)

if __name__ == "__main__":
    # Default path you gave; override with: python test_pdf_parser.py "C:/Users/ftta.pdf"
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/faton/Downloads/cssf22_805eng.pdf"
    main(pdf_path)