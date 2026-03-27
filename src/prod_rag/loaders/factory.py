from __future__ import annotations

from pathlib import Path

from prod_rag.loaders.docx_loader import DocxLoader
from prod_rag.loaders.html_loader import HTMLLoader
from prod_rag.loaders.markdown_loader import MarkdownLoader
from prod_rag.loaders.pdf_loader import PDFLoader
from prod_rag.loaders.text_loader import TextLoader
from prod_rag.models import LoaderConfig


def get_loader(path: str | Path, config: LoaderConfig):
    p = Path(path)
    kind = config.type
    if kind == "auto":
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            return PDFLoader()
        if suffix == ".docx":
            return DocxLoader()
        if suffix in {".html", ".htm"}:
            return HTMLLoader()
        if suffix in {".md", ".markdown"}:
            return MarkdownLoader()
        return TextLoader()

    if kind == "pdf":
        return PDFLoader()
    if kind == "docx":
        return DocxLoader()
    if kind == "html":
        return HTMLLoader()
    if kind == "markdown":
        return MarkdownLoader()
    return TextLoader()
