from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from prod_rag.loaders.base import BaseLoader
from prod_rag.models import LoadedDocument


class PDFLoader(BaseLoader):
    def load(self, path: str | Path) -> LoadedDocument:
        p = Path(path)
        reader = PdfReader(str(p))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(f"[[PAGE_{i}]]\n{text}")
        return LoadedDocument(document_id=p.stem, source_path=str(p), text="\n\n".join(pages), metadata={"type": "pdf", "num_pages": len(reader.pages)})
