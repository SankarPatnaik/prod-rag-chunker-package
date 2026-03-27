from __future__ import annotations

from pathlib import Path

from prod_rag.loaders.base import BaseLoader
from prod_rag.models import LoadedDocument


class TextLoader(BaseLoader):
    def load(self, path: str | Path) -> LoadedDocument:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        return LoadedDocument(document_id=p.stem, source_path=str(p), text=text, metadata={"type": "text"})
