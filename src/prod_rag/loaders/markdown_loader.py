from __future__ import annotations

import re
from pathlib import Path

from prod_rag.loaders.base import BaseLoader
from prod_rag.models import LoadedDocument


class MarkdownLoader(BaseLoader):
    def load(self, path: str | Path) -> LoadedDocument:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        text = re.sub(r"```[\\s\\S]*?```", " ", text)
        text = re.sub(r"`([^`]*)`", r"\\1", text)
        text = re.sub(r"!\\[[^\\]]*\\]\\([^)]*\\)", " ", text)
        text = re.sub(r"\\[([^\\]]+)\\]\\([^)]*\\)", r"\\1", text)
        return LoadedDocument(document_id=p.stem, source_path=str(p), text=text, metadata={"type": "markdown"})
