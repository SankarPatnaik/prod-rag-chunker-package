from __future__ import annotations

import re
from pathlib import Path

from prod_rag.loaders.base import BaseLoader
from prod_rag.models import LoadedDocument


class HTMLLoader(BaseLoader):
    def load(self, path: str | Path) -> LoadedDocument:
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        text = re.sub(r"<script[\\s\\S]*?</script>", "", raw, flags=re.IGNORECASE)
        text = re.sub(r"<style[\\s\\S]*?</style>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\\s+", " ", text).strip()
        return LoadedDocument(document_id=p.stem, source_path=str(p), text=text, metadata={"type": "html"})
