from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from prod_rag.models import LoadedDocument


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str | Path) -> LoadedDocument:
        raise NotImplementedError
