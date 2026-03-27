from __future__ import annotations

import math
import re
from typing import Optional


class TokenCounter:
    def __init__(self, tokenizer_name: Optional[str] = None) -> None:
        self._tokenizer = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer  # type: ignore
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            except Exception:
                self._tokenizer = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        words = max(1, len(re.findall(r"\S+", text)))
        punctuation = len(re.findall(r"[,:;()\[\]{}]", text))
        return math.ceil(words * 1.33 + punctuation * 0.1)
