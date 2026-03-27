from __future__ import annotations

import hashlib


def stable_id(*parts: str, prefix: str = "id") -> str:
    raw = ":".join(parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"
