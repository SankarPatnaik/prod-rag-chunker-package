from __future__ import annotations

from typing import Dict, Sequence
import json


def build_map_prompt(query: str, chunk_text: str, heading_path: Sequence[str], chunk_id: str) -> str:
    heading = " > ".join(heading_path) if heading_path else "No heading"
    return (
        "You are extracting only evidence relevant to the user query from one chunk of a larger document.\n"
        f"Chunk ID: {chunk_id}\n"
        f"Section: {heading}\n\n"
        f"User query: {query}\n\n"
        "Instructions:\n"
        "1. Use only the provided chunk.\n"
        "2. Return concise evidence.\n"
        "3. List exact facts, obligations, numbers, dates, entities, and exceptions.\n"
        "4. If irrelevant, output exactly IRRELEVANT.\n\n"
        f"Chunk:\n{chunk_text}"
    )


def build_reduce_prompt(query: str, mapped_results: Sequence[Dict]) -> str:
    serialized = json.dumps(list(mapped_results), ensure_ascii=False, indent=2)
    return (
        "You are combining evidence extracted from multiple chunks of the same large document.\n"
        f"User query: {query}\n\n"
        "Instructions:\n"
        "1. Merge duplicate facts.\n"
        "2. Resolve conflicts conservatively.\n"
        "3. If evidence is insufficient, say so clearly.\n"
        "4. Cite chunk ids where possible.\n\n"
        f"Mapped evidence:\n{serialized}"
    )
