from __future__ import annotations

import re
from typing import List, Optional

from prod_rag.models import DocumentSection
from prod_rag.utils.hashing import stable_id


class TextNormalizer:
    def normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class StructureParser:
    HEADING_PATTERNS = [
        re.compile(r"^(#{1,6})\s+(.+)$"),
        re.compile(r"^((?:\d+\.){1,6}\d*|[A-Z]\.|[IVXLC]+\.)\s+(.+)$"),
        re.compile(r"^([A-Z][A-Z0-9\-\s&,/]{4,})$"),
    ]
    BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
    TABLE_ROW_RE = re.compile(r"\s{2,}|\t|\|")

    def parse(self, document_id: str, text: str) -> List[DocumentSection]:
        lines = text.split("\n")
        sections: List[DocumentSection] = []
        heading_stack: List[str] = []
        buffer: List[str] = []
        block_type = "body"
        char_pos = 0
        section_start = 0
        current_section_id = stable_id(document_id, "0", prefix="sec")

        def flush(end_char: int) -> None:
            nonlocal buffer, section_start, current_section_id, block_type
            body = "\n".join(buffer).strip()
            if not body:
                buffer = []
                section_start = end_char
                return
            sections.append(DocumentSection(
                section_id=current_section_id,
                heading_path=list(heading_stack),
                text=body,
                block_type=block_type,
                char_start=section_start,
                char_end=end_char,
            ))
            buffer = []
            section_start = end_char

        for idx, line in enumerate(lines):
            stripped = line.strip()
            next_char = char_pos + len(line) + 1
            heading = self._extract_heading(stripped)
            if heading:
                flush(char_pos)
                heading_stack = heading
                current_section_id = stable_id(document_id, str(idx), prefix="sec")
                block_type = "heading_section"
                char_pos = next_char
                continue
            if not stripped:
                if buffer and buffer[-1] != "":
                    buffer.append("")
                char_pos = next_char
                continue
            inferred = "body"
            if self.BULLET_RE.search(line):
                inferred = "list"
            elif self._looks_like_table(line):
                inferred = "table"
            if buffer and inferred != block_type and block_type in {"table", "list"}:
                flush(char_pos)
                current_section_id = stable_id(document_id, str(idx), prefix="sec")
            block_type = inferred
            buffer.append(line)
            char_pos = next_char
        flush(len(text))
        return sections

    def _extract_heading(self, line: str) -> Optional[List[str]]:
        if not line:
            return None
        for pattern in self.HEADING_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            if pattern is self.HEADING_PATTERNS[0]:
                return [f"L{len(match.group(1))}: {match.group(2).strip()}"]
            if pattern is self.HEADING_PATTERNS[1]:
                return [f"{match.group(1).strip()} {match.group(2).strip()}"]
            title = match.group(1).strip()
            if len(title.split()) <= 14:
                return [title]
        return None

    def _looks_like_table(self, line: str) -> bool:
        return len(line) >= 12 and bool(self.TABLE_ROW_RE.search(line)) and (sum(ch.isdigit() for ch in line) + line.count("|")) >= 2
