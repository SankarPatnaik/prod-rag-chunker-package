from __future__ import annotations

import re
from dataclasses import dataclass

from prod_rag.models import DocumentType


@dataclass
class DocumentSignals:
    heading_frequency: float
    table_density: float
    code_density: float
    citation_density: float
    avg_paragraph_length: float
    structural_richness: float
    token_density: float


class DocumentTypeClassifier:
    LEGAL_PATTERNS = [r"\bhereby\b", r"\bwhereas\b", r"\bclause\s+\d+", r"\bannex(?:ure)?\b"]
    POLICY_PATTERNS = [r"\bpolicy\b", r"\bcompliance\b", r"\bgovernance\b", r"\bcontrol\b"]
    FINANCIAL_PATTERNS = [r"\brevenue\b", r"\bebitda\b", r"\bbalance sheet\b", r"\bnet income\b"]
    EMAIL_PATTERNS = [r"\bfrom:\s", r"\bto:\s", r"\bsubject:\s", r"\bon .* wrote:\b"]
    CHAT_PATTERNS = [r"^\[?\d{1,2}:\d{2}(:\d{2})?\]?", r"^\w+:\s", r"\buser:\b", r"\bassistant:\b"]
    FAQ_PATTERNS = [r"^q[:\-]", r"^question[:\-]", r"^a[:\-]", r"^answer[:\-]"]

    def classify(self, text: str, source: str = "") -> tuple[DocumentType, DocumentSignals]:
        lines = [line for line in text.splitlines() if line.strip()]
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        heading_count = sum(1 for l in lines if re.match(r"^(#{1,6}\s+|\d+\.\s+|[A-Z][A-Z\s]{4,})", l.strip()))
        table_lines = sum(1 for l in lines if "|" in l or "\t" in l)
        code_lines = sum(1 for l in lines if l.strip().startswith(("def ", "class ", "function ", "{")) or "```" in l)
        citations = len(re.findall(r"\[(\d+|[A-Za-z][^\]]{0,20})\]|\([A-Z][A-Za-z]+,\s*\d{4}\)", text))
        tokens_est = max(1, len(re.findall(r"\S+", text)))

        signals = DocumentSignals(
            heading_frequency=heading_count / max(1, len(lines)),
            table_density=table_lines / max(1, len(lines)),
            code_density=code_lines / max(1, len(lines)),
            citation_density=citations / max(1, len(lines)),
            avg_paragraph_length=sum(len(p.split()) for p in paragraphs) / max(1, len(paragraphs)),
            structural_richness=min(1.0, (heading_count + table_lines + code_lines) / max(1, len(lines))),
            token_density=tokens_est / max(1, len(lines)),
        )

        lower = text.lower()
        source_lower = source.lower()
        if source_lower.endswith((".md", ".markdown")):
            return DocumentType.MARKDOWN, signals
        if source_lower.endswith((".html", ".htm")):
            return DocumentType.WIKI, signals
        if source_lower.endswith((".py", ".js", ".java", ".go", ".rs", ".sql")):
            return DocumentType.SOURCE_CODE, signals
        if self._match_any(lower, self.EMAIL_PATTERNS):
            return DocumentType.EMAIL, signals
        if self._match_any(lower, self.FAQ_PATTERNS):
            return DocumentType.FAQ, signals
        if self._match_any(lower, self.CHAT_PATTERNS):
            return DocumentType.CHAT_TRANSCRIPT, signals
        if self._match_any(lower, self.LEGAL_PATTERNS):
            return DocumentType.LEGAL, signals
        if self._match_any(lower, self.POLICY_PATTERNS):
            return DocumentType.POLICY, signals
        if self._match_any(lower, self.FINANCIAL_PATTERNS):
            return DocumentType.FINANCIAL, signals
        if signals.code_density > 0.2:
            return DocumentType.SOURCE_CODE, signals
        if signals.table_density > 0.3:
            return DocumentType.TABULAR, signals
        if signals.heading_frequency > 0.2:
            return DocumentType.TECHNICAL, signals
        return DocumentType.GENERIC, signals

    def _match_any(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(p, text, flags=re.MULTILINE) for p in patterns)
