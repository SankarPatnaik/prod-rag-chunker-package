from __future__ import annotations

import re
from collections import Counter
from typing import List, Sequence, Tuple

from prod_rag.chunking.tokenizer import TokenCounter
from prod_rag.models import Chunk, ChunkingConfig, ChunkingStrategy, DocumentSection, ParentChunk
from prod_rag.utils.hashing import stable_id


class SentenceSplitter:
    SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

    def split(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if "\t" in text or "\n|" in text:
            return [text]
        return [p.strip() for p in self.SENTENCE_RE.split(text) if p.strip()]


class HierarchicalChunker:
    def __init__(self, token_counter: TokenCounter, config: ChunkingConfig) -> None:
        self.token_counter = token_counter
        self.config = config
        self.splitter = SentenceSplitter()

    def effective_chunk_budget(self) -> int:
        model_budget = self.config.max_input_tokens - self.config.reserved_prompt_tokens - self.config.reserved_output_tokens
        if model_budget <= 256:
            raise ValueError("Remaining token budget too small after reserves")
        hard_cap = self.config.hard_max_chunk_tokens or model_budget
        return min(self.config.target_chunk_tokens, hard_cap, model_budget)

    def chunk_sections(self, document_id: str, source: str, sections: Sequence[DocumentSection]) -> Tuple[List[Chunk], List[ParentChunk]]:
        chunks: List[Chunk] = []
        parents: List[ParentChunk] = []
        budget = self.effective_chunk_budget()
        seq = 0
        for section in sections:
            child = self._chunk_single_section(document_id, source, section, seq, budget)
            seq += len(child)
            chunks.extend(child)
            parents.extend(self._build_parent_chunks(document_id, source, section, child))
        return chunks, parents

    def _chunk_single_section(self, document_id: str, source: str, section: DocumentSection, start_seq: int, budget: int) -> List[Chunk]:
        if section.block_type == "table" and self.config.enable_table_isolation:
            return self._chunk_table(document_id, source, section, start_seq, budget)

        units = self._units_for_strategy(section.text)
        chunks: List[Chunk] = []
        cur: List[str] = []
        cur_tokens = 0
        pending_start = section.char_start

        for unit in units:
            unit_tokens = self.token_counter.count(unit)
            if unit_tokens > budget:
                if cur:
                    chunks.append(self._make_chunk(document_id, source, section, start_seq + len(chunks), " ".join(cur).strip(), pending_start))
                    cur, cur_tokens = [], 0
                chunks.extend(self._split_oversized(document_id, source, section, unit, start_seq + len(chunks), budget))
                continue

            if cur and cur_tokens + unit_tokens > budget:
                chunk_text = " ".join(cur).strip()
                chunks.append(self._make_chunk(document_id, source, section, start_seq + len(chunks), chunk_text, pending_start))
                overlap = self._take_overlap_tail(cur)
                pending_start = max(section.char_start, pending_start + len(chunk_text) - len(" ".join(overlap)))
                cur = overlap + [unit]
                cur_tokens = self.token_counter.count(" ".join(cur))
            else:
                cur.append(unit)
                cur_tokens += unit_tokens

        if cur:
            chunks.append(self._make_chunk(document_id, source, section, start_seq + len(chunks), " ".join(cur).strip(), pending_start))

        merged = self._merge_tiny_tail(chunks)
        if self.config.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_merge(merged, budget)
        return merged

    def _units_for_strategy(self, text: str) -> List[str]:
        if self.config.strategy in {ChunkingStrategy.SECTION_AWARE, ChunkingStrategy.HIERARCHICAL}:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if paragraphs:
                return paragraphs
        return self.splitter.split(text)

    def _chunk_table(self, document_id: str, source: str, section: DocumentSection, start_seq: int, budget: int) -> List[Chunk]:
        rows = [r for r in section.text.split("\n") if r.strip()]
        if not rows:
            return []
        header = rows[0]
        out: List[Chunk] = []
        cur: List[str] = []
        for row in rows:
            candidate_rows = cur + [row]
            candidate = "\n".join(candidate_rows)
            if cur and self.token_counter.count(candidate) > budget:
                text = "\n".join(cur)
                if not text.startswith(header):
                    text = header + "\n" + text
                out.append(self._make_chunk(document_id, source, section, start_seq + len(out), text, section.char_start))
                cur = [row]
            else:
                cur.append(row)
        if cur:
            text = "\n".join(cur)
            if not text.startswith(header):
                text = header + "\n" + text
            out.append(self._make_chunk(document_id, source, section, start_seq + len(out), text, section.char_start))
        return out

    def _split_oversized(self, document_id: str, source: str, section: DocumentSection, unit: str, start_seq: int, budget: int) -> List[Chunk]:
        words = unit.split()
        out: List[Chunk] = []
        cur: List[str] = []
        for word in words:
            candidate = " ".join(cur + [word]).strip()
            if cur and self.token_counter.count(candidate) > budget:
                out.append(self._make_chunk(document_id, source, section, start_seq + len(out), " ".join(cur).strip(), section.char_start))
                cur = self._overlap_from_words(cur) + [word]
            else:
                cur.append(word)
        if cur:
            out.append(self._make_chunk(document_id, source, section, start_seq + len(out), " ".join(cur).strip(), section.char_start))
        return out

    def _take_overlap_tail(self, units: Sequence[str]) -> List[str]:
        tail: List[str] = []
        tokens = 0
        for sentence in reversed(units):
            t = self.token_counter.count(sentence)
            if tokens + t > self.config.overlap_tokens and tail:
                break
            tail.insert(0, sentence)
            tokens += t
        return tail

    def _overlap_from_words(self, words: Sequence[str]) -> List[str]:
        tail: List[str] = []
        tokens = 0
        for word in reversed(words):
            t = self.token_counter.count(word)
            if tokens + t > self.config.overlap_tokens and tail:
                break
            tail.insert(0, word)
            tokens += t
        return tail

    def _merge_tiny_tail(self, chunks: List[Chunk]) -> List[Chunk]:
        if len(chunks) < 2 or chunks[-1].token_count >= self.config.min_chunk_tokens:
            return chunks
        prev, last = chunks[-2], chunks[-1]
        merged_text = prev.text + "\n" + last.text
        merged_tokens = self.token_counter.count(merged_text)
        if merged_tokens <= max(self.effective_chunk_budget(), prev.token_count):
            prev.text = merged_text
            prev.token_count = merged_tokens
            prev.char_end = last.char_end
            chunks.pop()
        return chunks

    def _semantic_merge(self, chunks: List[Chunk], budget: int) -> List[Chunk]:
        if len(chunks) < 2:
            return chunks
        out: List[Chunk] = [chunks[0]]
        for nxt in chunks[1:]:
            prev = out[-1]
            similarity = self._jaccard_similarity(prev.text, nxt.text)
            if similarity >= self.config.semantic_similarity_threshold and prev.token_count + nxt.token_count <= budget:
                prev.text = f"{prev.text}\n{nxt.text}"
                prev.token_count = self.token_counter.count(prev.text)
                prev.char_end = nxt.char_end
            else:
                out.append(nxt)
        return out

    def _jaccard_similarity(self, a: str, b: str) -> float:
        words_a = Counter(re.findall(r"\w+", a.lower()))
        words_b = Counter(re.findall(r"\w+", b.lower()))
        if not words_a or not words_b:
            return 0.0
        sa, sb = set(words_a.keys()), set(words_b.keys())
        return len(sa & sb) / max(1, len(sa | sb))

    def _build_parent_chunks(self, document_id: str, source: str, section: DocumentSection, children: Sequence[Chunk]) -> List[ParentChunk]:
        if not children:
            return []
        target = self.config.parent_chunk_target_tokens
        out: List[ParentChunk] = []
        bucket: List[Chunk] = []
        bucket_tokens = 0
        for chunk in children:
            if bucket and bucket_tokens + chunk.token_count > target:
                out.append(self._make_parent(document_id, source, section, bucket))
                bucket, bucket_tokens = [chunk], chunk.token_count
            else:
                bucket.append(chunk)
                bucket_tokens += chunk.token_count
        if bucket:
            out.append(self._make_parent(document_id, source, section, bucket))
        return out

    def _make_parent(self, document_id: str, source: str, section: DocumentSection, children: Sequence[Chunk]) -> ParentChunk:
        text = "\n\n".join(c.text for c in children)
        return ParentChunk(
            parent_chunk_id=stable_id(document_id, section.section_id, children[0].chunk_id, children[-1].chunk_id, prefix="p"),
            document_id=document_id,
            heading_path=section.heading_path,
            text=text,
            token_count=self.token_counter.count(text),
            child_chunk_ids=[c.chunk_id for c in children],
            page_start=section.page_start,
            page_end=section.page_end,
            metadata={"block_type": section.block_type, "source": source},
        )

    def _make_chunk(self, document_id: str, source: str, section: DocumentSection, sequence_no: int, text: str, char_start: int) -> Chunk:
        chunk_id = stable_id(document_id, section.section_id, str(sequence_no), text[:80], prefix="c")
        return Chunk(
            chunk_id=chunk_id,
            parent_section_id=section.section_id,
            document_id=document_id,
            heading_path=section.heading_path,
            text=text,
            token_count=self.token_counter.count(text),
            sequence_no=sequence_no,
            page_start=section.page_start,
            page_end=section.page_end,
            char_start=char_start,
            char_end=char_start + len(text),
            block_type=section.block_type,
            metadata={
                "source": source,
                "section": " > ".join(section.heading_path),
                "block_type": section.block_type,
                "chunk_id": chunk_id,
                "parent_id": section.section_id,
                "token_count": self.token_counter.count(text),
            },
        )
