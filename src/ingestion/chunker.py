import math
from typing import List, Optional

import tiktoken

from src.ingestion.models import Chunk, ChunkMetadata, DocumentElement, RawDocument


class StructureAwareChunker:
    """Structure-aware recursive text chunker targeting 300-500 tokens with ~15% overlap."""

    DEFAULT_ENCODING = "cl100k_base"

    def __init__(
        self,
        min_tokens: int = 300,
        max_tokens: int = 500,
        overlap_ratio: float = 0.15,
        encoding_name: str = DEFAULT_ENCODING,
    ):
        if min_tokens > max_tokens:
            raise ValueError(f"min_tokens ({min_tokens}) cannot exceed max_tokens ({max_tokens})")
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError(f"overlap_ratio ({overlap_ratio}) must be between 0.0 and 1.0")

        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio
        self.target_overlap_tokens = int(max_tokens * overlap_ratio)

        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken encoder, with fallback word-based estimation."""
        if not text:
            return 0
        if self.encoder:
            return len(self.encoder.encode(text))
        # Fallback estimation (~4 characters per token or 0.75 words per token)
        return max(1, math.ceil(len(text) / 4))

    def chunk_document(self, doc: RawDocument) -> List[Chunk]:
        """Chunk a RawDocument into a list of Chunk objects."""
        if not doc.elements:
            return []

        # Combine all text to check total document size
        full_text = "\n\n".join(e.text for e in doc.elements if e.text.strip())
        total_doc_tokens = self.count_tokens(full_text)

        # Handle short document (< max_tokens): return as single chunk
        if total_doc_tokens <= self.max_tokens:
            first_element = doc.elements[0]
            last_element = doc.elements[-1]
            section = first_element.section_string or last_element.section_string
            heading = first_element.heading or last_element.heading
            page_number = first_element.page_number

            metadata = ChunkMetadata(
                filename=doc.filename,
                doc_type=doc.doc_type.value,
                page_number=page_number,
                section=section,
                heading=heading,
                chunk_index=0,
                total_chunks=1,
                token_count=total_doc_tokens,
                char_count=len(full_text),
            )

            return [
                Chunk(
                    id=f"{doc.filename}_chunk_0",
                    text=full_text,
                    source=str(doc.filepath),
                    metadata=metadata,
                )
            ]

        # Break elements into smaller structural units if needed
        units: List[DocumentElement] = []
        for element in doc.elements:
            element_tokens = self.count_tokens(element.text)
            if element_tokens > self.max_tokens:
                sub_units = self._split_element_recursively(element, self.max_tokens)
                units.extend(sub_units)
            else:
                units.append(element)

        # Accumulate units into chunks with rolling overlap
        raw_chunks: List[dict] = []
        current_units: List[DocumentElement] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit.text)

            # If adding unit exceeds max_tokens and we already reached min_tokens threshold
            if current_tokens + unit_tokens > self.max_tokens and current_tokens >= self.min_tokens:
                # Seal current chunk
                chunk_dict = self._create_chunk_payload(current_units)
                raw_chunks.append(chunk_dict)

                # Prepare overlap for next chunk
                overlap_units = self._get_overlap_units(current_units)
                current_units = list(overlap_units)
                current_tokens = sum(self.count_tokens(u.text) for u in current_units)

            current_units.append(unit)
            current_tokens += unit_tokens

        # Flush remaining units
        if current_units:
            chunk_dict = self._create_chunk_payload(current_units)
            # Avoid duplicate if last chunk equals previous overlap exactly
            if not raw_chunks or raw_chunks[-1]["text"] != chunk_dict["text"]:
                raw_chunks.append(chunk_dict)

        # Construct final Chunk instances with correct total_chunks and chunk_index
        total_chunks = len(raw_chunks)
        final_chunks: List[Chunk] = []

        for idx, payload in enumerate(raw_chunks):
            metadata = ChunkMetadata(
                filename=doc.filename,
                doc_type=doc.doc_type.value,
                page_number=payload["page_number"],
                section=payload["section"],
                heading=payload["heading"],
                chunk_index=idx,
                total_chunks=total_chunks,
                token_count=payload["token_count"],
                char_count=payload["char_count"],
            )

            chunk = Chunk(
                id=f"{doc.filename}_chunk_{idx}",
                text=payload["text"],
                source=str(doc.filepath),
                metadata=metadata,
            )
            final_chunks.append(chunk)

        return final_chunks

    def _create_chunk_payload(self, units: List[DocumentElement]) -> dict:
        text = "\n\n".join(u.text for u in units).strip()
        first = units[0] if units else DocumentElement(text="")

        # Determine section and heading context from elements
        section = first.section_string
        heading = first.heading
        page_number = first.page_number

        return {
            "text": text,
            "section": section,
            "heading": heading,
            "page_number": page_number,
            "token_count": self.count_tokens(text),
            "char_count": len(text),
        }

    def _get_overlap_units(self, units: List[DocumentElement]) -> List[DocumentElement]:
        """Collect trailing units that fit within target_overlap_tokens."""
        overlap_units: List[DocumentElement] = []
        accumulated_tokens = 0

        for unit in reversed(units):
            tokens = self.count_tokens(unit.text)
            if accumulated_tokens + tokens <= self.target_overlap_tokens or not overlap_units:
                overlap_units.insert(0, unit)
                accumulated_tokens += tokens
            else:
                break

        return overlap_units

    def _split_element_recursively(self, element: DocumentElement, max_tokens: int) -> List[DocumentElement]:
        """Recursively split a large DocumentElement text on separators (\n\n, \n, . , ' ')."""
        text = element.text
        tokens = self.count_tokens(text)
        if tokens <= max_tokens:
            return [element]

        separators = ["\n\n", "\n", ". ", " "]
        chosen_sep = None
        for sep in separators:
            if sep in text:
                chosen_sep = sep
                break

        if not chosen_sep:
            # Fallback character split
            mid = len(text) // 2
            left_text, right_text = text[:mid], text[mid:]
            left_elem = DocumentElement(
                text=left_text,
                heading=element.heading,
                section_path=element.section_path,
                page_number=element.page_number,
            )
            right_elem = DocumentElement(
                text=right_text,
                heading=element.heading,
                section_path=element.section_path,
                page_number=element.page_number,
            )
            return self._split_element_recursively(left_elem, max_tokens) + self._split_element_recursively(
                right_elem, max_tokens
            )

        parts = text.split(chosen_sep)
        result: List[DocumentElement] = []
        current_part_lines: List[str] = []

        for part in parts:
            test_text = chosen_sep.join(current_part_lines + [part])
            if self.count_tokens(test_text) <= max_tokens:
                current_part_lines.append(part)
            else:
                if current_part_lines:
                    chunk_text = chosen_sep.join(current_part_lines)
                    result.append(
                        DocumentElement(
                            text=chunk_text,
                            heading=element.heading,
                            section_path=element.section_path,
                            page_number=element.page_number,
                        )
                    )
                    current_part_lines = [part]
                else:
                    # Single part is still larger than max_tokens, recurse deeper
                    sub_elem = DocumentElement(
                        text=part,
                        heading=element.heading,
                        section_path=element.section_path,
                        page_number=element.page_number,
                    )
                    result.extend(self._split_element_recursively(sub_elem, max_tokens))

        if current_part_lines:
            chunk_text = chosen_sep.join(current_part_lines)
            result.append(
                DocumentElement(
                    text=chunk_text,
                    heading=element.heading,
                    section_path=element.section_path,
                    page_number=element.page_number,
                )
            )

        return result
