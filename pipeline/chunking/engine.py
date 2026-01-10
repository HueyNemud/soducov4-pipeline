"""
Document Chunking Service.

This module provides the logic to segment an OCRDocument into manageable chunks
based on token counts, preserving local context via overlapping margins.
"""

from typing import Any, Iterator, List, NamedTuple, Optional
from transformers import AutoTokenizer
from pydantic import Field, dataclasses

from pipeline.ocr.schemas import OCRDocument
from pipeline.chunking.schemas import Chunk, ChunkLine
from pipeline.logging import logger


class TokenizedLine(NamedTuple):
    """Intermediary container for line content and pre-calculated metadata."""

    token_count: int
    page_idx: int
    line_idx: int
    text: str
    confidence: Optional[float]
    spuriousness: Optional[float]
    layout_ix: Optional[int]
    layout_label: Optional[str]


class ChunkCreator:
    """
    Service responsible for partitioning OCR documents into token-limited chunks.

    By encapsulating the parameters and tokenizer within the instance, it provides
    a clean interface for batch processing documents with consistent settings.
    """

    @dataclasses.dataclass
    class Parameters:
        """Configuration for the chunking and filtering engine."""

        tokenizer_model: str = Field(
            default="almanach/camembert-base",
            description="HuggingFace tokenizer model name.",
        )
        max_tokens: int = Field(
            default=2000, description="Maximum tokens allowed per core chunk."
        )
        margin_lines: int = Field(
            default=2,
            description="Number of context lines to include as margin after each chunk.",
        )
        spuriousness_threshold: float = Field(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Threshold above which lines are considered spurious and excluded.",
        )

    def __init__(self, params: Optional[Parameters] = None):
        """Initializes the service with specific runtime parameters."""
        self.params = params or self.Parameters()
        self._tokenizer: Any = None

    @property
    def tokenizer(self) -> Any:
        """Lazy-loads the HuggingFace tokenizer using the configured model."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.params.tokenizer_model)
        return self._tokenizer

    def generate(self, document: OCRDocument) -> Iterator[Chunk]:
        """
        Segments the provided OCRDocument into a sequence of Chunks.

        Yields:
            Iterator[Chunk]: A stream of chunks containing core text and context margins.
        """
        # 1. Pre-process: Filter noise and calculate token weights
        lines = self._get_clean_tokenized_lines(document)
        if not lines:
            return

        # 2. Planning: Calculate chunk boundaries
        spans = self._calculate_chunk_spans(lines)

        # 3. Execution: Build the nested Pydantic models
        for start, end in spans:
            core = [
                self._to_chunk_line(lines[i], is_margin=False)
                for i in range(start, end)
            ]

            # Contextual look-ahead (margin)
            margin_limit = min(end + self.params.margin_lines, len(lines))
            margins = [
                self._to_chunk_line(lines[i], is_margin=True)
                for i in range(end, margin_limit)
            ]

            yield Chunk(root=core + margins)

    def _get_clean_tokenized_lines(self, doc: OCRDocument) -> List[TokenizedLine]:
        """Flatten pages into lines, filtering out spurious entries."""
        prepared_lines: List[TokenizedLine] = []

        for p_idx, page in enumerate(doc.ocr):
            for l_idx, line in enumerate(page.text_lines):
                # Ignore lines identified as scan noise/artifacts
                if (
                    line.spuriousness
                    and line.spuriousness >= self.params.spuriousness_threshold
                ):
                    continue

                token_count = len(
                    self.tokenizer(line.text, add_special_tokens=False)["input_ids"]
                )

                # Resolve layout context
                label = None
                if line.layout_ix is not None:
                    label = doc.layout[p_idx].bboxes[line.layout_ix].label

                prepared_lines.append(
                    TokenizedLine(
                        token_count=token_count,
                        page_idx=p_idx,
                        line_idx=l_idx,
                        text=line.text,
                        confidence=line.confidence,
                        spuriousness=line.spuriousness,
                        layout_ix=line.layout_ix,
                        layout_label=label,
                    )
                )
        return prepared_lines

    def _calculate_chunk_spans(
        self, lines: List[TokenizedLine]
    ) -> List[tuple[int, int]]:
        """Determines the start and end indices for core chunks based on token budget."""
        spans: List[tuple[int, int]] = []
        cursor, total = 0, len(lines)

        while cursor < total:
            accumulated_tokens, end = 0, cursor

            while end < total:
                line_tokens = lines[end].token_count
                if accumulated_tokens + line_tokens > self.params.max_tokens:
                    break
                accumulated_tokens += line_tokens
                end += 1

            # Safety: If a single line is bigger than max_tokens, put it in its own chunk
            if end == cursor:
                logger.warning(
                    f"Oversized line at page {lines[cursor].page_idx} (tokens: {lines[cursor].token_count})"
                )
                end += 1

            spans.append((cursor, end))
            cursor = end

        return spans

    def _to_chunk_line(self, data: TokenizedLine, is_margin: bool) -> ChunkLine:
        """Converts internal TokenizedLine tuple to the public ChunkLine schema."""
        return ChunkLine(
            page=data.page_idx,
            line=data.line_idx,
            text=data.text,
            confidence=data.confidence,
            spuriousness=data.spuriousness,
            layout_ix=data.layout_ix,
            layout_label=data.layout_label,
            is_margin=is_margin,
        )
