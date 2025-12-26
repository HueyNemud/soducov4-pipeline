from typing import Any, Iterator
from pipeline.ocr.schemas import OCRDocument
from transformers import AutoTokenizer
from pipeline.chunking.schemas import Chunk, ChunkLine
from pipeline.logging import logger


class ChunkCreator:
    """Creates chunks from an OCRDocument using a tokenizer to count tokens."""

    _tokenizer: Any = None

    def __init__(
        self,
        document: OCRDocument,
        tokenizer_model: str,
        max_tokens: int,
        margin_lines: int,
        spuriousness_threshold: float,
    ):
        self.document = document
        self.tokenizer_model = tokenizer_model
        self.max_tokens = max_tokens
        self.margin_lines = margin_lines
        self.spuriousness_threshold = spuriousness_threshold

    @property
    def tokenizer(self):
        # Lazy-load to avoid heavy import-time side effects
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
        return self._tokenizer

    def generate(self) -> Iterator[Chunk]:
        """
        Yields (core_indices, margin_indices) where both are list[int].
        """
        document = self.document
        if not document.ocr:
            return

        # Build a flat list of (page, line, token_len) for all pages
        # Optionally reject spurious lines here
        lines = [
            (
                len(self.tokenizer(line.text, add_special_tokens=False)["input_ids"]),
                page_ix,
                line_ix,
                line.text,
                line.confidence,
                line.spuriousness,
                line.layout_ix,
                (
                    document.layout[page_ix].bboxes[line.layout_ix].label
                    if line.layout_ix is not None
                    else None
                ),
            )
            for page_ix, page in enumerate(document.ocr)
            for line_ix, line in enumerate(page.text_lines)
            if not line.spuriousness or line.spuriousness < self.spuriousness_threshold
        ]

        # Build core chunks as ranges [start, end)
        chunk_spans: list[tuple[int, int]] = []

        # The idea is to accumulate lines until we meet a total of max_tokens or we've consumed the whole text.
        # A chunk may contain less than max_tokens if adding the next line would exceed the limit.
        # Lines are never split: a line longer than max_tokens will raise a warning.
        start = 0
        while start < len(lines):
            accumulated_tokens = 0
            end = start

            while end < len(lines):
                token_len = lines[end][0]
                if accumulated_tokens + token_len > self.max_tokens:
                    break
                accumulated_tokens += token_len
                end += 1

            # If end == start, it means the current line alone exceeds max_tokens
            if end == start:
                token_len, page, line, text, *_ = lines[end]
                logger.warning(
                    f"Warning: line at page {page}, line {line} has {token_len} tokens, "
                    f"which exceeds the max_tokens limit of {self.max_tokens}."
                    f"Long line text: {text[:100]}..."
                    f"The line will be included in its own chunk."
                )
                end += 1  # Force to include this long line in its own chunk

            chunk_spans.append((start, end))
            start = end

        # Yield core + margin-after
        for start, end in chunk_spans:

            # Core chunk lines
            core = [
                ChunkLine(
                    page=lines[i][1],
                    line=lines[i][2],
                    text=lines[i][3],
                    confidence=lines[i][4],
                    spuriousness=lines[i][5],
                    layout_ix=lines[i][6],
                    layout_label=lines[i][7],
                    is_margin=False,
                )
                for i in range(start, end)
            ]

            # Margin lines after
            margin_after_end = min(end + self.margin_lines, len(lines))
            margin_after = [
                ChunkLine(
                    page=lines[i][1],
                    line=lines[i][2],
                    text=lines[i][3],
                    confidence=lines[i][4],
                    spuriousness=lines[i][5],
                    layout_ix=lines[i][6],
                    layout_label=lines[i][7],
                    is_margin=True,
                )
                for i in range(end, margin_after_end)
            ]
            yield Chunk(root=core + margin_after)
