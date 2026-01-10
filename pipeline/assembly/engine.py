from typing import Iterator
from thefuzz import fuzz

from pipeline.assembly.schemas import RichItem, RichStructured
from pipeline.chunking.schemas import Chunk, Chunks
from pipeline.extraction.postprocessing import _recursive_string_repr
from pipeline.extraction.schemas import (
    Structured,
    StructuredCompact,
    StructuredSeq,
)
from pipeline.ocr.schemas import OCRDocument
from pipeline.logging import logger


class Assembler:
    """
    Reconstructs rich document structures by mapping LLM-extracted data
    back to original OCR geometric information and source text.
    """

    def assemble_multiple(
        self,
        ocr_doc: OCRDocument,
        chunks: Chunks,
        structured_sequence: StructuredSeq,
        strict_mode: bool = False,
    ) -> Iterator[RichStructured]:
        """
        Processes a sequence of chunks and their corresponding extractions.

        Args:
            ocr_doc: The source OCR document containing geometric bounding boxes.
            chunks: The list of text chunks used for extraction.
            structured_sequence: The structured objects returned by the LLM.
            strict_mode: If True, raises errors on line index mismatches.
        """
        if len(chunks) < len(structured_sequence):
            raise ValueError(
                "Mismatched sequence: More extractions than source chunks."
            )

        if len(chunks) > len(structured_sequence):
            logger.error(
                "Mismatched sequence: Some chunks lack corresponding extractions."
            )

        for chunk, extraction in zip(chunks, structured_sequence):
            # Ensure we are working with expanded schemas
            full_struct = (
                extraction.expand()
                if isinstance(extraction, StructuredCompact)
                else extraction
            )

            yield self.assemble_single(
                ocr_doc=ocr_doc,
                chunk=chunk,
                structured=full_struct,
                strict_mode=strict_mode,
            )

    def assemble_single(
        self,
        ocr_doc: OCRDocument,
        chunk: Chunk,
        structured: Structured,
        strict_mode: bool = False,
        skip_artifacts: bool = True,
    ) -> RichStructured:
        """
        Maps a single structured extraction to its physical OCR location and raw text.

        Args:
            ocr_doc: Source OCR data for bounding box retrieval.
            chunk: The specific text chunk containing line metadata.
            structured: The LLM extraction for this chunk.
            strict_mode: Whether to fail on out-of-bounds line indices.
            skip_artifacts: If True, filters out margin lines and chunking noise.
        """
        rich_items = []

        for index, item in enumerate(structured.items):
            # 1. Filter out common chunking artifacts (e.g., empty leading entries)
            if (
                skip_artifacts
                and item.cat == "ent"
                and index == 0
                and not getattr(item, "name", None)
            ):
                logger.debug(
                    f"Filtering likely chunking artifact at index {index}: {item}"
                )
                continue

            # 2. Resolve line indices to actual ChunkLine objects
            resolved_lines = []
            for line_idx in item.lines:
                try:
                    resolved_lines.append(chunk[line_idx])
                except IndexError:
                    msg = f"Line index {line_idx} out of range (Chunk size: {len(chunk)})."
                    if strict_mode:
                        raise IndexError(msg)
                    logger.error(f"{msg} Skipping line.")

            # 3. Ignore items that consist exclusively of margin/overlap text
            if skip_artifacts and all(line.is_margin for line in resolved_lines):
                continue

            # 4. Reconstruct raw text and calculate alignment confidence
            raw_source_text = "\n".join(line.text for line in resolved_lines).strip()
            item_content_repr = _recursive_string_repr(
                item, exclude_fields=["lines", "cat"]
            ).strip()

            alignment_score = fuzz.ratio(item_content_repr, raw_source_text) / 100.0

            if alignment_score < 0.50:
                logger.warning(
                    f"Low alignment ({alignment_score:.2f}) for item {index}. "
                    f"Possible hallucination or mis-mapping."
                )

            # 5. Fetch spatial coordinates (BBoxes) from OCR source
            # Accessing: OCRDocument -> Page -> TextLine -> BBox
            bboxes = [
                ocr_doc.ocr[line.page].text_lines[line.line].bbox
                for line in resolved_lines
            ]

            # 6. Create the enriched "Rich" version of the item
            rich_items.append(
                RichItem.from_item(
                    source_item=item,
                    raw_text=raw_source_text,
                    lines_resolved=resolved_lines,
                    alignment=alignment_score,
                    bboxes=bboxes,
                )
            )

        return RichStructured(items=rich_items)
