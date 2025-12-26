from typing import Iterator
from pipeline.assembly.schemas import RichItem, RichStructured
from pipeline.chunking.schemas import Chunk, Chunks
from pipeline.extraction.postprocessing import _simple_string_repr
from pipeline.extraction.schemas import (
    Structured,
    StructuredCompact,
    StructuredSeq,
)
from pipeline.ocr.schemas import OCRDocument
from thefuzz import fuzz
from pipeline.logging import logger


class Assembler:

    def assemble_multiple(
        self,
        ocr: OCRDocument,
        chunks: Chunks,
        structuredseq: StructuredSeq,
        strict: bool,
    ) -> Iterator[RichStructured]:
        if len(chunks) < len(structuredseq):
            raise ValueError(
                "Number of chunks and structured items must be the same for assembly."
            )
        elif len(chunks) > len(structuredseq):
            logger.error(
                "More chunks than structured items; some chunks will be ignored during assembly."
            )

        for chunk, struct in zip(chunks, structuredseq):
            s: Structured = (
                struct.expand() if isinstance(struct, StructuredCompact) else struct
            )

            yield self.assemble_single(
                ocr=ocr,
                chunk=chunk,
                structured=s,
                strict=strict,
            )

    def assemble_single(
        self,
        ocr: OCRDocument,
        chunk: Chunk,
        structured: Structured,
        strict: bool,
    ) -> RichStructured:
        # Crée un counter qui compte le nombre de fois que chaque ligne du chunk est utilisée dans les items extraits
        # coverage = {
        #     i: Counter(
        #         chain.from_iterable([item.lines for item in structured.items])
        #     ).get(i, 0)
        #     for i, _ in enumerate(chunk)
        # }

        # # Tests :
        # # - Est-ce que toutes les lignes du chunk sont bien couvertes ?
        # uncovered_lines = [i for i, count in coverage.items() if count == 0]
        # if uncovered_lines:
        #     logger.warning(f"Chunk has uncovered lines: {uncovered_lines}")
        rich_items = []
        for item in structured.items:

            # Safe line resolution
            lines = []
            for i in item.lines:
                try:
                    lines.append(chunk[i])
                except IndexError:
                    if strict:
                        raise IndexError(
                            f"Line index {i} out of range for chunk with {len(chunk)} lines."
                        )
                    else:
                        logger.error(
                            f"Line index {i} out of range for chunk with {len(chunk)} lines. Skipping line."
                        )

            # Ignore full "margin" items.
            if all(line.is_margin for line in lines):
                continue

            raw_text = "\n".join([line.text for line in lines]).strip()

            item_repr = _simple_string_repr(
                item, exclude_fields=["lines", "cat"]
            ).strip()
            alignment = fuzz.ratio(item_repr, raw_text) / 100.0

            if alignment < 0.50:
                logger.warning(
                    f"Low alignment ({alignment:.2f}) between extracted item and OCR text:\n"
                    f"Extracted: >{item_repr}<\n"
                    f"OCR Text: >{raw_text}<"
                )

            # Recupère les Bbboxes des lignes utilisées
            bboxes = [ocr.ocr[line.page].text_lines[line.line].bbox for line in lines]

            rich_items.append(
                RichItem.from_item(
                    item=item,
                    raw_text=raw_text,
                    lines_resolved=lines,
                    alignment=alignment,
                    bboxes=bboxes,
                )
            )

        return RichStructured(items=rich_items)
