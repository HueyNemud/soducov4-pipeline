from __future__ import annotations

from typing import Union
from pydantic import BaseModel

from pipeline.assembly.schemas import (
    RichEntry,
    RichStructured,
    RichText,
    RichTitle,
)
from pipeline.chunking.stage import Chunking
from pipeline.core.artifact import Artifact
from pipeline.core.context import RunContext
from pipeline.core.stage import PipelineStage, safe_get_dependency, stage_config
from pipeline.extraction.stage import Extraction
from pipeline.ocr.stage import OCR


class AssemblyParameters(BaseModel):
    """Configuration parameters for the document assembly process."""

    strict: bool = False


@stage_config(
    produces=RichStructured,
    depends_on=[OCR, Chunking, Extraction],
    params_model=AssemblyParameters(),
)
class Assembly(PipelineStage[RichStructured, AssemblyParameters]):
    """
    Final pipeline stage that fuses data from OCR, Chunking, and Extraction.

    It maps logical extractions back to physical document coordinates to produce
    a 'RichStructured' document containing both text and spatial metadata.
    """

    def run(
        self,
        ctx: RunContext,
        parameters: AssemblyParameters,
        dependencies: dict[str, Artifact],
    ) -> RichStructured:
        """
        Executes the assembly logic by merging cross-stage dependencies.
        """
        self.logger.info(
            f"Initializing {self.name.upper()} with strict_mode={parameters.strict}"
        )

        # 1. Resolve upstream dependencies
        ocr_data = safe_get_dependency(dependencies, OCR)
        chunk_data = safe_get_dependency(dependencies, Chunking)
        extraction_results = safe_get_dependency(dependencies, Extraction)

        from .engine import Assembler

        assembler = Assembler()

        # 2. Reconstruct rich items (Text + BBoxes + Alignment)
        enriched_items: list[Union[RichEntry, RichTitle, RichText]] = []

        # Iterate through the generator to reconstruct segments
        for rich_bundle in assembler.assemble_multiple(
            ocr_doc=ocr_data,
            chunks=chunk_data,
            structured_sequence=extraction_results,
            strict_mode=parameters.strict,
        ):
            # Emit individual bundles for real-time monitoring/downstream consumption
            ctx.emit(rich_bundle)
            enriched_items.extend(rich_bundle.items)

        # 3. Aggregate global document metadata (e.g., Image positions)
        # Flattening image bounding boxes from all pages
        document_image_bboxes = [
            [int(coord) for coord in page.image_bbox] for page in ocr_data.ocr
        ]

        self.logger.info(
            f"Assembly complete: {len(enriched_items)} rich items reconstructed."
        )

        return RichStructured(
            items=enriched_items,
            images_bboxes=document_image_bboxes,
        )
