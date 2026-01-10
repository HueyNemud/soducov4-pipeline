"""
Pipeline Stage for Document Chunking.

This module integrates the ChunkCreator engine into the pipeline. It handles
dependency resolution from the OCR stage, parameter validation, and
streaming/collecting text chunks for downstream processing.
"""

from __future__ import annotations
from pydantic import BaseModel, Field

from pipeline.chunking.engine import ChunkCreator
from pipeline.chunking.schemas import Chunks
from pipeline.core.artifact import Artifact
from pipeline.core.context import RunContext
from pipeline.core.stage import (
    PipelineStage,
    safe_get_dependency,
    stage_config,
)
from pipeline.ocr.stage import OCR


class ChunkingParameters(BaseModel):
    """
    Orchestration parameters for the Chunking stage.

    Wraps the core engine parameters to allow for future stage-specific
    configurations without polluting the engine's logic.
    """

    engine: ChunkCreator.Parameters = Field(
        default_factory=ChunkCreator.Parameters,
        description="Core parameters for the underlying ChunkCreator engine.",
    )


@stage_config(produces=Chunks, depends_on=[OCR], params_model=ChunkingParameters())
class Chunking(PipelineStage[Chunks, ChunkingParameters]):
    """
    Pipeline stage that partitions OCR documents into semantic text chunks.

    It consumes OCR artifacts, applies token-based segmentation via the
    ChunkCreator engine, and emits individual chunks to the run context.
    """

    params_model = ChunkingParameters

    def run(
        self,
        ctx: RunContext,
        parameters: ChunkingParameters,
        dependencies: dict[str, Artifact],
    ) -> Chunks:
        """
        Executes the chunking stage.

        Args:
            ctx: The runtime context for emitting artifacts.
            parameters: Validated ChunkingParameters.
            dependencies: Dictionary containing the required OCR artifact.

        Returns:
            Chunks: A collection of all generated text chunks.
        """
        # Resolve dependency from the previous OCR stage
        ocr_document = safe_get_dependency(dependencies, OCR)

        # Import engine locally to keep stage footprint light
        from .engine import ChunkCreator

        # Initialize the service with parameters and process the document
        engine_params = ChunkCreator.Parameters(**parameters.model_dump())
        creator = ChunkCreator(params=engine_params)

        chunk_list = []
        for chunk in creator.generate(ocr_document):
            # Emit individually for real-time monitoring/streaming
            ctx.emit(chunk)
            chunk_list.append(chunk)

        return Chunks(root=chunk_list)
