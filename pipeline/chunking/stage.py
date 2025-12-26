from __future__ import annotations
from pydantic import BaseModel, Field
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
    tokenizer_model: str = "almanach/camembert-base"
    max_tokens: int = 2000
    margin_lines: int = 2
    spuriousness_threshold: float = Field(default=1.0, ge=0.0, le=1.0)


@stage_config(produces=Chunks, depends_on=[OCR], params_model=ChunkingParameters())
class Chunking(PipelineStage[Chunks, ChunkingParameters]):

    params_model = ChunkingParameters

    def run(
        self,
        ctx: RunContext,
        parameters: ChunkingParameters,
        dependencies: dict[str, Artifact],
    ) -> Chunks:

        # Récupère la dépendance OCR
        ocr_artifact = safe_get_dependency(dependencies, OCR)

        from .engine import ChunkCreator

        chunk_creator = ChunkCreator(
            document=ocr_artifact, **parameters.model_dump(exclude_none=True)
        )

        chunks = []
        for c in chunk_creator.generate():
            ctx.emit(c)
            chunks.append(c)
        return Chunks(chunks)
