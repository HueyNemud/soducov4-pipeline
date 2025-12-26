"""Extraction stage: stream chunks (JSONL) -> stream extractions (JSONL)."""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from pipeline.chunking.schemas import Chunk
from pipeline.chunking.stage import Chunking
from pipeline.core.artifact import Artifact
from pipeline.core.context import RunContext
from pipeline.core.stage import PipelineStage, safe_get_dependency, stage_config
from pipeline.extraction.schemas import StructuredSeq


class ExtractionParameters(BaseModel):
    engine: Literal["mistral", "ollama"] = "ollama"
    model: str = "ministral-3:14b-instruct-2512-q8_0"
    temperature: float = 0.1
    repetition_penalty: float = 1.4
    num_predict: int = 16_000
    num_ctx: int = 16_000
    repeat_last_n: int = 128
    system_prompt: str = (
        "Rick Roll's 'Never gonna give you up' lyrics as a JSON object."
    )
    mistral_api_key: str | None = None


@stage_config(
    produces=StructuredSeq,
    depends_on=[Chunking],
    params_model=ExtractionParameters(),
)
class Extraction(PipelineStage[StructuredSeq, ExtractionParameters]):
    """Stage that performs structured extraction from chunks using an extraction engine."""

    def run(
        self,
        ctx: RunContext,
        parameters: ExtractionParameters,
        dependencies: dict[str, Artifact],
    ) -> StructuredSeq:
        self.logger.info(
            f"Starting {self.name.upper()} stage with parameters: {parameters.model_dump(exclude={'system_prompt'})}"
        )

        chunks = safe_get_dependency(dependencies, Chunking)

        from .engine import create_extraction_engine

        engine = create_extraction_engine(
            engine=parameters.engine,
            system_prompt=parameters.system_prompt,
            config={
                "ollama": {
                    "model": parameters.model,
                    "options": {
                        "temperature": parameters.temperature,
                        "repetition_penalty": parameters.repetition_penalty,
                        "num_predict": parameters.num_predict,
                        "num_ctx": parameters.num_ctx,
                        "repeat_last_n": parameters.repeat_last_n,
                    },
                },
                "mistral": {
                    "model": parameters.model,
                    "api_key": parameters.mistral_api_key,
                    "options": {
                        "temperature": parameters.temperature,
                        # "max_tokens": parameters.num_predict,
                    },
                },
            },
        )

        result = []
        for extraction in engine.process_multiple(chunks):
            ctx.emit(extraction)
            result.append(extraction)
        return StructuredSeq(root=result)


def chunk_to_numbered_text(chunk: Chunk) -> str:
    """Convert chunk to numbered text format for LLM input.

    Args:
        chunk: List of ChunkLine objects

    Returns:
        Formatted text with line numbers
    """
    # Use 0-based indices for compatibility with existing code
    lines: list[str] = []
    for i, line in enumerate(chunk):
        lines.append(f"{i} @ {line.text}")

    return "\n".join(lines)
