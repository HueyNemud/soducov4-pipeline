"""Extraction stage: stream chunks (JSONL) -> stream extractions (JSONL)."""

from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, Field

from pipeline.chunking.schemas import Chunk
from pipeline.chunking.stage import Chunking
from pipeline.core.artifact import Artifact
from pipeline.core.context import RunContext
from pipeline.core.stage import PipelineStage, safe_get_dependency, stage_config

from pipeline.extraction.schemas import StructuredSeq
from .engine.mistral import MistralEngine
from .engine.ollama import OllamaEngine


class ExtractionParameters(BaseModel):
    """
    Orchestration parameters for the Extraction pipeline stage.
    """

    provider: Literal["mistral", "ollama"] = "ollama"

    # Engine-specific configuration objects
    mistral: MistralEngine.Parameters = Field(default_factory=MistralEngine.Parameters)
    ollama: OllamaEngine.Parameters = Field(default_factory=OllamaEngine.Parameters)

    system_prompt: str = Field(
        default="Extract entities into a structured JSON format.",
        description="The instructions provided to the LLM.",
    )
    mistral_api_key: str | None = None


@stage_config(
    produces=StructuredSeq,
    depends_on=[Chunking],
    params_model=ExtractionParameters(),
)
class Extraction(PipelineStage[StructuredSeq, ExtractionParameters]):
    """
    Orchestrates structured data extraction from text chunks.

    Delegates inference to Mistral or Ollama backends using a
    standardized Parameters pattern.
    """

    def run(
        self,
        ctx: RunContext,
        parameters: ExtractionParameters,
        dependencies: dict[str, Artifact],
    ) -> StructuredSeq:
        """Executes the extraction pipeline over dependencies."""

        # Retrieve chunks from the previous stage
        chunks = safe_get_dependency(dependencies, Chunking)

        from .engine import create_extraction_engine

        # 1. Initialize the engine (Connection / Auth / System Prompt)
        engine = create_extraction_engine(
            provider=parameters.provider,
            system_prompt=parameters.system_prompt,
            api_key=parameters.mistral_api_key,
        )

        # 2. Select the specific parameters for the chosen provider
        # This mirrors the 'p.layout' / 'p.spuriousness' pattern from OCR
        engine_params = (
            parameters.mistral
            if parameters.provider == "mistral"
            else parameters.ollama
        )

        self.logger.info(
            f"Starting {self.name.upper()} stage via {parameters.provider.upper()} "
            f"(Model: {engine_params.model})"
        )

        # 3. Process stream and emit results
        extractions = []
        for item in engine.process_multiple(chunks, params=engine_params):
            ctx.emit(item)
            extractions.append(item)

        return StructuredSeq(root=extractions)


def format_chunk_as_numbered_lines(chunk: Chunk) -> str:
    """
    Prepares a text chunk for LLM processing by prepending line numbers.
    Format example: '0 @ Content of first line'
    """
    return "\n".join(f"{index} @ {line.text}" for index, line in enumerate(chunk))
