"""Extraction stage: stream chunks (JSONL) -> stream extractions (JSONL)."""

from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, Field

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

        # 1. Initialize the LLM extraction engine
        engine = create_extraction_engine(provider=parameters.provider)

        # 2. Select the specific parameters for the chosen provider
        engine_params = parameters.__getattribute__(parameters.provider)

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
