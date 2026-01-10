from __future__ import annotations

from typing import Any, Iterator, Iterable
from pydantic import Field
from pydantic.dataclasses import dataclass
from ollama import Client

from pipeline.chunking.schemas import Chunk
from pipeline.extraction.postprocessing import fix_lines_alignment
from pipeline.extraction.schemas import Structured
from pipeline.extraction.stage import format_chunk_as_numbered_lines
from pipeline.logging import logger


class OllamaEngine:
    """
    Inference engine for Ollama with nested parameter management.
    Ensures local LLM extraction follows a strictly enforced JSON schema.
    """

    @dataclass
    class Parameters:
        """
        Runtime parameters for the Ollama engine.
        Allows fine-tuning of the local model behavior.
        """

        model: str = "ministral-3:14b-instruct-2512-q8_0"
        # Options dict for Ollama-specific runner settings (num_ctx, temperature, etc.)
        options: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, system_prompt: str = "") -> None:
        """
        Initializes the Ollama client.
        Model-specific settings are injected via Parameters during processing.
        """
        self.system_prompt = system_prompt
        self.client = Client()

    def process_single(
        self, chunk: Chunk, params: OllamaEngine.Parameters
    ) -> Structured:
        """
        Extracts structured data from a single text chunk using provided parameters.
        """
        formatted_input = format_chunk_as_numbered_lines(chunk)

        # Execute completion with schema enforcement
        raw_json_response = self._generate_completion(formatted_input, params)

        logger.debug(f"Ollama raw response: {raw_json_response}")

        # Validate against Pydantic schema
        structured_data = Structured.model_validate_json(raw_json_response)

        # Correct any line-index drift
        return fix_lines_alignment(chunk, structured_data)

    def process_multiple(
        self, chunks: Iterable[Chunk], params: OllamaEngine.Parameters | None = None
    ) -> Iterator[Structured]:
        """
        Sequentially processes a stream of chunks.
        """
        p = params or self.Parameters()
        for chunk in chunks:
            yield self.process_single(chunk, p)

    def _generate_completion(self, text: str, params: OllamaEngine.Parameters) -> str:
        """
        Internal helper to manage the Ollama chat request with JSON format enforcement.
        """
        response = self.client.chat(
            model=params.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            # Use Structured model's schema to guide Ollama's output
            format=Structured.model_json_schema(),
            options=params.options,
        )

        return response["message"]["content"]
