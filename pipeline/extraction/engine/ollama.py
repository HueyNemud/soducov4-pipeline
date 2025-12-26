from __future__ import annotations

from typing import Any,  Iterator
from ollama import Client
from pyparsing import Iterable

from pipeline.chunking.schemas import Chunk
from pipeline.extraction.postprocessing import fix_lines_alignment
from pipeline.extraction.schemas import Structured
from pipeline.extraction.stage import chunk_to_numbered_text


class OllamaEngine:
    """LLM extraction engine using Ollama."""

    def __init__(
        self,
        *,
        model: str,
        options: dict[str, Any] | None = None,
        system_prompt: str = "",
    ) -> None:
        self.model = model
        self.options = options or {}
        self.system_prompt = system_prompt
        self.client = Client()

    def process_single(self, chunk: Chunk) -> Structured:
        text = chunk_to_numbered_text(chunk)
        response_json = self._call_ollama(text)
        print(f"Ollama response JSON: {response_json}")  # Debug print
        response = Structured.model_validate_json(response_json)
        return fix_lines_alignment(chunk, response)

    def process_multiple(
        self, chunks: Iterable[Chunk]
    ) -> Iterator[Structured]:
        for chunk in chunks:
            yield self.process_single(chunk)

    def _call_ollama(self, text: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            format=Structured.model_json_schema(),
            options=self.options,
        )
        return response["message"]["content"]
