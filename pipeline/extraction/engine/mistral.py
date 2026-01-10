from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Iterable, Iterator, Dict
from pydantic import Field
from pydantic.dataclasses import dataclass

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.chunking.schemas import Chunk
from pipeline.extraction.postprocessing import fix_lines_alignment
from pipeline.extraction.schemas import Structured
from pipeline.extraction.utils import format_chunk_as_numbered_lines
from pipeline.logging import logger


def on_extraction_failure(retry_state) -> Structured:
    """Fallback handler for failed API calls."""
    error = retry_state.outcome.exception()
    logger.error(f"Mistral extraction failed after multiple retries: {error}")
    return Structured(items=[])


class MistralEngine:
    """
    Inference engine for Mistral AI with nested parameter management.
    """

    @dataclass
    class Parameters:
        """
        Runtime parameters for the Mistral engine.
        Matches the structure of the OCR engine parameters.
        """

        api_key: str | None = Field(default=None, description="Mistral API key.")

        model: str = Field(
            default="mistral-large-latest",
            description="Mistral model name to use for extraction.",
        )

        system_prompt: str = Field(
            default="",
            description="System prompt to guide the Mistral model's behavior.",
        )

        temperature: float = Field(
            default=0.1,
            ge=0.0,
            le=1.0,
            description="Sampling temperature for the model.",
        )
        # Using Field for options to allow extra mistral-specific kwargs
        options: dict = Field(
            default_factory=dict,
            description="Additional options passed directly to the Mistral API.",
        )
        request_delay: float = Field(
            default=3.0,
            ge=0.0,
            description="Minimum delay between requests to respect rate limits.",
        )
        max_concurrent: int = Field(
            default=20,
            ge=1,
            description="Maximum number of concurrent requests to the Mistral API.",
        )

    def __init__(self) -> None:
        self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=600),
        reraise=True,
        retry_error_callback=on_extraction_failure,
    )
    def process_single(
        self,
        chunk: Chunk,
        params: MistralEngine.Parameters,
        chunk_index: int | None = None,
    ) -> Structured:
        """
        Processes a single chunk using the provided parameters.
        Assumes that the Mistral client is already initialized.
        """

        if self._client is None:
            raise ValueError(
                "Mistral client is not initialized. Call _ensure_client first."
            )

        formatted_text = format_chunk_as_numbered_lines(chunk)

        logger.debug(
            f"Invoking Mistral {params.model} (Chunk {chunk_index if chunk_index is not None else 'N/A'})",
            temp=params.temperature,
            options=params.options,
        )
        messages = [
            {"role": "system", "content": params.system_prompt},
            {
                "role": "user",
                "content": formatted_text
                or "Line-by-line lyrics of Rick Astley 'Never Gonna Give You Up'",
            },
        ]
        response = self._client.chat.parse(
            model=params.model,
            messages=messages,
            response_format=Structured,
            temperature=params.temperature,
            **params.options,
        )

        if (
            not response.choices
            or not response.choices[0]
            or not response.choices[0].message
            or not response.choices[0].message.parsed
        ):
            raise ValueError("Mistral API returned an empty or invalid response")

        extracted_data = response.choices[0].message.parsed
        return fix_lines_alignment(chunk, extracted_data)

    def process_multiple(
        self,
        chunks: Iterable[Chunk],
        params: MistralEngine.Parameters,
    ) -> Iterator[Structured]:
        """
        Concurrent execution logic using the Parameters object for configuration.
        """

        self._ensure_client(params.api_key)

        pending_futures: Dict[int, Future] = {}
        next_expected_index = 0
        last_submission_time = 0.0

        with ThreadPoolExecutor(max_workers=params.max_concurrent) as executor:
            for i, chunk in enumerate(chunks):
                # Rate limiting logic
                elapsed = time.time() - last_submission_time
                wait_duration = params.request_delay - elapsed
                if wait_duration > 0:
                    time.sleep(wait_duration)

                # We pass the params object into process_single
                future = executor.submit(
                    self.process_single, chunk, params, chunk_index=i
                )
                pending_futures[i] = future
                last_submission_time = time.time()

                while next_expected_index in pending_futures:
                    current_future = pending_futures[next_expected_index]
                    if not current_future.done():
                        break

                    yield current_future.result()
                    pending_futures.pop(next_expected_index)
                    next_expected_index += 1

            # Final drain
            while next_expected_index in pending_futures:
                yield pending_futures.pop(next_expected_index).result()
                next_expected_index += 1

    def _ensure_client(self, api_key: str | None) -> None:
        """Initializes the Mistral client if not already done."""
        if self._client is None:
            if not api_key:
                raise ValueError("Mistral API key must be provided for initialization.")
            self._client = Mistral(api_key=api_key)
