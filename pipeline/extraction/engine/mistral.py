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
from pipeline.extraction.stage import format_chunk_as_numbered_lines
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

        model: str = "ministral-3:14b-instruct-2512-q8_0"
        temperature: float = 0.1
        # Using Field for options to allow extra mistral-specific kwargs
        options: dict = Field(default_factory=dict)
        request_delay: float = 3.0
        max_concurrent: int = 20

    def __init__(self, api_key: str, system_prompt: str = ""):
        """
        Initializes the Mistral client.
        Note: Model-specific settings are now handled via Parameters at runtime.
        """
        self.client = Mistral(api_key=api_key)
        self.system_prompt = system_prompt

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
        """
        formatted_text = format_chunk_as_numbered_lines(chunk)

        logger.debug(
            f"Invoking Mistral {params.model} (Chunk {chunk_index if chunk_index is not None else 'N/A'})",
            temp=params.temperature,
        )

        response = self.client.chat.parse(
            model=params.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_text},
            ],
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
        params: MistralEngine.Parameters | None = None,
    ) -> Iterator[Structured]:
        """
        Concurrent execution logic using the Parameters object for configuration.
        """
        p = params or self.Parameters()
        pending_futures: Dict[int, Future] = {}
        next_expected_index = 0
        last_submission_time = 0.0

        with ThreadPoolExecutor(max_workers=p.max_concurrent) as executor:
            for i, chunk in enumerate(chunks):
                # Rate limiting logic
                elapsed = time.time() - last_submission_time
                wait_duration = p.request_delay - elapsed
                if wait_duration > 0:
                    time.sleep(wait_duration)

                # We pass the params object into process_single
                future = executor.submit(self.process_single, chunk, p, chunk_index=i)
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
