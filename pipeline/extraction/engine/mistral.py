from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
import time
from typing import Iterable, Iterator

from mistralai import Mistral
from pipeline.chunking.schemas import Chunk
from pipeline.extraction.postprocessing import fix_lines_alignment
from pipeline.extraction.schemas import Structured
from pipeline.extraction.stage import chunk_to_numbered_text
from tenacity import retry, stop_after_attempt, wait_exponential
from pipeline.logging import logger


def handle_failure(retry_state):
    # .exception() récupère l'erreur sans la lever
    exception = retry_state.outcome.exception()
    logger.error(f"Failed to process chunk after retries: {exception}")
    # On relance l'exception d'origine
    return Structured(items=[])


class MistralEngine:
    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str = "",
        *,
        options: dict | None = None,
    ):
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.options = options or {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=600),
        reraise=True,
        retry_error_callback=handle_failure,
    )
    def process_single(
        self, chunk: Chunk, chunk_index: int | None = None
    ) -> Structured:
        text = chunk_to_numbered_text(chunk)
        logger.debug(
            "Sending request to Mistral",
            model=self.model,
            options=self.options,
            text_snippet=text[:100],
            chunk_index=chunk_index,
        )

        resp = self.client.chat.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            response_format=Structured,
            **self.options,
        )

        if (
            not resp.choices
            or not resp.choices[0].message
            or resp.choices[0].message.parsed is None
        ):
            raise ValueError("No valid response from Mistral API")

        result = resp.choices[0].message.parsed
        return fix_lines_alignment(chunk, result) if result else result

    def process_multiple(
        self,
        chunks: Iterable[Chunk],
        max_concurrent: int = 10,
        delay_seconds: float = 3.0,
    ) -> Iterator[Structured]:
        """
        Yield results dès qu'ils sont prêts, mais toujours dans l'ordre des chunks.
        """
        futures: dict[int, Future] = {}
        next_index = 0
        last_submit = 0.0

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            for i, chunk in enumerate(chunks):
                # Rate limit strict
                sleep_time = delay_seconds - (time.time() - last_submit)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                futures[i] = executor.submit(self.process_single, chunk, chunk_index=i)
                last_submit = time.time()

                # Yield dès que possible dans l'ordre
                while next_index in futures and futures[next_index].done():
                    yield futures.pop(next_index).result()
                    next_index += 1

            # Yield le reste
            while next_index in futures:
                yield futures.pop(next_index).result()
                next_index += 1
