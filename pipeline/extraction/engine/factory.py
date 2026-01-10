from typing import Iterable, Iterator, Type, Dict, Any, Protocol, runtime_checkable

from pipeline.chunking.schemas import Chunk
from pipeline.extraction.schemas import Structured
from .mistral import MistralEngine
from .ollama import OllamaEngine


@runtime_checkable
class ExtractionEngine(Protocol):
    """
    Protocol defining the interface for extraction backends.
    Now supports the nested Parameters pattern.
    """

    class Parameters:
        """Structural placeholder for engine-specific parameters."""

        ...

    def process_single(self, chunk: Chunk, params: Any) -> Structured:
        """Process a single text chunk using specific runtime parameters."""
        ...

    def process_multiple(
        self, chunks: Iterable[Chunk], params: Any | None = None
    ) -> Iterator[Structured]:
        """Stream multiple chunks using specific runtime parameters."""
        ...


# Registry mapping engine identifiers to their implementation classes
ENGINE_REGISTRY: Dict[str, Type[Any]] = {
    "mistral": MistralEngine,
    "ollama": OllamaEngine,
}


def create_extraction_engine(provider: str) -> ExtractionEngine:
    """
    Factory function to initialize an extraction engine backend.

    Configuration (model, temperature, etc.) is no longer passed here,
    but injected later via the engine's Parameters dataclass.
    """
    provider_key = provider.lower()
    engine_class = ENGINE_REGISTRY.get(provider_key)

    if not engine_class:
        raise ValueError(
            f"Unsupported engine provider: '{provider}'. "
            f"Available providers: {list(ENGINE_REGISTRY.keys())}"
        )

    return engine_class()
