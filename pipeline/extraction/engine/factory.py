from typing import Iterable, Iterator, Type, Dict, Any, Protocol

from pipeline.chunking.schemas import Chunk
from pipeline.extraction.schemas import Structured
from .mistral import MistralEngine
from .ollama import OllamaEngine


# On définit ce qu'est un Engine via un Protocol
class Engine(Protocol):
    def process_single(self, chunk: Chunk) -> Structured: ...
    def process_multiple(self, chunks: Iterable[Chunk]) -> Iterator[Structured]: ...


# Mapping dynamique des moteurs
ENGINE_MAP: Dict[str, Type[Engine]] = {
    "mistral": MistralEngine,
    "ollama": OllamaEngine,
}


def create_extraction_engine(
    engine: str, config: Dict[str, Any], system_prompt: str = ""
) -> Engine:
    """
    Factory function to create an extraction engine based on the specified name.

    Args:
        engine (str): Name of the engine, e.g., "mistral" or "ollama".
        config (dict): Configuration dictionary for all engines.
        system_prompt (str): Optional system prompt to initialize the engine.

    Returns:
        Engine: An instance of the requested engine implementing the Engine protocol.

    Raises:
        ValueError: If the engine name is unsupported.
    """
    engine_key = engine.lower()
    engine_cls = ENGINE_MAP.get(engine_key)
    if engine_cls is None:
        raise ValueError(f"Unsupported engine: {engine}")

    engine_cfg = config.get(engine_key, {})

    kwargs = {
        "model": engine_cfg.get("model", ""),
        "system_prompt": system_prompt,
    }

    # Ajouter api_key si le moteur le supporte
    if "api_key" in engine_cfg:
        kwargs["api_key"] = engine_cfg["api_key"]

    # Ajouter options si elles existent
    if "options" in engine_cfg:
        kwargs["options"] = engine_cfg["options"]

    return engine_cls(**kwargs)
