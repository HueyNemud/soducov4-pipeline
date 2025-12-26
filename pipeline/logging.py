import sys
from typing import Any
from loguru import logger
from pathlib import Path

INITIALIZED = False


if not INITIALIZED:

    FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<bold>{module}</bold> | "
        "<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level> | "
        "<yellow>{extra}</yellow>"
    )

    sinks: set[Any] = set()

    # Replace the default logger
    logger.remove(0)
    # logger.configure(extra={"stage": ""})  # Default values
    logger.add(sys.stderr, format=FORMAT, level="DEBUG")

    def log_to_file(file_path: Path) -> None:
        """Ajoute un sink de logging vers un fichier spécifié."""

        if file_path in sinks:
            logger.warning(f"Logging to {file_path} is already set up.")
            return

        logger.add(file_path, format=FORMAT, rotation="100 MB")
        sinks.add(file_path)

    INITIALIZED = True
