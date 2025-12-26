"""A run context for a pipeline execution."""

from pathlib import Path
from pipeline.core.artifact import Artifact, JSONLWriter
from pipeline.core.store import Store


class RunContext:
    """Context for a pipeline run, providing access to configuration and paths."""

    store: Store

    # TODO Introduire un DebugSink (ou DebugWriter) simple pour gérer les sorties de debug (fichiers, logs, etc.)

    def __init__(
        self,
        pdf_path: Path,
        artifacts_dir: Path,
        debug: bool = False,
        verbose: bool = False,
    ) -> None:
        self.store = Store(
            {
                "pdf_path": pdf_path,
                "artifacts_dir": artifacts_dir,
                "debug": debug,
                "verbose": verbose,
            }
        )
        self.store.stages = Store()
        self.store.artifacts = Store()
        self._ensure_artifacts_dir()

    @property
    def pdf_path(self) -> Path:
        return self.store.pdf_path

    @property
    def artifacts_dir(self) -> Path:
        return self.store.artifacts_dir

    @property
    def debug(self) -> bool:
        return self.store.debug

    @property
    def verbose(self) -> bool:
        return self.store.verbose

    def _ensure_artifacts_dir(self, stage: str = "") -> None:
        """Ensures that the artifacts directory (and stage subdirectory, if specified) exists."""
        dir_path = self.artifacts_dir / stage if stage else self.artifacts_dir
        dir_path.mkdir(parents=True, exist_ok=True)

    _artifact_stream: JSONLWriter | None = None

    def set_stream(self, writer: JSONLWriter) -> None:
        """Active un flux d'émission pour le streaming."""
        self._artifact_stream = writer

    def clear_stream(self) -> None:
        """Désactive le flux d'émission."""
        self._artifact_stream = None

    def emit(self, event: Artifact) -> None:
        if self._artifact_stream is not None:
            self._artifact_stream.write(event)
