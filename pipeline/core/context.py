"""
Pipeline Runtime Context.

Manages execution state, configuration parameters, and the lifecycle of
output artifacts. This context acts as the central coordination point
between different pipeline stages.
"""

from pathlib import Path
from typing import Optional

from pipeline.core.artifact import Artifact, JSONLWriter
from pipeline.core.store import Store


class RunContext:
    """
    Orchestrates state and storage for a specific pipeline execution.

    Provides access to global configuration (paths, debug flags) and manages
    the streaming of artifacts through an active JSONL writer.
    """

    def __init__(
        self,
        pdf_path: Path,
        artifacts_dir: Path,
        debug: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the run context and prepares the filesystem.
        """
        # Global shared data store
        self.store = Store(
            {
                "pdf_path": pdf_path,
                "artifacts_dir": artifacts_dir,
                "debug": debug,
                "verbose": verbose,
            }
        )

        # Namespaced storage for stage-specific data and artifacts
        self.store.stages = Store()
        self.store.artifacts = Store()

        # Active stream for incremental artifact writing
        self._artifact_stream: Optional[JSONLWriter] = None

        self._ensure_artifacts_dir()

    @property
    def pdf_path(self) -> Path:
        """Original source document path."""
        return self.store.pdf_path

    @property
    def artifacts_dir(self) -> Path:
        """Root directory for all generated outputs."""
        return self.store.artifacts_dir

    @property
    def debug_enabled(self) -> bool:
        """Flag indicating if debug mode is active."""
        return self.store.debug

    @property
    def verbose_enabled(self) -> bool:
        """Flag indicating if verbose logging is active."""
        return self.store.verbose

    def _ensure_artifacts_dir(self, stage_name: str = "") -> None:
        """Creates the artifact root or stage-specific subdirectories on disk."""
        target_path = self.artifacts_dir
        if stage_name:
            target_path /= stage_name

        target_path.mkdir(parents=True, exist_ok=True)

    def attach_stream(self, writer: JSONLWriter) -> None:
        """Connects a JSONL writer to enable real-time artifact streaming."""
        self._artifact_stream = writer

    def detach_stream(self) -> None:
        """Disconnects the active artifact stream."""
        self._artifact_stream = None

    def emit(self, artifact: Artifact) -> None:
        """
        Writes an artifact to the active stream if one is connected.

        Used by stages to output records incrementally during processing.
        """
        if self._artifact_stream:
            self._artifact_stream.write(artifact)
