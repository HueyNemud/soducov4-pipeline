"""
Pipeline Artifact Management.

This module handles the lifecycle of Pipeline artifacts, providing utilities for
serialization, file-based caching with metadata validation, and incremental
streaming (JSONL).
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import (
    IO,
    Any,
    Iterator,
    Literal,
    Optional,
    Type,
    TypeVar,
    cast,
)
import shelve
from dbm import error as dbm_error
from pydantic import BaseModel
from pipeline.logging import logger

# Type alias for clarity; Artifacts are essentially Pydantic models
Artifact = BaseModel

_T = TypeVar("_T", bound=Artifact)


def save_artifact(artifact: Artifact, path: Path) -> None:
    """Serializes an artifact to a JSON file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(), encoding="utf-8")


def load_artifact(path: Path, cls: Type[_T]) -> _T:
    """Loads and validates an artifact from a JSON file."""
    return cls.model_validate_json(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Artifact Caching
# -----------------------------------------------------------------------------


class FileCache:
    """
    Persistent file-based cache for JSON-serializable artifacts.

    Uses a shelf database to store artifacts associated with a specific PDF
    processing run, supporting metadata-based invalidation (fingerprinting).
    """

    def __init__(self, source_path: str | Path):
        """Initializes cache path based on the input document path."""
        self.cache_path = Path(source_path).with_suffix(".cache")

    @contextmanager
    def _open_database(self, flag: str = "c") -> Iterator[shelve.Shelf]:
        """Manages the lifecycle of the underlying DBM database."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Cast for type-checker compatibility with shelve.open flags
            shelf_flag = cast(Literal["r", "w", "c", "n"], flag)
            with shelve.open(self.cache_path.as_posix(), flag=shelf_flag) as db:
                yield db
        except dbm_error as e:
            raise RuntimeError(
                f"Failed to access cache at {self.cache_path}: {e}"
            ) from e

    def save(
        self,
        stage: str,
        artifact: Artifact,
        *,
        override: bool = False,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        """Stores an artifact in the cache under a stage-specific key."""
        key = self._generate_key(stage, type(artifact))
        entry = {"artifact_json": artifact.model_dump_json(), "meta": meta or {}}

        with self._open_database("c") as db:
            if not override and key in db:
                raise KeyError(
                    f"Cache collision: {key} already exists. Use override=True."
                )
            db[key] = entry

    def load(
        self,
        stage: str,
        artifact_type: type[_T],
        *,
        expected_meta: Optional[dict[str, Any]] = None,
    ) -> _T:
        """
        Retrieves an artifact from cache.
        Validates metadata (e.g., fingerprints) if expected_meta is provided.
        """
        key = self._generate_key(stage, artifact_type)

        with self._open_database("r") as db:
            data = db[key]

        # Handle backward compatibility and extract components
        if isinstance(data, str):
            json_str, meta = data, {}
        else:
            json_str = data.get("artifact_json")
            meta = data.get("meta", {})

        if expected_meta is not None and meta != expected_meta:
            raise KeyError(f"Cache invalid: Metadata mismatch for {key}")

        return artifact_type.model_validate_json(json_str)

    def invalidate(self, stage_name: Optional[str] = None) -> None:
        """Clears the cache for a specific stage or the entire document."""
        action = f"for stage '{stage_name}'" if stage_name else "entirely"
        logger.debug(f"Invalidating cache {action} at {self.cache_path}")

        if not self.cache_path.exists():
            return

        with self._open_database("w") as db:
            if stage_name is None:
                db.clear()
            else:
                prefix = f"{stage_name}::"
                keys_to_delete = [k for k in db.keys() if str(k).startswith(prefix)]
                for k in keys_to_delete:
                    del db[k]

    @staticmethod
    def _generate_key(stage: str, artifact_cls: type) -> str:
        """Creates a unique string key for a stage/artifact pair."""
        return f"{stage}::{artifact_cls.__name__}"


def fingerprint(data: Any) -> str:
    """
    Generates a stable 16-character SHA256 hash of the input data.
    Used to invalidate cache when parameters or models change.
    """
    serialized = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        default=str,  # Fallback for non-serializable objects
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Streaming Utilities
# -----------------------------------------------------------------------------


class JSONLWriter:
    """
    Incremental writer for JSON Lines format.

    Ensures artifacts are written one per line to disk, allowing for
    memory-efficient streaming of large datasets.
    """

    def __init__(self, export_path: Path, *, auto_flush: bool = True):
        self.path = export_path
        self._auto_flush = auto_flush
        self._stream: Optional[IO[str]] = None
        self._record_count = 0

    def _init_stream(self) -> None:
        """Opens the file handle lazily upon the first write operation."""
        if self._stream is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8")

    def write(self, artifact: Artifact) -> None:
        """Writes a single artifact to the stream as a JSON line."""
        self._init_stream()
        # MyPy check: _init_stream ensures self._stream is not None
        cast(IO[str], self._stream).write(artifact.model_dump_json() + "\n")

        if self._auto_flush:
            self._stream.flush()  # type: ignore
        self._record_count += 1

    def close(self) -> None:
        """Closes the stream and cleans up empty files."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        elif self.path.exists() and self._record_count == 0:
            # Clean up if file was touched but no data written
            self.path.unlink()

    @property
    def has_content(self) -> bool:
        """Returns True if at least one record has been written."""
        return self._record_count > 0


def open_jsonl(path: Path) -> JSONLWriter:
    """Factory function to create a new JSONLWriter."""
    return JSONLWriter(path)
