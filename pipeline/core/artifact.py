"""Artifacts are Pydantic models that represent the output of a pipeline stage."""

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

Artifact = BaseModel

_T = TypeVar("_T", bound=Artifact)


def save_artifact(artifact: Artifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(), encoding="utf-8")


def load_artifact(path: Path, cls: Type[_T]) -> _T:
    return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ----------------------------------------
# Artifact caching
# ----------------------------------------


class FileCache:
    """File-based cache for JSON-serializable artifacts associated with a PDF."""

    def __init__(self, pdf_path: str | Path):
        self.cache_path = Path(pdf_path).with_suffix(".cache")

    @contextmanager
    def _open_cache(self, flag: str = "c") -> Iterator[shelve.Shelf]:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Because of how shelve declares its open function, we have to explicitly cast the flag to _TFlags
            # otherwise mypy will complain
            f = cast(Literal["r", "w", "c", "n", "rf", "wf"], flag)
            with shelve.open(self.cache_path.as_posix(), flag=f) as db:
                yield db
        except dbm_error as e:
            raise RuntimeError(f"Cannot open cache at {self.cache_path}: {e}") from e

    def save(
        self,
        stage: str,
        artifact: Artifact,
        *,
        override: bool = False,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        cache_key = self._cache_key(stage, type(artifact))
        payload = {"artifact_json": artifact.model_dump_json(), "meta": meta or {}}

        with self._open_cache("c") as db:
            if not override and cache_key in db:
                raise KeyError(f"Cache entry already exists: {cache_key}")
            db[cache_key] = payload

    def load(
        self,
        stage: str,
        artifact_type: type[_T],
        *,
        expected_meta: Optional[dict[str, Any]] = None,
    ) -> _T:
        cache_key = self._cache_key(stage, artifact_type)

        with self._open_cache("c") as db:
            raw = db[cache_key]

        # Backward-compatible: old entries stored as JSON string
        if isinstance(raw, str):
            artifact_json = raw
            meta = {}
        else:
            artifact_json = raw.get("artifact_json")
            meta = raw.get("meta", {})

        if expected_meta is not None and meta != expected_meta:
            raise KeyError(f"Cache metadata mismatch for {cache_key}")

        return artifact_type.model_validate_json(artifact_json)

    def invalidate(self, stage_name: Optional[str] = None) -> None:
        logger.debug(
            f"Invalidating cache {'for stage ' + stage_name if stage_name else 'entirely'}"
        )
        with self._open_cache("c") as db:
            if stage_name is None:
                db.clear()
                return

            prefix = f"{stage_name}::"
            for key in list(db.keys()):
                if str(key).startswith(prefix):
                    del db[key]

    @staticmethod
    def _cache_key(stage: str, artifact_cls: type[_T]) -> str:
        return f"{stage}::{artifact_cls.__name__}"


# TODO move to utils.py?
def fingerprint(data: Any) -> str:
    """
    Empreinte stable (best-effort) pour invalider le cache quand params/modèle changent.
    - JSON trié
    - fallback sur str() pour les objets non sérialisables
    """
    payload = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# -----------
# Streaming
# -----------


class JSONLWriter:
    """Writes JSON events incrementally (one JSON object per line)."""

    def __init__(self, path: Path, *, flush: bool = True):
        self._path = path
        self._flush = flush
        self._fp: IO[str] | None = None
        self._has_written = False

    def _ensure_open(self) -> None:
        if self._fp is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self._path.open("w", encoding="utf-8")

    def write(self, event: Artifact) -> None:
        self._ensure_open()
        assert self._fp is not None
        self._fp.write(event.model_dump_json() + "\n")
        if self._flush:
            self._fp.flush()
        self._has_written = True

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
        elif self._path.exists():
            self._path.unlink()

    @property
    def has_written(self) -> bool:
        return self._has_written


def open_jsonl(path: Path) -> JSONLWriter:
    return JSONLWriter(path)
