"""
Pipeline Stage Definitions.

Provides base classes and decorators for defining pipeline stages,
managing their dependencies, and handling artifact typing.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Generic,
    Sequence,
    TypeVar,
    cast,
    TYPE_CHECKING,
)

from pydantic import BaseModel
from pipeline.core.artifact import Artifact
from pipeline.logging import logger

if TYPE_CHECKING:
    from .context import RunContext

# Type variables for Artifact (Output) and Parameters (Input)
_T = TypeVar("_T", bound=Artifact)
_P = TypeVar("_P", bound=BaseModel)

Parameters = _P


class StageMeta(BaseModel):
    """Metadata describing a stage's contract within the pipeline."""

    name: str
    produces: type[Artifact] | None
    consumes: frozenset[type[PipelineStage]]
    params_model: BaseModel | None


class PipelineStage(ABC, Generic[_T, _P]):
    """
    Abstract base class for all pipeline processing steps.

    Stages define what they consume (dependencies), what they produce
    (artifacts), and the parameters required for their execution.
    """

    # Class-level metadata initialized via the @stage_config decorator
    _produces: ClassVar[type[Artifact] | None] = None
    _consumes: ClassVar[frozenset[type[PipelineStage]]] = frozenset()
    _params_model: ClassVar[BaseModel | None] = None

    def __init_subclass__(cls, **kwargs):
        """Ensures that metadata attributes exist on all subclasses."""
        super().__init_subclass__(**kwargs)
        cls._produces = getattr(cls, "_produces", None)
        cls._consumes = frozenset(getattr(cls, "_consumes", ()))
        cls._params_model = getattr(cls, "params_model", None)

    @abstractmethod
    def run(
        self,
        ctx: RunContext,
        parameters: _P,
        dependencies: dict[str, Artifact],
    ) -> _T | None:
        """
        Executes the stage logic.

        Args:
            ctx: The execution context for emitting logs or items.
            parameters: Validated parameters for this run.
            dependencies: Mapping of stage names to their produced artifacts.
        """
        ...

    @property
    def logger(self):
        """Returns a logger instance bound to the specific stage name."""
        return logger.bind(stage=self.name)

    @property
    def produces(self) -> type[Artifact] | None:
        """The type of Artifact class this stage generates."""
        return self._produces

    @property
    def consumes(self) -> frozenset[type[PipelineStage]]:
        """A set of Stage classes that must run before this one."""
        return self._consumes

    @property
    def name(self) -> str:
        """The lowercase unique identifier for this stage."""
        return self.name_cls()

    @classmethod
    def name_cls(cls) -> str:
        """Returns the default stage name based on the class name."""
        return cls.__name__.lower()

    @classmethod
    def metadata(cls) -> StageMeta:
        """Returns a snapshot of the stage's configuration."""
        return StageMeta(
            name=cls.name_cls(),
            produces=cls._produces,
            consumes=cls._consumes,
            params_model=cls._params_model,
        )

    @classmethod
    def validate_params(cls, data: dict[str, Any]) -> _P | None:
        """
        Validates and constructs the parameters model from raw data.

        Raises:
            ValueError: If no params_model is defined for this stage.
        """
        if cls._params_model is None:
            return None
        p = cls._params_model.model_validate(data)
        return cast(_P, p)


def stage_config(
    produces: type[Artifact] | None = None,
    depends_on: Sequence[type[PipelineStage]] | None = None,
    params_model: BaseModel | None = None,
):
    """
    Decorator to configure a PipelineStage.

    Example:
        @stage_config(produces=MyArtifact, depends_on=[OCRStage])
        class MyStage(PipelineStage[MyArtifact, MyParams]):
            ...
    """

    def decorator(cls: type[PipelineStage[_T, _P]]) -> type[PipelineStage[_T, _P]]:
        if params_model is not None:
            if not isinstance(params_model, BaseModel):
                raise TypeError(
                    f"{cls.__name__} params_model must be a Pydantic instance."
                )
            cls._params_model = params_model

        cls._produces = produces
        cls._consumes = frozenset(depends_on) if depends_on else frozenset()
        return cls

    return decorator


def toposort(stages: list[PipelineStage]) -> list[PipelineStage]:
    """
    Sorts stages based on their dependency graph.
    Currently returns the list as provided (placeholder for Kahn's algorithm).
    """
    # TODO: Implement full dependency resolution
    return stages


def safe_get_dependency(
    dependencies: dict[str, Artifact], dep_stage: type[PipelineStage[_T, Any]]
) -> _T:
    """
    Retrieves and type-checks a required artifact from the dependency map.

    Raises:
        ValueError: If the dependency is missing.
        TypeError: If the artifact type doesn't match the stage's declaration.
    """
    meta = dep_stage.metadata()
    name = meta.name
    expected_type = meta.produces

    if name not in dependencies:
        raise ValueError(f"Required dependency '{name}' not found.")

    artifact = dependencies[name]

    if expected_type is None:
        raise TypeError(f"Stage '{name}' does not produce an artifact.")

    if not isinstance(artifact, expected_type):
        actual_type = type(artifact).__name__
        raise TypeError(
            f"Type mismatch for '{name}': expected {expected_type.__name__}, got {actual_type}"
        )

    return cast(_T, artifact)
