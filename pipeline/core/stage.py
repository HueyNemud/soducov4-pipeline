"""Pipeline stage base classes and artifact handlers."""

from __future__ import annotations
from typing import (
    ClassVar,
    Generic,
    Sequence,
    TypeVar,
    cast,
)
from pydantic import BaseModel
from pipeline.core.artifact import Artifact
from .context import RunContext
from abc import ABC, abstractmethod
from pipeline.logging import logger

# Artifact
_T = TypeVar("_T", bound=Artifact)
# Parameters
_P = TypeVar("_P", bound=BaseModel)
Parameters = _P


class StageMeta(BaseModel):
    name: str
    produces: type[Artifact] | None
    consumes: frozenset[type[PipelineStage]]
    params_model: Parameters | None


class PipelineStage(ABC, Generic[_T, _P]):
    """Base class for all pipeline stages."""

    # Propriété de classe indiquant le type d'artifact produit par cette étape
    _produces: ClassVar[type[Artifact] | None]

    # Propriété de classe listant les PipelineStages dont cette étape dépend
    _consumes: ClassVar[frozenset[type[PipelineStage]]]

    params_model: ClassVar[Parameters]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Initialise les attributs de classe si non definis
        cls._produces = getattr(cls, "_produces", None)
        cls._consumes = frozenset(getattr(cls, "_consumes", ()))
        cls.params_model = getattr(cls, "params_model", None)

    @abstractmethod
    def run(
        self,
        _ctx: RunContext,
        parameters: _P,
        dependencies: dict[str, Artifact],
    ) -> _T | None: ...

    @property
    def logger(self):
        """Helper function to get a logger bound to this stage."""
        return logger.bind(stage=self.name)

    @property
    def produces(self) -> type[Artifact] | None:
        return type(self)._produces

    @property
    def consumes(self) -> frozenset[type[PipelineStage]]:
        return frozenset(type(self)._consumes)

    @property
    def defaults_parameters(self) -> _P | None:
        """Defaults are given by the class-level params_model instance, if any."""
        cls_defaults = type(self).params_model
        return cast(_P, cls_defaults.model_copy(deep=True)) if cls_defaults else None

    @property
    def name(self) -> str:
        return type(self).name_cls()

    @classmethod
    def name_cls(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def metadata(cls) -> StageMeta:
        return StageMeta(
            name=cls.name_cls(),
            produces=cls._produces,
            consumes=cls._consumes or frozenset(),
            params_model=cls.params_model,
        )


def stage_config(
    produces: type[Artifact] | None = None,
    depends_on: Sequence[type[PipelineStage]] | None = None,
    params_model: Parameters | None = None,
):
    def decorator(cls: type[PipelineStage[_T, _P]]) -> type[PipelineStage[_T, _P]]:

        if params_model is not None:
            if not isinstance(params_model, BaseModel):
                raise TypeError(
                    f"{cls.__name__} parameter model must be a Pydantic BaseModel instance."
                )
            cls.params_model = params_model

        cls._produces = produces
        cls._consumes = frozenset(depends_on) if depends_on else frozenset()
        return cls

    return decorator


def toposort(stages: list[PipelineStage[_T, _P]]) -> list[PipelineStage[_T, _P]]:
    """Topologically sort the given stages based on their dependencies."""
    logger.warning(
        "Not yet implemented: toposort function in pipeline/core/executor.py"
    )
    return stages


def safe_get_dependency(
    dependencies: dict[str, Artifact], dep_type: type[PipelineStage[_T, _P]]
) -> _T:
    """Helper to get a dependency and ensure it is of the expected type."""
    name = dep_type.metadata().name
    artifact_type = dep_type.metadata().produces
    try:
        artifact = dependencies[name]
    except KeyError:
        raise ValueError(f"Dependency '{name}' not found in dependencies.")

    if artifact_type is None:
        raise TypeError(f"Dependency '{name}' does not declare any produced artifact.")

    if not isinstance(artifact, artifact_type):
        raise TypeError(
            f"Dependency '{name}' is of type {type(artifact).__name__}, "
            f"expected {artifact_type.__name__}."
        )

    return cast(_T, artifact)
