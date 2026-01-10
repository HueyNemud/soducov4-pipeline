"""
Pipeline Orchestration Engine.

This module provides the core Pipeline class responsible for topological sorting
of stages, dependency resolution, execution with caching support, and
real-time artifact streaming.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypedDict, Optional

from pydantic import BaseModel
from pipeline.core.artifact import (
    Artifact,
    FileCache,
    fingerprint,
    load_artifact,
    open_jsonl,
    save_artifact,
)
from pipeline.core.context import RunContext
from pipeline.core.stage import (
    Parameters,
    PipelineStage,
    toposort,
)
from ..logging import logger


class ExecutionParameters(TypedDict):
    """Global configuration for a pipeline execution run."""

    pdf_path: Path
    artifacts_dir: Path
    debug: bool
    verbose: bool


class Pipeline:
    """
    Manages the sequential execution of document processing stages.

    It ensures stages are run in the correct order based on dependencies,
    manages the persistence of artifacts, and handles caching to avoid
    redundant computations.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        ctx: RunContext,
        cache: Optional[FileCache] = None,
    ):
        """Initializes the pipeline with a sorted set of stages."""
        self.stages = toposort(stages)
        self.ctx = ctx
        self.cache = cache
        self._post_stage_hooks: list[Callable[[PipelineStage, Any], None]] = []

        # Register stages and their default parameters in the context store
        for stage in self.stages:
            if stage.name not in self.ctx.store.stages:
                self.ctx.store.stages[stage.name] = {
                    "params": stage.validate_params({})
                }

    def run(
        self,
        stage_params: Optional[dict[str, Mapping | Parameters]] = None,
        force_compute: Optional[Sequence[str]] = None,
    ) -> Iterator[tuple[str, Optional[Artifact]]]:
        """
        Executes all stages in the pipeline.

        Args:
            stage_params: Parameter overrides indexed by stage name.
            force_compute: List of stage names to recompute regardless of cache.

        Yields:
            Tuple of (stage_name, produced_artifact).
        """
        force_compute = force_compute or []

        for index, stage in enumerate(self.stages):
            logger.info(f"▶️  Step {index + 1}/{len(self.stages)}: {stage.name.upper()}")

            # 1. Resolve parameters (defaults + overrides)
            params = self._get_stage_params(stage, stage_params)

            # 2. Resolve dependencies from previous stages or disk
            dependencies = self._resolve_dependencies(stage)

            artifact: Optional[Artifact] = None

            # 3. Cache Check: Try to load existing artifact unless forced
            if self.cache:
                if stage.name in force_compute:
                    self.cache.invalidate(stage.name)
                else:
                    artifact = self._load_from_cache(stage, params)

            # 4. Computation: Run the stage if no cached artifact was found
            if artifact is None:
                logger.debug(f"Computing stage '{stage.name}'...")

                with self.artifact_streaming(stage):
                    artifact = stage.run(self.ctx, params, dependencies)

                # Persist result to the main artifact file
                if artifact:
                    save_artifact(artifact, self._get_artifact_path(type(stage)))

            # 5. Update Cache: Save the newly computed artifact
            if self.cache and artifact:
                self.cache.save(
                    stage.name,
                    artifact,
                    override=True,
                    meta=compute_stage_meta(stage, params),
                )

            # 6. Finalize: Store in context memory and notify hooks
            self.ctx.store.artifacts[stage.name] = artifact
            yield stage.name, artifact

            for hook in self._post_stage_hooks:
                hook(stage, artifact)

    @contextmanager
    def artifact_streaming(self, stage: PipelineStage) -> Iterator[None]:
        """Context manager to enable real-time JSONL streaming during stage execution."""
        path = self._get_stream_path(type(stage))
        writer = open_jsonl(path)
        self.ctx.attach_stream(writer)
        try:
            yield
        finally:
            self.ctx.detach_stream()
            writer.close()

    def _get_artifact_path(self, stage_cls: type[PipelineStage]) -> Path:
        """Returns the standard path for a stage's JSON artifact."""
        return self.ctx.artifacts_dir / f"{stage_cls.metadata().name}.json"

    def _get_stream_path(self, stage_cls: type[PipelineStage]) -> Path:
        """Returns the standard path for a stage's JSONL stream file."""
        return self.ctx.artifacts_dir / f"{stage_cls.metadata().name}.items.jsonl"

    def _get_stage_params(
        self,
        stage: PipelineStage,
        overrides: Optional[dict[str, Mapping | Parameters]] = None,
    ) -> Parameters:
        """Merges store parameters with runtime overrides and validates them."""
        params_dict = dict(self.ctx.store.stages[stage.name].params)

        stage_overrides = (overrides or {}).get(stage.name, {})
        if isinstance(stage_overrides, BaseModel):
            stage_overrides = stage_overrides.model_dump()

        params_dict.update(stage_overrides)
        return stage.validate_params(params_dict)

    def _resolve_dependencies(self, stage: PipelineStage) -> dict[str, Artifact]:
        """Ensures all required artifacts for a stage are loaded into memory."""
        resolved: dict[str, Artifact] = {}

        for dep_cls in stage.consumes:
            meta = dep_cls.metadata()

            if not meta.produces:
                raise ValueError(
                    f"Stage {stage.name} depends on non-producing stage {meta.name}"
                )

            # If not in memory, attempt to load from the artifact file
            if meta.name not in self.ctx.store.artifacts:
                path = self._get_artifact_path(dep_cls)
                artifact = load_artifact(path, meta.produces)
                self.ctx.store.artifacts[meta.name] = artifact

            resolved[meta.name] = self.ctx.store.artifacts[meta.name]

        return resolved

    def _load_from_cache(self, stage: PipelineStage, params: Any) -> Optional[Artifact]:
        """Attempts to retrieve a valid artifact from the file cache."""
        if not self.cache or not stage.produces:
            return None

        try:
            expected_meta = compute_stage_meta(stage, params)
            artifact = self.cache.load(
                stage.name, stage.produces, expected_meta=expected_meta
            )
            logger.debug(f"Cache hit for stage '{stage.name}'")
            return artifact
        except KeyError:
            logger.debug(f"Cache miss for stage '{stage.name}'")
            return None

    @staticmethod
    def for_pdf(
        pdf_path: Path,
        stages: Optional[list[PipelineStage]] = None,
        debug: bool = False,
        verbose: bool = False,
        enable_cache: bool = True,
    ) -> "Pipeline":
        """Factory method to initialize a Pipeline for a specific PDF file."""
        artifacts_dir = pdf_path.parent / pdf_path.stem / "artifacts"

        config = ExecutionParameters(
            pdf_path=pdf_path.resolve(),
            artifacts_dir=artifacts_dir.resolve(),
            debug=debug,
            verbose=verbose,
        )

        ctx = RunContext(**config)
        cache = FileCache(pdf_path) if enable_cache else None

        return Pipeline(stages=stages or [], ctx=ctx, cache=cache)


def compute_stage_meta(stage: PipelineStage, params: Any) -> dict[str, Any]:
    """Generates a metadata fingerprint for cache validation."""
    cls = type(stage)
    return {
        "params_fp": fingerprint(params),
        "metadata": cls.metadata(),
        "stage_impl": f"{cls.__module__}.{cls.__qualname__}",
    }
