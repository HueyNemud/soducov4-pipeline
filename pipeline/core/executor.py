from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypedDict

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
    pdf_path: Path
    artifacts_dir: Path
    debug: bool
    verbose: bool


class Pipeline:

    def __init__(
        self,
        stages: list[PipelineStage],
        ctx: RunContext,
        cache: FileCache | None = None,
    ):
        self.stages = toposort(stages)
        self.ctx = ctx
        self.cache = cache
        self._post_stage_hooks: list[Callable[[PipelineStage, Any], None]] = []

        # Register stages in the context store
        for stage in self.stages:
            if stage.params_model:
                self.ctx.store.deep_set(
                    f"stages.{stage.name}.params", stage.params_model
                )

    def run(
        self,
        stage_params: dict[str, Mapping | Parameters] | None = None,
        force_compute: Sequence[str] | None = None,
    ) -> Iterator[tuple[str, Artifact | None]]:

        force_compute = force_compute or []
        cache = self.cache

        for pos, stage in enumerate(self.stages):
            logger.info(f"▶️  {pos + 1}:{stage.name.upper()}")

            # 1. Récupère les paramètres pour cette étape
            # Avec mise à jour par les overrides passés en argument
            params = self._get_stage_params(stage, stage_params)

            artifact: Artifact | None = None

            # 3. Résout les dépendances
            dependencies = self._resolve_dependencies(stage)

            # 4. Tente de charger depuis le cache ou calcule l'artifact
            if cache:
                if stage.name in force_compute:
                    cache.invalidate(stage.name)
                else:
                    artifact = self._load_from_cache(stage, params)

            if artifact is None:
                logger.debug(
                    f"Computing artifact for stage {stage.name}...",
                    params=params,
                    dependencies=dependencies.keys(),
                )

                with self.artifact_streaming(stage):
                    artifact = stage.run(self.ctx, params, dependencies)

                if artifact:
                    save_artifact(
                        artifact, self._artifact_path_for_stage_class(type(stage))
                    )

            if cache and artifact:
                cache.save(
                    stage.name,
                    artifact,
                    override=True,
                    meta=compute_stage_meta(stage, params),
                )

            self.ctx.store.artifacts[stage.name] = artifact

            yield stage.name, artifact

            for hook in self._post_stage_hooks:
                hook(stage, artifact)

    @contextmanager
    def artifact_streaming(self, stage: PipelineStage) -> Iterator[None]:
        writer = open_jsonl(self._artifact_stream_path_for_stage_class(type(stage)))
        self.ctx.set_stream(writer)
        try:
            yield
        finally:
            self.ctx.clear_stream()
            writer.close()

    def _artifact_path_for_stage_class(self, stage: type[PipelineStage]) -> Path:
        return self.ctx.artifacts_dir / f"{stage.metadata().name}.json"

    def _artifact_stream_path_for_stage_class(self, stage: type[PipelineStage]) -> Path:
        return self.ctx.artifacts_dir / f"{stage.metadata().name}.items.jsonl"

    def _get_stage_params(
        self,
        stage: PipelineStage,
        stage_params: dict[str, Mapping | Parameters] | None = None,
    ) -> Parameters:
        params = dict(self.ctx.store.stages[stage.name].params)
        overrides = (stage_params or {}).get(stage.name, {})
        if isinstance(overrides, BaseModel):
            overrides = overrides.model_dump()
        params.update(overrides)

        validated_params = stage.params_model.model_validate(params)
        return validated_params

    def _resolve_dependencies(
        self,
        stage: PipelineStage,
    ) -> dict[str, Artifact]:
        resolved_dependencies: dict[str, Artifact] = {}
        for dep_cls in stage.consumes:
            dep_meta = dep_cls.metadata()
            if dep_meta.produces is None:
                raise ValueError(
                    f"Stage {stage.name} depends on {dep_meta.name} which does not produce any artifact."
                )
            if dep_meta.name not in self.ctx.store.artifacts:
                # Essaye de charger depuis le fichier d'artifact
                dep_artifact_path = self._artifact_path_for_stage_class(dep_cls)
                dep_artifact = load_artifact(
                    dep_artifact_path, dep_meta.produces or Artifact
                )
                self.ctx.store.artifacts[dep_meta.name] = dep_artifact

            try:
                resolved_dependencies[dep_meta.name] = self.ctx.store.artifacts[
                    dep_meta.name
                ]
            except KeyError:
                raise ValueError(
                    f"Dependency '{dep_meta.name}' for stage '{stage.name}' not found in context artifacts."
                )
        return resolved_dependencies

    def _load_from_cache(
        self,
        stage: PipelineStage,
        params: dict,
    ) -> Artifact | None:
        cache = self.cache
        artifact_type = stage.produces

        artifact: Artifact | None = None
        if cache and artifact_type:
            try:
                expected_meta = compute_stage_meta(stage, params)
                artifact = cache.load(
                    stage.name, artifact_type, expected_meta=expected_meta
                )
                stage.logger.debug("Loaded artifact from cache.", meta=expected_meta)
                return artifact
            except KeyError:
                stage.logger.debug("Cache miss; computing artifact.")

        return artifact

    @staticmethod
    def for_pdf(
        pdf_path: Path,
        stages: list[PipelineStage] | None = None,
        debug: bool = False,
        verbose: bool = False,
        enable_cache: bool = True,
    ) -> "Pipeline":
        """Factory method to create a Pipeline for a given PDF file."""

        artifacts_dir = pdf_path.parent / pdf_path.stem / "artifacts"
        execution_parameters = ExecutionParameters(
            pdf_path=pdf_path.resolve(),
            artifacts_dir=artifacts_dir.resolve(),
            debug=debug,
            verbose=verbose,
        )
        ctx = RunContext(**execution_parameters)
        cache = FileCache(ctx.store.pdf_path) if enable_cache else None
        return Pipeline(
            stages=stages or [],
            ctx=ctx,
            cache=cache,
        )


def compute_stage_meta(
    stage: PipelineStage,
    params: dict,
) -> dict[str, Any]:
    cls = type(stage)
    return {
        "params_fp": fingerprint(params),
        "metadata": cls.metadata(),
        "stage_impl": f"{cls.__module__}.{cls.__qualname__}",
    }
