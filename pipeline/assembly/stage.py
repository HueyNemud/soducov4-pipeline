from pydantic import BaseModel
from pipeline.assembly.schemas import (
    RichEntry,
    RichStructured,
    RichText,
    RichTitle,
)
from pipeline.chunking.stage import Chunking
from pipeline.core.artifact import Artifact
from pipeline.core.context import RunContext
from pipeline.core.stage import PipelineStage, safe_get_dependency, stage_config
from pipeline.extraction.schemas import Structured
from pipeline.extraction.stage import Extraction
from pipeline.ocr.stage import OCR


class AssemblyParameters(BaseModel):
    strict: bool = False


@stage_config(
    produces=Structured,
    depends_on=[OCR, Chunking, Extraction],
    params_model=AssemblyParameters(),
)
class Assembly(PipelineStage[Structured, AssemblyParameters]):

    def run(
        self,
        ctx: RunContext,
        parameters: AssemblyParameters,
        dependencies: dict[str, Artifact],
    ) -> RichStructured:
        self.logger.info(
            f"Starting {self.name.upper()} stage with parameters: {parameters}"
        )

        # Implémentation de l'assemblage ici

        # Charge les dépendances
        ocr_artifact = safe_get_dependency(dependencies, OCR)
        chunking_artifact = safe_get_dependency(dependencies, Chunking)
        extraction_artifact = safe_get_dependency(dependencies, Extraction)

        from .engine import Assembler

        assembler = Assembler()

        result: list[RichEntry | RichTitle | RichText] = []
        for assembled in assembler.assemble_multiple(
            ocr=ocr_artifact,
            chunks=chunking_artifact,
            structuredseq=extraction_artifact,
            strict=parameters.strict,
        ):
            ctx.emit(assembled)
            result.extend(assembled.items)
        images_bboxes = [[int(_) for _ in page.image_bbox] for page in ocr_artifact.ocr]
        # Logique d'assemblage à implémenter
        return RichStructured(
            items=result,
            images_bboxes=images_bboxes,
        )
