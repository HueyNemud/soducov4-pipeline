from __future__ import annotations
import random
from pathlib import Path
from typing import Dict, Type

import tabulate
from matplotlib import patches, pyplot as plt
from pydantic import BaseModel, Field

from pipeline.core.artifact import Artifact
from pipeline.core.context import RunContext
from pipeline.core.stage import PipelineStage, stage_config

# Main data and engine imports
from pipeline.ocr.schemas import OCRDocument
from pipeline.ocr.engine import SuryaOCR


class OCRParameters(BaseModel):
    """
    Parameters for the OCR pipeline stage.
    """

    engine: SuryaOCR.Parameters = Field(
        default_factory=SuryaOCR.Parameters,
        description="Parameters for the Surya OCR engine (Layout & Spuriousness settings).",
    )


@stage_config(produces=OCRDocument, params_model=OCRParameters())
class OCR(PipelineStage[OCRDocument, OCRParameters]):
    """
    Pipeline stage responsible for running OCR and post-processing filters.

    This stage acts as the orchestrator: it handles the PDF input, calls the
    SuryaOCR engine, and provides rich logging and visual debugging output.
    """

    def run(
        self,
        ctx: RunContext,
        parameters: OCRParameters,
        dependencies: Dict[Type[PipelineStage], Artifact],
    ) -> OCRDocument:
        """
        Executes the OCR engine on the provided PDF path.
        """
        # Lazy load the instance to manage heavy predictor initialization
        from .engine import SuryaOCR

        engine = SuryaOCR()
        document = engine.process_pdf(ctx.pdf_path, params=parameters.engine)

        if ctx.store.verbose or ctx.store.debug:
            self._log_pages(document)

        if ctx.store.debug:
            self._save_page_debug_visuals(ctx, document, ctx.pdf_path)

        return document

    def _log_pages(self, document: OCRDocument) -> None:
        """
        Generates a summary table of OCR results for logs.
        Helps verify layout classification and spuriousness scores at a glance.
        """
        if len(document.ocr) != len(document.layout):
            self.logger.warning(
                f"Page count mismatch: OCR({len(document.ocr)}) vs Layout({len(document.layout)})"
            )

        for p_idx, (p_res, l_res) in enumerate(zip(document.ocr, document.layout)):
            bbox_count = len(l_res.bboxes)
            rows = [
                [
                    i + 1,
                    (
                        l_res.bboxes[line.layout_ix].label
                        if line.layout_ix is not None
                        and 0 <= line.layout_ix < bbox_count
                        else "?"
                    ),
                    f"{line.spuriousness:.2f}",
                    f"{line.confidence:.2f}",
                    line.text[:50] + "..." if len(line.text) > 50 else line.text,
                ]
                for i, line in enumerate(p_res.text_lines)
            ]
            self.logger.info(
                f"📄 PAGE {p_idx + 1}\n{tabulate.tabulate(rows, tablefmt='github')}"
            )

    def _save_page_debug_visuals(
        self, ctx: RunContext, document: OCRDocument, pdf_path: Path
    ) -> None:
        """
        Renders PDF pages and overlays layout polygons for visual verification.
        Uses scaling to match Surya's 72 DPI coordinate system with render DPI.
        """
        import pypdfium2 as pdfium

        debug_dir = ctx.artifacts_dir / "ocr_debug_viz"
        debug_dir.mkdir(parents=True, exist_ok=True)

        with pdfium.PdfDocument(str(pdf_path)) as pdf:  # type: ignore[attr-defined] # pdfium2 stubs missing
            for idx, page_result in enumerate(document.ocr):
                # Render page at 2x scale for visual clarity
                page = pdf[idx]
                image = page.render(scale=2).to_pil()

                # Coordinate scaling: Convert Surya coords to image pixel space
                x0, _, x1, _ = page_result.image_bbox
                scale = image.width / (float(x1) - float(x0))

                fig, ax = plt.subplots(
                    figsize=(image.width / 100, image.height / 100), dpi=100
                )
                ax.imshow(image)

                colors = {}
                for line in page_result.text_lines:
                    if line.layout_ix is None or not line.polygon:
                        continue

                    # Ensure consistent coloring for lines belonging to the same layout block
                    color = colors.setdefault(
                        line.layout_ix,
                        (random.random(), random.random(), random.random()),
                    )
                    scaled_poly = [(x * scale, y * scale) for (x, y) in line.polygon]

                    ax.add_patch(
                        patches.Polygon(
                            scaled_poly,
                            closed=True,
                            linewidth=1.5,
                            edgecolor=color,
                            facecolor=color,
                            alpha=0.15,  # Semi-transparent to keep text readable
                        )
                    )

                ax.axis("off")
                fig.savefig(
                    debug_dir / f"page_{idx + 1:03d}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close(fig)
