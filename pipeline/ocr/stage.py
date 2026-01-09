from __future__ import annotations
import random
from typing import Any
from matplotlib import patches, pyplot as plt
from pydantic import BaseModel
import tabulate
from pipeline.core.artifact import Artifact
from pipeline.core.context import RunContext
from pipeline.core.stage import PipelineStage, stage_config
from .schemas import OCRDocument


class OCRParameters(BaseModel):
    spuriousness: dict[str, Any] = {"min_chars": 10}


@stage_config(produces=OCRDocument, params_model=OCRParameters())
class OCR(PipelineStage[OCRDocument, OCRParameters]):

    def run(
        self,
        ctx: RunContext,
        parameters: OCRParameters,
        dependencies: dict[type[PipelineStage], Artifact],
    ) -> OCRDocument:

        # Grab the SuryaOCR engine (and all its dependencies) only when needed
        from .engine import SuryaOCR

        engine = SuryaOCR()
        document = engine.process_pdf(ctx.pdf_path, parameters)

        if ctx.verbose or ctx.debug:
            self._log_pages(document)

        if ctx.debug:
            self._save_page_debug_visuals(ctx, document, ctx.pdf_path)

        return document

    def _log_pages(self, document: OCRDocument) -> None:
        if len(document.ocr) != len(document.layout):
            self.logger.warning(
                f"OCR pages count ({len(document.ocr)}) does not match layout pages count ({len(document.layout)})"
            )

        for p, l in zip(document.ocr, document.layout):
            bbox_count = len(l.bboxes)
            rows = [
                [
                    i + 1,
                    (
                        l.bboxes[line.layout_ix].label
                        if line.layout_ix is not None
                        and 0 <= line.layout_ix < bbox_count
                        else "?"
                    ),
                    f"{line.spuriousness:.2f}",
                    f"{line.confidence:.2f}",
                    line.text,
                ]
                for i, line in enumerate(p.text_lines)
            ]
            self.logger.info(f"📄 PAGE\n{tabulate.tabulate(rows, tablefmt='github')}")

    def _save_page_debug_visuals(self, ctx: RunContext, document, pdf_path) -> None:
        layout_debug_path = ctx.artifacts_dir / "ocr_layout_debug"
        layout_debug_path.mkdir(parents=True, exist_ok=True)
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        for page_idx, page_result in enumerate(document.ocr):
            pdf_page = pdf[page_idx]

            bitmap = pdf_page.render(scale=2)
            image = bitmap.to_pil()

            fig, ax = plt.subplots(
                figsize=(image.width / 100, image.height / 100), dpi=150
            )
            ax.imshow(image)
            colors = {}
            # Ratio Surya (72 DPI) vs Render (144 DPI)
            if page_result.image_bbox is None or len(page_result.image_bbox) != 4:
                raise ValueError(
                    "ocr_page.image_bbox is required for debug visualization"
                )

            x0, _, x1, _ = page_result.image_bbox
            width = float(x1) - float(x0)
            if width <= 0:
                raise ValueError(
                    f"Invalid ocr_page.image_bbox width: {page_result.image_bbox}"
                )
            scale = image.width / width

            for line in page_result.text_lines:
                if line.layout_ix is None:
                    continue
                colors.setdefault(
                    line.layout_ix, (random.random(), random.random(), random.random())
                )
                if not line.polygon:
                    raise ValueError(f"Missing polygon for line: {line}")

                scaled_poly = [
                    (float(x) * scale, float(y) * scale) for (x, y) in line.polygon
                ]
                ax.add_patch(
                    patches.Polygon(
                        scaled_poly,
                        closed=True,
                        linewidth=2,
                        edgecolor=colors[line.layout_ix],
                        facecolor="none",
                    )
                )

            fig.savefig(
                layout_debug_path / f"page_{page_idx + 1}.png",
                bbox_inches="tight",
            )
            plt.close(fig)
