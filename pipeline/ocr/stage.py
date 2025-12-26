from __future__ import annotations
from typing import Any
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

        # Lazy import to avoid pulling Surya unless OCR is actually used
        from .engine import SuryaOCR

        engine = SuryaOCR()
        document = engine.process_pdf(ctx.pdf_path, parameters)

        if ctx.verbose:
            self._log_pages(document)

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

    # def _save_jsonl(self, document: OCRDocument, output_path: Path) -> None:
    #     self.jsonl_handler.save(
    #         (
    #             OCRPageResult(page_idx=i, ocr_page=ocr, layout_page=layout)
    #             for i, (ocr, layout) in enumerate(zip(document.ocr, document.layout))
    #         ),
    #         output_path,
    #     )

    # def _log_and_debug(self, document: OCRDocument) -> None:
    #     if not (self.ctx.verbose or self.ctx.debug):
    #         return

    #     pdf = pdfium.PdfDocument(str(self.ctx.pdf_path)) if self.ctx.debug else None
    #     try:
    #         for pidx, (ocr_page, layout_page) in enumerate(
    #             zip(document.ocr, document.layout)
    #         ):
    #             page = OCRPageResult(
    #                 page_idx=pidx, ocr_page=ocr_page, layout_page=layout_page
    #             )
    #             if self.ctx.verbose:
    #                 self._log_page(page)
    #             if self.ctx.debug and pdf is not None:
    #                 self._save_page_debug_visual(page, pdf[pidx])
    #     finally:
    #         if pdf is not None:
    #             pdf.close()

    # def run(self) -> OCRDocument:
    #     params = self.get_params()
    #     output_path = self.ctx.stage_dir(self.name) / params.output_artifact_filename

    #     if params.use_cache:
    #         document = self._load_from_cache()

    #     if document is None:
    #         logger.bind(stage=self.name).debug("Starting a fresh OCR...")
    #         # 2. OCR (Surya) + conversion vers OCRDocument
    #         ocr_pages = []
    #         layout_pages = []
    #         for ocr_p, lay_p in self.engine.process_pdf_stream(self.ctx.pdf_path):
    #             ocr_pages.append(ocr_p)
    #             layout_pages.append(lay_p)

    #         document = OCRDocument.from_surya(layout=layout_pages, ocr=ocr_pages)

    #     # 3. Post-processing (layout_ix + reading_order + spurious)
    #     document = postprocess_document(document)

    #     # 4. Save JSONL artifact (page-wise)
    #     self._save_jsonl(document, output_path)

    #     # 5. Cache
    #     if params.use_cache:
    #         save_cache(self.ctx.pdf_path, self.name, document, override=True)

    #     # 6. Verbose/debug visuals
    #     self._log_and_debug(document)

    #     return document

    # def _load_from_cache(self) -> Optional[OCRDocument]:
    #     params = self.get_params()
    #     artifact_model = OCRDocument
    #     document: Optional[OCRDocument] = None
    #     if params.use_cache:
    #         try:
    #             document = load_cache(self.ctx.pdf_path, self.name, artifact_model)
    #             log_msg = f"Loaded {artifact_model.__name__}."
    #             log_info = {
    #                 "cache": str(self.ctx.cache_path),
    #             }
    #         except dbm_error:
    #             log_msg = f"No {artifact_model.__name__}."
    #             log_info = {"error": dbm_error}
    #         finally:
    #             logger.bind(stage=self.name).info(log_msg, **log_info)
    #     return document

    # def _log_page(self, page_result: OCRPageResult) -> None:
    #     p, l = page_result.ocr_page, page_result.layout_page
    #     bbox_count = len(l.bboxes)
    #     rows = [
    #         [
    #             i + 1,
    #             (
    #                 l.bboxes[line.layout_ix].label
    #                 if line.layout_ix is not None and 0 <= line.layout_ix < bbox_count
    #                 else "?"
    #             ),
    #             f"{line.spurious:.2f}",
    #             f"{line.confidence:.2f}",
    #             line.text,
    #         ]
    #         for i, line in enumerate(p.text_lines)
    #     ]
    #     logger.info(
    #         f"📄 PAGE {page_result.page_idx + 1}\n{tabulate(rows, tablefmt='github')}"
    #     )

    # def _save_page_debug_visual(self, page_result: OCRPageResult, pdf_page) -> None:
    #     layout_debug_path = self.ctx.path(
    #         "layout", stage=self.name, ensure_exists=True, is_dir=True
    #     )
    #     bitmap = pdf_page.render(scale=2.0)
    #     image = bitmap.to_pil()

    #     fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=150)
    #     ax.imshow(image)
    #     colors = {}
    #     # Ratio Surya (72 DPI) vs Render (144 DPI)
    #     if (
    #         page_result.ocr_page.image_bbox is None
    #         or len(page_result.ocr_page.image_bbox) != 4
    #     ):
    #         raise ValueError("ocr_page.image_bbox is required for debug visualization")

    #     x0, _, x1, _ = page_result.ocr_page.image_bbox
    #     width = float(x1) - float(x0)
    #     if width <= 0:
    #         raise ValueError(
    #             f"Invalid ocr_page.image_bbox width: {page_result.ocr_page.image_bbox}"
    #         )
    #     scale = image.width / width

    #     for line in page_result.ocr_page.text_lines:
    #         if line.layout_ix is None:
    #             continue
    #         colors.setdefault(
    #             line.layout_ix, (random.random(), random.random(), random.random())
    #         )
    #         if not line.polygon:
    #             raise ValueError(f"Missing polygon for line: {line}")

    #         scaled_poly = [
    #             (float(x) * scale, float(y) * scale) for (x, y) in line.polygon
    #         ]
    #         ax.add_patch(
    #             patches.Polygon(
    #                 scaled_poly,
    #                 closed=True,
    #                 linewidth=2,
    #                 edgecolor=colors[line.layout_ix],
    #                 facecolor="none",
    #             )
    #         )

    #     fig.savefig(
    #         layout_debug_path / f"page_{page_result.page_idx + 1}.png",
    #         bbox_inches="tight",
    #     )
    #     plt.close(fig)
