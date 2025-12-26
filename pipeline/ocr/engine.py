from __future__ import annotations
from pathlib import Path
from pipeline.ocr.schemas import OCRDocument
from pipeline.ocr.stage import OCRParameters
from .postprocessing.layout import SuryaPostProcessor
from .postprocessing.spuriousness import Scorer

# Surya imports
from surya.common.surya.schema import TaskNames
from surya.input.load import load_from_file
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.settings import settings


class SuryaOCR:
    def __init__(self):
        self.foundation_predictor = FoundationPredictor(
            checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
        )
        self.detector = DetectionPredictor()
        self.layout_predictor = LayoutPredictor(self.foundation_predictor)
        self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
        self.post_processor = SuryaPostProcessor()

    def process_pdf(self, pdf_path: Path, parameters: OCRParameters) -> OCRDocument:

        # TODO : implement a loader mecanism for PDFs and folders similar to Suryas' CLILoader
        # https://github.com/datalab-to/surya/blob/master/surya/scripts/config.py

        images, _ = load_from_file(pdf_path)
        layout = self.layout_predictor(images)
        ocr = self.recognition_predictor(
            images=images,
            det_predictor=self.detector,
            task_names=[TaskNames.ocr_with_boxes for _ in images],
        )

        document = OCRDocument.from_surya(layout=layout, ocr=ocr)

        # Fix layout and reading order
        document = self.post_processor.process_ocr_output(document)

        # Assign spuriousness scores
        scorer = Scorer(min_chars=parameters.spuriousness["min_chars"])
        scorer.compute_and_assign(document)

        return document
