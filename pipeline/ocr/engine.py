from __future__ import annotations
from pathlib import Path
from pydantic import Field
from pydantic.dataclasses import dataclass

# Pipeline Imports
from pipeline.ocr.schemas import OCRDocument
from pipeline.ocr.postprocessing.layout import SuryaLayoutPostProcessor
from pipeline.ocr.postprocessing.spuriousness import SpuriousnessScorer

# Surya Imports
from surya.common.surya.schema import TaskNames
from surya.input.load import load_from_file
from surya.settings import settings


class SuryaOCR:
    """
    Core OCR Engine leveraging Surya models for detection, layout analysis, and recognition.
    
    This engine handles the transition from raw PDF/images to a structured OCRDocument
    pivot format, applying post-processing filters (layout refinement and noise scoring).
    """

    @dataclass
    class Parameters:
        """
        Runtime parameters for the OCR engine and its post-processors.
        """
        layout: SuryaLayoutPostProcessor.Parameters = Field(
            default_factory=SuryaLayoutPostProcessor.Parameters
        )
        spuriousness: SpuriousnessScorer.Parameters = Field(
            default_factory=SpuriousnessScorer.Parameters
        )

    def __init__(self) -> None:
        """
        Initialize the heavy predictor models. 
        """
        
        # Lazy imports to avoid loading heavy models unless this class is instantiated        
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        
        # Load foundation model shared across layout and recognition
        self.foundation = FoundationPredictor(
            checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
        )
        
        # Initialize specialized predictors
        self.detector = DetectionPredictor()
        self.layout_predictor = LayoutPredictor(self.foundation)
        self.recognition_predictor = RecognitionPredictor(self.foundation)
        
        # Initialize stateless post-processing tools
        self.layout_pp = SuryaLayoutPostProcessor()
        self.spuriousness_scorer = SpuriousnessScorer()

    def process_pdf(self, pdf_path: Path, params: SuryaOCR.Parameters | None = None) -> OCRDocument:
        """
        Executes the full OCR pipeline on a PDF file.

        Args:
            pdf_path: Path to the source PDF.
            params: Runtime parameters for the engine and post-processors. 
                    Uses default parameters if None.

        Returns:
            An enriched OCRDocument instance.
        """

        p = params or self.Parameters()
        
        # 1. Load and Predict
        images, _ = load_from_file(pdf_path)
        if not images:
            raise ValueError(f"No images could be loaded from {pdf_path}")

        layout_results = self.layout_predictor(images)
        ocr_results = self.recognition_predictor(
            images=images,
            det_predictor=self.detector,
            task_names=[TaskNames.ocr_with_boxes] * len(images),
        )

        # 2. Convert to Pivot Format
        document = OCRDocument.from_surya(layout=layout_results, ocr=ocr_results)
    
        # 3. Apply Post-Processing

        # Refine layout boxes and determine fine-grained reading order
        document = self.layout_pp.process_ocr_output(document, params=p.layout)
        
        # Score lines for noise (spuriousness), specifically targeting margin artifacts
        self.spuriousness_scorer.compute_and_assign(document, params=p.spuriousness)

        return document

    def __repr__(self) -> str:
        return f"<SuryaOCR(model='{settings.LAYOUT_MODEL_CHECKPOINT}')>"