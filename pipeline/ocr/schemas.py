from __future__ import annotations
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from surya.recognition.schema import (
    OCRResult as SuryaOCRResult,
    TextLine as SuryaTextLine,
)
from surya.layout.schema import LayoutResult


class TextLine(SuryaTextLine):
    """
    Extended Surya TextLine with additional metadata for directory processing.
    """

    reading_order_ix: Optional[int] = Field(
        None, description="Index in the final reading sequence."
    )
    layout_ix: Optional[int] = Field(
        None, description="Index of the associated layout block."
    )
    spuriousness: float = Field(
        default=float("nan"), description="Noise score (0.0 to 1.0)."
    )

    @classmethod
    def from_surya(cls, line: SuryaTextLine | dict[str, Any]) -> TextLine:
        """Converts a raw Surya line or dict into an enriched TextLine."""
        data = line.model_dump() if isinstance(line, SuryaTextLine) else line
        return cls(**data)


class OCRResult(SuryaOCRResult):
    """
    Enhanced page-level OCR result containing enriched text lines.
    """

    text_lines: List[TextLine]

    @classmethod
    def from_surya(cls, result: SuryaOCRResult | dict[str, Any]) -> OCRResult:
        """
        Transfers data from Surya's OCRResult to our extended OCRResult,
        ensuring all text lines are upgraded to our TextLine class.
        """
        if isinstance(result, SuryaOCRResult):
            return cls(
                text_lines=[TextLine.from_surya(tl) for tl in result.text_lines],
                image_bbox=result.image_bbox,
            )

        # Dictionary-based initialization for flexibility
        return cls(
            text_lines=[TextLine.from_surya(tl) for tl in result.get("text_lines", [])],
            image_bbox=result.get("image_bbox", []),
        )


class OCRDocument(BaseModel):
    """
    The main pivot document structure.
    It holds both the geometric layout blocks and the recognized text.
    """

    layout: List[LayoutResult]
    ocr: List[OCRResult]

    @classmethod
    def from_surya(
        cls, layout: List[LayoutResult], ocr: List[SuryaOCRResult]
    ) -> OCRDocument:
        """
        Factory method to assemble an OCRDocument from raw Surya model outputs.
        """
        return cls(layout=layout, ocr=[OCRResult.from_surya(r) for r in ocr])
