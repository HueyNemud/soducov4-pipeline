from __future__ import annotations
from typing import Any, List, Optional, Type, cast
from pydantic import BaseModel
from surya.recognition.schema import (
    OCRResult as SuryaOCRResult,
    TextLine as SuryaTextLine,
)
from surya.layout.schema import LayoutResult


class TextLine(SuryaTextLine):
    reading_order_ix: Optional[int] = None
    layout_ix: Optional[int] = None
    spuriousness: Optional[float] = float("nan")

    @classmethod
    def from_surya(cls, line: SuryaTextLine | dict[str, Any]) -> "TextLine":
        if isinstance(line, SuryaTextLine):
            return cls(**line.model_dump())
        return cls(**line)


class OCRResult(SuryaOCRResult):
    text_lines: List[TextLine]

    @classmethod
    def from_surya(cls, result: SuryaOCRResult | dict[str, Any]) -> "OCRResult":
        # cast cls pour Pylance, pour indiquer qu'il a bien la méthode from_surya
        target_cls = cast(Type[OCRResult], cls)

        if isinstance(result, SuryaOCRResult):
            return target_cls(
                text_lines=[TextLine.from_surya(tl) for tl in result.text_lines],
                image_bbox=result.image_bbox,
            )

        # dict-like
        text_lines = result.get("text_lines", [])
        return target_cls(
            text_lines=[TextLine.from_surya(tl) for tl in text_lines],
            image_bbox=result.get("image_bbox", []),
        )


class OCRDocument(BaseModel):
    layout: List[LayoutResult]
    ocr: List[OCRResult]

    @classmethod
    def from_surya(
        cls, layout: List[LayoutResult], ocr: List[SuryaOCRResult]
    ) -> "OCRDocument":
        return cls(layout=layout, ocr=[OCRResult.from_surya(r) for r in ocr])
