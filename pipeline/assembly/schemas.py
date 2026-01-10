from __future__ import annotations

from typing import Any, Union
from pydantic import BaseModel, Field

from pipeline.chunking.schemas import ChunkLine
from pipeline.extraction.schemas import Entry, Structured, Text, Title


class RichItem(BaseModel):
    """
    Base class for items enriched with source text metadata and spatial information.

    Attributes:
        raw_text: The original text as it appeared in the source document.
        alignment: Confidence score (0 to 1) of the text alignment.
        lines_resolved: Detailed line objects mapped back to the source chunk.
        bboxes: Spatial coordinates [x0, y0, x1, y1] for each resolved line.
    """

    raw_text: str
    alignment: float = Field(..., ge=0.0, le=1.0)
    lines_resolved: list[ChunkLine] = Field(
        ..., description="List of resolved ChunkLines for this item"
    )
    bboxes: list[list[int]] = Field(
        ..., description="Bounding boxes corresponding to each resolved line"
    )

    @classmethod
    def from_item(
        cls,
        source_item: Union[Entry, Title, Text],
        **extra_data: Any,
    ) -> "RichItem":
        """
        Factory method to convert a standard extraction item into its 'Rich' equivalent.

        Args:
            source_item: The basic Entry, Title, or Text object.
            **extra_data: Enrichment data (raw_text, alignment, bboxes, etc.).
        """
        source_type_name = type(source_item).__name__
        rich_type_name = f"Rich{source_type_name}"

        # Retrieve the enriched class definition from the module namespace
        rich_class: type[RichItem] | None = globals().get(rich_type_name)

        if not rich_class:
            raise ValueError(f"Mapping failed: {rich_type_name} is not defined.")

        return rich_class(
            **source_item.model_dump(),
            **extra_data,
        )


class RichEntry(Entry, RichItem):
    """Enriched version of a structured Entry."""

    pass


class RichTitle(Title, RichItem):
    """Enriched version of a document Title."""

    pass


class RichText(Text, RichItem):
    """Enriched version of a generic Text block."""

    pass


class RichStructured(Structured):
    """
    A collection of enriched items, including optional spatial data for
    non-textual elements like images.
    """

    items: list[Union[RichEntry, RichTitle, RichText]]
    images_bboxes: list[list[int]] = Field(
        default_factory=list,
        description="Bounding boxes for detected images within the structured scope",
    )
