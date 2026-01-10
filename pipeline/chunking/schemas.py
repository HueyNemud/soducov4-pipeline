"""
Text Chunking & Structured Output Schemas.

This module defines the data structures for 'chunks'—logical groupings of text lines.
It acts as the final Data Transfer Object (DTO) layer, encapsulating OCR results,
spatial metadata, and quality scores for downstream extraction tasks.
"""

from typing import Optional, Iterator, List
from pydantic import BaseModel, RootModel, Field


class ChunkLine(BaseModel):
    """
    Represents an enriched line of text with its associated metadata.

    Contains the raw content and diagnostic signals (confidence, spuriousness)
    required to evaluate the quality of directory listings and layout context.
    """

    page: int
    line: int
    layout_ix: Optional[int] = Field(
        None, description="Index of the parent layout block"
    )
    layout_label: Optional[str] = Field(
        None, description="Label assigned to the layout block"
    )

    text: str
    confidence: Optional[float] = Field(None, description="OCR engine confidence score")
    spuriousness: Optional[float] = Field(
        None, description="Probability score of the line being noise"
    )
    is_margin: bool = Field(
        False, description="Flag indicating if the line belongs to page margins"
    )

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class Chunk(RootModel[List[ChunkLine]]):
    """
    A logical sequence of ChunkLine objects representing a single text segment.

    Provides standard collection behaviors (iteration, indexing, length) to
    seamlessly handle groups of lines belonging to the same entity or listing.
    """

    def __iter__(self) -> Iterator[ChunkLine]:
        return iter(self.root)

    def __getitem__(self, index: int) -> ChunkLine:
        return self.root[index]

    def __len__(self) -> int:
        return len(self.root)


class Chunks(RootModel[List[Chunk]]):
    """
    A top-level container for multiple Chunk instances.

    Used to aggregate extracted segments across a document or page for batch
    processing or final data export.
    """

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self.root)

    def __getitem__(self, index: int) -> Chunk:
        return self.root[index]

    def __len__(self) -> int:
        return len(self.root)
