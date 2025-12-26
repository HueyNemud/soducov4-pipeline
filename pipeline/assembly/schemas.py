from pydantic import BaseModel, Field
from pipeline.chunking.schemas import ChunkLine
from pipeline.extraction.schemas import Entry, Structured, Text, Title


class RichItem(BaseModel):
    raw_text: str
    alignment: float = Field(..., ge=0.0, le=1.0)
    lines_resolved: list[ChunkLine] = Field(
        ..., description="List of resolved ChunkLines for this item"
    )
    bboxes: list[list[int]] = Field(
        ..., description="List of bounding boxes for each line in lines_resolved"
    )

    @classmethod
    def from_item(
        cls,
        item: Entry | Title | Text,
        **kwargs,
    ) -> "RichItem":
        item_cls = type(item).__name__

        # Append "Rich" to the class name to get the corresponding Rich class
        rich_cls = globals().get(f"Rich{item_cls}")
        if not rich_cls:
            raise ValueError(f"No Rich class found for item type {item_cls}")
        return rich_cls(
            **item.model_dump(),
            **kwargs,
        )


class RichEntry(Entry, RichItem):
    pass


class RichTitle(Title, RichItem):
    pass


class RichText(Text, RichItem):
    pass


class RichStructured(Structured):
    items: list[RichEntry | RichTitle | RichText]
    images_bboxes: list[list[int]] = []


# class RichStructuredSeq(RootModel[Sequence[RichStructured]]):

#     def __iter__(self):
#         return iter(self.root)

#     def __getitem__(self, item):
#         return self.root[item]

#     def __len__(self):
#         return len(self.root)
