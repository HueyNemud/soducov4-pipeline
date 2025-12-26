from typing import Optional
from pydantic import BaseModel, RootModel


class ChunkLine(BaseModel):
    page: int
    line: int
    layout_ix: Optional[int] = None
    layout_label: Optional[str] = None
    confidence: Optional[float] = None
    spuriousness: Optional[float] = None
    text: str
    is_margin: bool = False


class Chunk(RootModel[list[ChunkLine]]):

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)


class Chunks(RootModel[list[Chunk]]):

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)
