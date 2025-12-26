from __future__ import annotations
from typing import Any, Iterable
from thefuzz import fuzz
from pydantic import BaseModel
from pipeline.chunking.schemas import Chunk
from .schemas import (
    EntryCompact,
    Structured,
    Entry,
    TextCompact,
    Title,
    Text,
    TitleCompact,
)
from pipeline.logging import logger


def text_fingerprint(text: str) -> str:
    """Simplified text fingerprint: lowercase, alphanumeric, sorted."""
    filtered = "".join(c.lower() for c in text if c.isalnum())
    return "".join(sorted(filtered))


def chunk_text_for_lines(chunk: Chunk, lines: list[int]) -> str:
    """Extract and join text from given chunk lines."""
    r = " ".join(chunk[l].text for l in lines if 0 <= l < len(chunk)).strip()
    return r


def compute_best_shift(
    chunk: Chunk, lines: list[int], target_text: str, max_shift: int
) -> tuple[int, float]:
    """Return the shift within [-max_shift, max_shift] giving best similarity."""
    best_shift, best_score = 0, -1.0
    target_fp = text_fingerprint(target_text)

    for shift in range(-max_shift, max_shift + 1):
        shifted_lines = [l + shift for l in lines]
        chunk_fp = text_fingerprint(chunk_text_for_lines(chunk, shifted_lines))
        similarity = fuzz.partial_ratio(target_fp, chunk_fp) / 100.0
        if similarity > best_score:
            best_shift, best_score = shift, similarity

    return best_shift, best_score


def extract_item_text(item) -> str:
    """Get all textual content from an item."""

    assert isinstance(
        item, (Entry, Title, Text, EntryCompact, TitleCompact, TextCompact)
    ), f"Unsupported item type: {type(item).__name__}"

    return _simple_string_repr(item, exclude_fields=["cat", "lines"])


def new_item_with_lines(item, lines):
    """Return a copy of item with updated line indices."""
    if not hasattr(item, "lines"):
        raise ValueError(f"Cannot set lines for {type(item).__name__} without 'lines'")
    return type(item)(**item.model_dump(exclude={"lines"}), lines=lines)


def fix_lines_alignment(
    chunk: Chunk,
    response: Structured,
    max_lookahead: int = 3,
    detect_threshold: float = 0.5,
    fix_threshold: float = 0.7,
) -> Structured:
    """Detect and fix small misalignments in line indices."""
    new_items = []
    line_offset = 0

    for item in response.items:
        if not hasattr(item, "lines"):
            raise ValueError(
                f"Cannot fix alignment for {type(item).__name__} without 'lines'"
            )

        # Apply accumulated offset
        item = new_item_with_lines(item, [l + line_offset for l in item.lines])

        # Compare chunk vs extracted text
        chunk_text = chunk_text_for_lines(chunk, item.lines)
        extracted_text = extract_item_text(item)
        similarity = (
            fuzz.ratio(text_fingerprint(chunk_text), text_fingerprint(extracted_text))
            / 100.0
        )
        if similarity < detect_threshold:
            logger.info(f"Misalignment detected: `{extracted_text}` ↮ `{chunk_text}`")

            best_shift, best_similarity = compute_best_shift(
                chunk, item.lines, extracted_text, max_lookahead
            )

            if best_shift != 0 and best_similarity >= fix_threshold:
                sign = "+" if best_shift > 0 else ""
                logger.info(
                    f"Applying {sign}{best_shift} line offset at {item.lines[0]} (sim={best_similarity:.2f})"
                )
                item = new_item_with_lines(item, [l + best_shift for l in item.lines])
                line_offset += best_shift
            elif logger:
                logger.info(
                    f"No suitable offset found (best similarity={best_similarity:.2f})"
                )

        new_items.append(item)

    return Structured(items=new_items)


def _simple_string_repr(
    obj: Any, exclude_fields: list[str] | None = None, exclude_none: bool = True
) -> str:
    exclude = set(exclude_fields or [])

    # CAS 1 : C'est un modèle Pydantic
    if isinstance(obj, BaseModel):
        parts = []
        # On itère directement sur l'objet pour garder les types BaseModel intacts
        for name, value in obj:
            if name in exclude or (value is None and exclude_none):
                continue
            parts.append(_simple_string_repr(value, exclude_fields, exclude_none))
        return " ".join(p for p in parts if p.strip())

    # CAS 2 : C'est une liste ou un tuple (itérable mais pas une string)
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        parts = [
            _simple_string_repr(item, exclude_fields, exclude_none) for item in obj
        ]
        return " ".join(p for p in parts if p.strip())

    # CAS 3 : Valeur simple (str, int, etc.)
    return str(obj) if obj is not None else ""


# def test_fix_lines_alignment_realistic():
#     """Test fix_lines_alignment using the real Pydantic models."""

#     # 1️⃣ Créer un chunk simulé avec des lignes OCR
#     chunk = Chunk(
#         root=[
#             ChunkLine(page=1, line=0, text="A, rap., Louvre, "),
#             ChunkLine(page=1, line=1, text="40"),
#             ChunkLine(page=1, line=2, text="ODO"),
#             ChunkLine(page=1, line=3, text="B, per., Paris"),
#             ChunkLine(
#                 page=1, line=4, text="C, koe., 75001 <br> D, xyz., Rouen sur Rrr"
#             ),
#             ChunkLine(page=2, line=0, text="TITRE DE LA PAGE D'APRES"),
#         ]
#     )

#     # 2️⃣ Créer une structure avec des items mal alignés
#     response = Structured(
#         items=[
#             Entry(
#                 cat="ent",
#                 name="A",
#                 activity="rap.",
#                 addresses=[Address(label="Louvre", number="40", complement=None)],
#                 additional_info=[],
#                 lines=[1],
#             ),
#             Text(
#                 cat="txt",
#                 text="ODO",
#                 lines=[2],
#             ),
#             Entry(
#                 cat="ent",
#                 name="B",
#                 activity="per.",
#                 addresses=[Address(label="Paris", number=None, complement=None)],
#                 additional_info=[],
#                 lines=[2],
#             ),
#             Entry(
#                 cat="ent",
#                 name="C",
#                 activity="koe.",
#                 addresses=[
#                     Address(label="75001", number=None, complement=None),
#                 ],
#                 additional_info=[],
#                 lines=[4],
#             ),
#             Entry(
#                 cat="ent",
#                 name="D",
#                 activity="xyz.",
#                 addresses=[
#                     Address(label="Rouen sur Rrr", number=None, complement=None),
#                 ],
#                 additional_info=[],
#                 lines=[5],
#             ),
#             Title(
#                 cat="title",
#                 text="TITRE DE LA PAGE D'APRES",
#                 lines=[5],
#             ),
#         ]
#     )

#     # 3️⃣ Appliquer la correction
#     fixed_response = fix_lines_alignment(chunk, response, max_lookahead=2)

#     # 4️⃣ Afficher le résultat
#     print("Avant correction:")
#     for item in response.items:
#         print(
#             f"{item}",
#         )

#     print("\nAprès correction:")
#     for item in fixed_response.items:
#         print(f"{item}")


# # Exécuter le test
# test_fix_lines_alignment_realistic()
