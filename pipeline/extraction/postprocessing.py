from __future__ import annotations
from typing import Any, Iterable
from thefuzz import fuzz
from pydantic import BaseModel
from pipeline.chunking.schemas import Chunk, ChunkLine
from .schemas import (
    Address,
    EntryCompact,
    Structured,
    Entry,
    TextCompact,
    Title,
    Text,
    TitleCompact,
)
from pipeline.logging import logger


def fingerprint_unordered(text: str) -> str:
    filtered = "".join(c.lower() for c in text if c.isalnum())
    return "".join(sorted(filtered))


def fingerprint_ordered(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalnum())


def ratio_unordered(a: str, b: str) -> float:
    return (
        fuzz.ratio(
            fingerprint_unordered(a),
            fingerprint_unordered(b),
        )
        / 100.0
    )


def partial_directional_ratio(a: str, b: str) -> float:
    fa = fingerprint_ordered(a)
    fb = fingerprint_ordered(b)
    if len(fa) > len(fb):
        return 0.0
    return fuzz.partial_ratio(fa, fb) / 100.0


def assess_alignment(
    chunk: Chunk,
    lines: list[int],
    entity_text: str,
    *,
    global_threshold: float,
    inclusion_threshold: float,
) -> tuple[str, float]:
    """
    Returns:
        status ∈ {"OK", "FUSION", "MISMATCH"}
        score  ∈ [0,1]
    """

    chunk_text = chunk_text_for_lines(chunk, lines)

    # 1. Test global (ordre-insensible)
    r = ratio_unordered(entity_text, chunk_text)
    if r >= global_threshold:
        return "OK", r

    # 2. Test inclusion (ordre-sensible, directionnel)
    r_prime = partial_directional_ratio(entity_text, chunk_text)
    if r_prime >= inclusion_threshold:
        return "FUSION", r_prime

    return "MISMATCH", max(r, r_prime)


def find_best_local_match(
    chunk: Chunk,
    lines: list[int],
    entity_text: str,
    max_shift: int,
    *,
    global_threshold: float,
    inclusion_threshold: float,
) -> tuple[int, float]:
    best_shift = 0
    best_score = 0.0

    for shift in range(-max_shift, max_shift + 1):
        shifted = [l + shift for l in lines]
        status, score = assess_alignment(
            chunk,
            shifted,
            entity_text,
            global_threshold=global_threshold,
            inclusion_threshold=inclusion_threshold,
        )
        if score > best_score:
            best_shift = shift
            best_score = score

    return best_shift, best_score


def chunk_text_for_lines(chunk: Chunk, lines: list[int]) -> str:
    """Extract and join text from given chunk lines."""
    r = " ".join(chunk[l].text for l in lines if 0 <= l < len(chunk)).strip()
    return r


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
    global_threshold: float = 0.7,
    inclusion_threshold: float = 0.8,
) -> Structured:
    """
    Recalage local des indices de lignes pour les items extraits.
    """

    new_items = []
    line_offset = 0
    max_line_index = len(chunk) - 1  # dernière ligne valide

    for item in response.items:
        if not hasattr(item, "lines"):
            raise ValueError(
                f"Cannot fix alignment for {type(item).__name__} without 'lines'"
            )

        # --- 1️⃣ Sanitation : ignorer les items hors chunk ---
        if any(l < 0 or l > max_line_index for l in item.lines):
            logger.warning(
                f"Skipping alignment fix for item with out-of-scope lines: {item}"
            )
            new_items.append(item)
            continue

        # --- 2️⃣ Appliquer l'offset cumulé existant ---
        lines = [l + line_offset for l in item.lines]
        item = new_item_with_lines(item, lines)

        entity_text = extract_item_text(item)

        # --- 3️⃣ Vérifier l'alignement actuel ---
        status, score = assess_alignment(
            chunk,
            lines,
            entity_text,
            global_threshold=global_threshold,
            inclusion_threshold=inclusion_threshold,
        )

        if status in {"OK", "FUSION"}:
            new_items.append(item)
            continue

        # --- 4️⃣ MISMATCH → recherche locale ---
        best_shift, best_score = find_best_local_match(
            chunk,
            lines,
            entity_text,
            max_lookahead,
            global_threshold=global_threshold,
            inclusion_threshold=inclusion_threshold,
        )

        if best_score > score and best_shift != 0:
            logger.info(
                f"Fix alignment at line {lines[0]}: shift {best_shift} "
                f"({score:.2f} → {best_score:.2f})"
            )
            # appliquer le shift, cardinalité conservée
            item = new_item_with_lines(item, [l + best_shift for l in lines])
            line_offset += best_shift

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

# def test_fix_lines_alignment_scope_only():
#     """Test fix_lines_alignment: only verifies local line alignment, not content correctness."""

#     # --- 1️⃣ Chunk simulé ---
#     chunk = Chunk(
#         root=[
#             ChunkLine(page=1, line=0, text="ANNUAIRE DES PROFESSIONNELS"),
#             ChunkLine(page=1, line=1, text="A, rap., Louvre"),
#             ChunkLine(page=1, line=2, text="40"),
#             ChunkLine(page=1, line=3, text="ODO"),
#             ChunkLine(page=1, line=4, text="B, per., Paris"),
#             ChunkLine(page=1, line=5, text="C, koe., 75001 <br> D, xyz., Rouen sur Rrr"),
#             ChunkLine(page=1, line=6, text="NOTE: Informations non contractuelles"),
#             ChunkLine(page=2, line=0, text="E, foo., Lyon"),
#             ChunkLine(page=2, line=1, text="42"),
#             ChunkLine(page=2, line=2, text="TITRE DE LA PAGE SUIVANTE"),
#         ]
#     )

#     # --- 2️⃣ Items extraits par le LLM (avec bruit / hallucinations) ---
#     response = Structured(
#         items=[
#             Title(cat="title", text="ANNUAIRE DES PROFESSIONNELS", lines=[0]),
#             Entry(cat="ent", name="A", activity="rap.", addresses=[Address(label="Louvre", number="40", complement=None)], additional_info=[], lines=[1,2]),
#             Text(cat="txt", text="ODO", lines=[3,4]),
#             Entry(cat="ent", name="B", activity="per.", addresses=[Address(label="Paris", number=None, complement=None)], additional_info=[], lines=[2,3]),
#             Entry(cat="ent", name="C", activity="koe.", addresses=[Address(label="75001", number=None, complement=None)], additional_info=[], lines=[5,6]),
#             Entry(cat="ent", name="D", activity="xyz.", addresses=[Address(label="Rouen sur Rrr", number=None, complement=None)], additional_info=[], lines=[10,11]),  # hors chunk
#             Text(cat="txt", text="Lorem ipsum dolor sit amet", lines=[7]),
#             Entry(cat="ent", name="E", activity="foo.", addresses=[Address(label="Lyon", number='42', complement=None)], additional_info=[], lines=[7]),
#             Title(cat="title", text="TITRE DE LA PAGE SUIVANTE", lines=[100]),  # hors chunk
#         ]
#     )

#     # --- 3️⃣ Appliquer la correction ---
#     fixed = fix_lines_alignment(chunk, response, max_lookahead=2)

#     max_line_index = len(chunk) - 1

#     # --- 4️⃣ Vérifications des invariants ---
#     for before, after in zip(response.items, fixed.items):
#         # 4a. Cardinalité conservée
#         assert len(after.lines) == len(before.lines), f"Cardinality changed for {before}"

#         # 4b. Décalage uniforme pour les items recalés
#         offsets = [a - b for a, b in zip(after.lines, before.lines)]
#         if any(offsets):
#             # tous les décalages doivent être identiques
#             assert len(set(offsets)) == 1, f"Non-uniform line shift for {before}"

#         # 4c. Items hors chunk restent inchangés
#         if any(l < 0 or l > max_line_index for l in before.lines):
#             assert after.lines == before.lines, f"Lines outside chunk modified for {before}"

#         # 4d. Items valides doivent rester dans le chunk
#         else:
#             assert all(0 <= l <= max_line_index for l in after.lines), f"Shifted lines out of chunk for {before}"

#     # --- 5️⃣ Affichage pour debug (optionnel) ---
#     print("=== Items after fixing (scope-only) ===")
#     for item in fixed.items:
#         print(item)


# # Exécuter le test
# test_fix_lines_alignment_scope_only()
