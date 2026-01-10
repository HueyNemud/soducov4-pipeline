from __future__ import annotations
from typing import Any, Iterable, Tuple, List
from thefuzz import fuzz
from pydantic import BaseModel

from pipeline.chunking.schemas import Chunk
from pipeline.logging import logger
from .schemas import (
    EntryCompact,
    Structured,
    Entry,
    TextCompact,
    Title,
    Text,
    TitleCompact,
)


def get_fuzzy_fingerprint(text: str, sort_chars: bool = False) -> str:
    """
    Generates a normalized alphanumeric string for fuzzy comparison.

    Args:
        text: Input string to normalize.
        sort_chars: If True, returns characters in alphabetical order (unordered comparison).
    """
    chars = [char.lower() for char in text if char.isalnum()]
    if sort_chars:
        chars.sort()
    return "".join(chars)


def calculate_unordered_ratio(text_a: str, text_b: str) -> float:
    """Calculates similarity ratio ignoring character order."""
    return (
        fuzz.ratio(
            get_fuzzy_fingerprint(text_a, sort_chars=True),
            get_fuzzy_fingerprint(text_b, sort_chars=True),
        )
        / 100.0
    )


def calculate_directional_partial_ratio(source: str, target: str) -> float:
    """
    Calculates similarity assuming 'source' should be contained within 'target'.
    Returns 0.0 if source is longer than target.
    """
    fp_source = get_fuzzy_fingerprint(source)
    fp_target = get_fuzzy_fingerprint(target)

    if len(fp_source) > len(fp_target):
        return 0.0
    return fuzz.partial_ratio(fp_source, fp_target) / 100.0


def assess_alignment(
    chunk: Chunk,
    line_indices: List[int],
    entity_text: str,
    *,
    global_threshold: float,
    inclusion_threshold: float,
) -> Tuple[str, float]:
    """
    Evaluates how well an extracted entity matches the text at specific chunk lines.

    Returns:
        A tuple of (status, score) where status is 'OK', 'FUSION', or 'MISMATCH'.
    """
    candidate_text = get_text_from_indices(chunk, line_indices)

    # 1. Global unordered check (Identity)
    score_global = calculate_unordered_ratio(entity_text, candidate_text)
    if score_global >= global_threshold:
        return "OK", score_global

    # 2. Sequential inclusion check (Partial match)
    score_partial = calculate_directional_partial_ratio(entity_text, candidate_text)
    if score_partial >= inclusion_threshold:
        return "FUSION", score_partial

    return "MISMATCH", max(score_global, score_partial)


def find_best_local_shift(
    chunk: Chunk,
    line_indices: List[int],
    entity_text: str,
    max_search_range: int,
    *,
    global_threshold: float,
    inclusion_threshold: float,
) -> Tuple[int, float]:
    """
    Searches for the best line offset within a local range to improve alignment.
    """
    best_shift = 0
    best_score = 0.0

    for shift in range(-max_search_range, max_search_range + 1):
        shifted_indices = [idx + shift for idx in line_indices]
        _, score = assess_alignment(
            chunk,
            shifted_indices,
            entity_text,
            global_threshold=global_threshold,
            inclusion_threshold=inclusion_threshold,
        )
        if score > best_score:
            best_shift = shift
            best_score = score

    return best_shift, best_score


def get_text_from_indices(chunk: Chunk, indices: List[int]) -> str:
    """Concatenates text from a chunk based on a list of line indices."""
    valid_lines = [chunk[i].text for i in indices if 0 <= i < len(chunk)]
    return " ".join(valid_lines).strip()


def extract_item_full_text(item: Any) -> str:
    """Serializes all textual fields of a Pydantic item into a single string for comparison."""
    valid_types = (Entry, Title, Text, EntryCompact, TitleCompact, TextCompact)
    assert isinstance(item, valid_types), f"Unsupported type: {type(item).__name__}"
    return _recursive_string_repr(item, exclude_fields=["cat", "lines"])


def update_item_lines(item: Any, new_indices: List[int]) -> Any:
    """Creates a copy of the item with updated line indices."""
    if not hasattr(item, "lines"):
        raise ValueError(f"Object {type(item).__name__} lacks a 'lines' attribute.")
    return type(item)(**item.model_dump(exclude={"lines"}), lines=new_indices)


def fix_lines_alignment(
    chunk: Chunk,
    structured_data: Structured,
    max_lookahead: int = 3,
    global_threshold: float = 0.7,
    inclusion_threshold: float = 0.8,
) -> Structured:
    """
    Corrects line index drifting in extracted items by comparing LLM output with original chunk text.

    This adjusts for cases where the LLM correctly extracts content but provides slightly
    offset line numbers. It maintains a cumulative offset to guide subsequent items.
    """
    aligned_items = []
    cumulative_offset = 0
    last_valid_index = len(chunk) - 1

    for item in structured_data.items:
        # Safety check: bypass items with indices entirely outside the current chunk
        if any(idx < 0 or idx > last_valid_index for idx in item.lines):
            logger.warning(f"Skipping alignment for out-of-scope item: {item}")
            aligned_items.append(item)
            continue

        # Step 1: Apply current known drift
        shifted_indices = [idx + cumulative_offset for idx in item.lines]
        item = update_item_lines(item, shifted_indices)
        entity_content = extract_item_full_text(item)

        # Step 2: Evaluate current alignment
        status, current_score = assess_alignment(
            chunk,
            shifted_indices,
            entity_content,
            global_threshold=global_threshold,
            inclusion_threshold=inclusion_threshold,
        )

        if status in {"OK", "FUSION"}:
            aligned_items.append(item)
            continue

        # Step 3: Attempt local re-alignment on mismatch
        best_shift, best_score = find_best_local_shift(
            chunk,
            shifted_indices,
            entity_content,
            max_lookahead,
            global_threshold=global_threshold,
            inclusion_threshold=inclusion_threshold,
        )

        if best_score > current_score and best_shift != 0:
            logger.info(
                f"Realignment triggered at line {shifted_indices[0]}: "
                f"shift {best_shift} (Score: {current_score:.2f} -> {best_score:.2f})"
            )
            final_indices = [idx + best_shift for idx in shifted_indices]
            item = update_item_lines(item, final_indices)
            cumulative_offset += best_shift

        aligned_items.append(item)

    return Structured(items=aligned_items)


def _recursive_string_repr(
    obj: Any, exclude_fields: List[str] | None = None, exclude_none: bool = True
) -> str:
    """
    Recursively extracts string values from complex objects (Pydantic, Iterables, Primitives).
    Used to rebuild the 'raw' text an LLM saw from its structured output.
    """
    exclude = set(exclude_fields or [])

    if isinstance(obj, BaseModel):
        field_values = [
            _recursive_string_repr(val, exclude_fields, exclude_none)
            for name, val in obj
            if name not in exclude and not (val is None and exclude_none)
        ]
        return " ".join(v for v in field_values if v.strip())

    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        item_values = [
            _recursive_string_repr(item, exclude_fields, exclude_none) for item in obj
        ]
        return " ".join(v for v in item_values if v.strip())

    return str(obj) if obj is not None else ""
