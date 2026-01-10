"""
Post-Processing Engine for Parisian Directory (Annuaire) Reconstruction.

This module specializes in structured data recovery from high-density,
multi-column historical directories. It acts as a specialized refinement
layer following Surya OCR detection to transform raw coordinates into
ordered, context-aware business listings.

Directory-Specific Logic:
-------------------------
1. Leading-Edge Anchor Model (Listing Affinity):
   In dense directories, lines often overflow their bounding boxes or sit
   extremely close to adjacent entries. The 'Leading-Edge' model prioritizes
   the start of the entry (Name/Profession) to anchor text lines to their
   respective layout blocks, ensuring that indented addresses or phone numbers
   are correctly grouped with their parent listing.

Pipeline Integration:
---------------------
- Input: Receives raw `OCRDocument` results from Surya OCR.
- Processing: Filters low-confidence noise, assigns lines to layout
  segments, and reconstructs the vertical flow within detected columns.
- Output: A structured `OCRDocument` ready for downstream tasks.

Constraints:
------------
Requires axis-aligned input. While robust to the dense nature of
directories, highly skewed scans or curved pages (spine distortion)
should be pre-corrected (deskewed) for optimal column segmentation.
"""

from enum import Enum
from typing import Literal, Annotated
import numpy as np
from numpy.typing import NDArray
from pipeline.ocr.schemas import OCRResult, OCRDocument
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import os
from pipeline.logging import logger
from pydantic import Field, dataclasses


# =======================
# Type Aliases
# =======================

CoordArray = NDArray[np.int32]
BBox = Annotated[CoordArray, Literal[4]]
BBoxArray = Annotated[CoordArray, Literal["N", 4]]


class Direction(Enum):
    ABOVE = "above"
    BELOW = "below"


class ReadingDirection(Enum):
    LTR = "left-to-right"
    RTL = "right-to-left"


# =======================
# Main logic
# =======================


def box_affinity(
    layouts: BBoxArray,
    lines: BBoxArray,
    decay_factor: float = 2.0,
    anchor_offset: float = 0.20,
    direction: ReadingDirection = ReadingDirection.LTR,
) -> NDArray[np.float64]:
    """
    Compute spatial affinity between layout blocks (N) and text lines (M).

    Logic & Purpose:
    ----------------
    This function measures how well a text line "fits" into a layout block.
    Instead of a simple overlap, it uses a "Leading Anchor" model: we pick a point
    on the line (the anchor) and calculate its distance to the nearest edge of the
    layout box. An exponential decay then converts this distance into a score
    between 1.0 (inside or touching) and 0.0 (far away).

    The vertical decay is normalized by the median line height, making the
    matching robust to different font sizes and document scales.

    Parameters:
    -----------
    layouts : BBoxArray of shape (N, 4) -> [x1, y1, x2, y2]
    lines : BBoxArray of shape (M, 4) -> [x1, y1, x2, y2]
    decay_factor : Controls how fast the affinity drops with distance.
    anchor_offset : Horizontal position of the anchor (0.0 = edge, 0.5 = center).
    direction : Reading direction to determine which edge is the 'leading' one.

    Returns:
    --------
    NDArray[np.float64] : Affinity matrix of shape (N, M).
    """
    # Unpack coordinates
    l_x1, l_y1, l_x2, l_y2 = layouts.T
    t_x1, t_y1, t_x2, t_y2 = lines.T

    # 1. Define line anchor points (Horizontal: leading edge offset, Vertical: center)
    line_widths = t_x2 - t_x1
    if direction == ReadingDirection.LTR:
        anchors_x = t_x1 + (anchor_offset * line_widths)
    else:
        anchors_x = t_x2 - (anchor_offset * line_widths)

    anchors_y = (t_y1 + t_y2) / 2.0

    # 2. Calculate "Tunnel" distances (distance from anchor to layout box boundaries)
    # The result of [:, None] - vector is an (N, M) matrix via broadcasting
    dist_y = np.maximum(0, l_y1[:, None] - anchors_y) + np.maximum(
        0, anchors_y - l_y2[:, None]
    )
    dist_x = np.maximum(0, l_x1[:, None] - anchors_x) + np.maximum(
        0, anchors_x - l_x2[:, None]
    )

    # 3. Normalize decay scale using the document's median line height
    line_heights = t_y2 - t_y1
    reference_scale = np.median(line_heights) if line_heights.size > 0 else 1.0
    sigma = reference_scale * decay_factor

    # 4. Final affinity score: Exponential decay of the Manhattan distance
    total_distance = dist_x + dist_y
    return np.exp(-total_distance / sigma)


def compute_local_reading_order(
    assignment: NDArray[np.bool_],
    lines: BBoxArray,
    direction: ReadingDirection = ReadingDirection.LTR,
    n_bins: int = 400,
) -> NDArray[np.int32]:
    """
    Establish a logical reading order for text lines within their assigned layout blocks.

    Logic & Purpose:
    -----------
    This function handles multi-column layouts within a single block. It works in 3 phases:
    1. **Column Detection**: It projects the horizontal density of text lines onto an
       histogram (bins). High-density areas indicate text columns, while gaps indicate
       gutters.
    2. **Line Attribution**: Each text line is mapped to the nearest detected column
       based on its 'leading edge' (left for LTR, right for RTL).
    3. **Sorting**: Lines are sorted primarily by their column index and secondarily
       by their vertical position (top-to-bottom).

    This approach prevents the "zigzag" effect where a reader would jump between
    columns on the same horizontal level.

    **Drawbacks & Constraints**
    -----------
    - Spatial Alignment: Assumes consistent alignment (e.g., LTR/left-aligned).
    Right-aligned or centered text in a LTR block may cause false column splits.
    - Justification Gaps: Heavily justified text with wide inter-word spacing
    can be mistaken for gutters, leading to fragmented columns.
    - Gutter Sensitivity: The 'n_bins' parameter is a trade-off; low values
    may merge thin columns, while high values may over-segment text.

    Parameters:
    -----------
    assignment : Boolean matrix (M, N) mapping layout blocks to text lines.
    lines : BBoxArray of shape (N, 4) containing [x1, y1, x2, y2].
    direction : Reading direction (LTR or RTL).
    n_bins : Resolution of the horizontal projection profile.
    """
    num_layouts, num_lines = assignment.shape
    reading_order = np.zeros((num_layouts, num_lines), dtype=np.int32)

    # Pre-calculate coordinates and vertical centers
    x_min, y_min, x_max, y_max = lines.T
    y_centers = (y_min + y_max) / 2.0
    leading_edges = x_min if direction == ReadingDirection.LTR else x_max

    for layout_id in range(num_layouts):
        # Extract indices of lines belonging to this layout block
        line_indices = np.flatnonzero(assignment[layout_id])
        if line_indices.size == 0:
            continue

        # --- Phase 1: Horizontal Density Projection ---
        block_x_min, block_x_max = x_min[line_indices].min(), x_max[line_indices].max()
        block_width = max(block_x_max - block_x_min, 1.0)
        bin_size = block_width / n_bins

        # Calculate start/end of the 'mass' of each line for column detection
        # We use the first half of the line to emphasize the column start
        if direction == ReadingDirection.LTR:
            starts, ends = (
                x_min[line_indices],
                (x_min[line_indices] + x_max[line_indices]) / 2,
            )
        else:
            starts, ends = (x_min[line_indices] + x_max[line_indices]) / 2, x_max[
                line_indices
            ]

        # Map coordinates to bin indices and fill density histogram
        density = np.zeros(n_bins, dtype=np.int32)
        start_bins = np.clip(
            ((starts - block_x_min) / bin_size).astype(int), 0, n_bins - 1
        )
        end_bins = np.clip(((ends - block_x_min) / bin_size).astype(int), 0, n_bins - 1)

        for s, e in zip(start_bins, end_bins):
            density[s : e + 1] += 1

        # --- Phase 2: Column Gutter Detection ---
        # Identify occupied bins (ignoring noise/outliers via percentile)
        threshold = (
            np.percentile(density[density > 0], 10) if np.any(density > 0) else 0
        )
        is_occupied = density > threshold

        # Detect transitions to find column boundaries
        transitions = np.diff(is_occupied.astype(int), prepend=0, append=0)
        col_starts = np.where(transitions == 1)[0]
        col_ends = np.where(transitions == -1)[0]

        if col_starts.size == 0:
            column_spans = np.array([[block_x_min, block_x_max]])
        else:
            column_spans = np.column_stack(
                [block_x_min + col_starts * bin_size, block_x_min + col_ends * bin_size]
            )

        # --- Phase 3: Column Assignment & Final Sort ---
        # Measure distance from line leading edges to each detected column span
        anchors = leading_edges[line_indices][:, None]
        dist_to_columns = np.maximum(
            0, np.maximum(column_spans[:, 0] - anchors, anchors - column_spans[:, 1])
        )
        column_assignments = np.argmin(dist_to_columns, axis=1)

        # Sort: Column first, then Y-center
        # For RTL, we invert column priority to read from right to left
        column_priority = (
            column_assignments
            if direction == ReadingDirection.LTR
            else -column_assignments
        )
        sort_indices = np.lexsort((y_centers[line_indices], column_priority))

        # Map back to global line indices (1-based ordering)
        reading_order[layout_id, line_indices[sort_indices]] = np.arange(
            1, line_indices.size + 1
        )

    return reading_order


# =======================
# Surya Post-Processor
# =======================


class SuryaLayoutPostProcessor:
    """
    Orchestrates the conversion of raw OCR detections into a structured document.

    Logic & Purpose:
    ----------------
    This processor acts as the "brain" that connects layout analysis with text
    recognition. Its primary goal is to solve the assignment problem: "Which
    layout block does this text line belong to, and in what order should it be read?"

    Workflow:
    1. **Filtering**: Removes low-confidence text detections to reduce noise.
    2. **Spatial Affinity**: Maps text lines to layout blocks using a
       multi-dimensional distance model (see `box_affinity`).
    3. **Local Ordering**: Detects columns and sorts lines within each block
       (see `compute_local_reading_order`).
    4. **Global Assembly**: Reconstructs a sequential OCRDocument where lines
       are indexed by their final reading order and layout parent.

    Drawbacks:
    ----------
    - Rotated Documents: Assumes axis-aligned bounding boxes; skewed or rotated
      pages will significantly degrade assignment accuracy.
    - Strict Filtering: A high confidence threshold (0.8) may discard valid text
      in poor-quality scans.
    """

    @dataclasses.dataclass
    class Parameters:
        """
        Configuration parameters for the SuryaPostProcessor.
        """

        # Box affinity parameters
        decay_factor: float = Field(
            default=2.0,
            ge=0.1,
            description="Controls how fast the affinity drops as the line moves away from the layout box.",
        )
        anchor_offset: float = Field(
            default=0.20,
            ge=0,
            le=1.0,
            description="Leading edge position (0.0 = edge, 0.5 = center). Default is 20% from edge.",
        )

        # Column detection parameters
        n_bins: int = Field(
            default=400,
            gt=0,
            description="Number of bins for horizontal projection histogram."
            "Higher values increase resolution but may over-segment.",
        )
        occupancy_percentile: float = Field(
            default=10.0,
            ge=0,
            le=100,
            description="Percentile threshold to determine occupied bins in column detection.",
        )

        # Filtering
        confidence_threshold: float = Field(
            default=0.8,
            ge=0,
            le=1.0,
            description="Minimum confidence for text lines to be considered reliable."
            "Text lines below this threshold are ignored.",
        )

        # Debugging
        debug_plots: bool = Field(
            default=False,
            description="Enable generation of debug plots for visual inspection."
            "Warning: will dramatically increase processing time.",
        )
        debug_dir: str = Field(
            default="./debug_plots", description="Directory to save debug plots."
        )

        # Global
        reading_direction: ReadingDirection = ReadingDirection.LTR

    def process_ocr_output(
        self, document: OCRDocument, params: Parameters
    ) -> OCRDocument:
        """
        Refines layout and OCR results into a structured, ordered document.
        """

        processed_pages = OCRDocument(layout=[], ocr=[])
        # Track affinity scores across pages for visualization
        debug_scores = {}

        for page_idx, (layout, ocr) in enumerate(zip(document.layout, document.ocr)):
            logger.debug(
                f"Processing Page {page_idx}: {len(layout.bboxes)} blocks, {len(ocr.text_lines)} lines."
            )

            # 1. Prepare coordinate matrices
            layout_boxes = np.array(
                [quad.bbox for quad in layout.bboxes], dtype=np.int32
            )
            all_text_boxes = np.array(
                [line.bbox for line in ocr.text_lines], dtype=np.int32
            )

            # 2. Filter lines by confidence

            is_reliable = np.array(
                [
                    (
                        line.confidence
                        if line.confidence is not None
                        else 1.0 >= params.confidence_threshold
                    )
                    for line in ocr.text_lines
                ],
                dtype=bool,
            )

            if not np.any(is_reliable) or layout_boxes.size == 0:
                logger.warning(
                    f"Skipping Page {page_idx}: No valid content after filtering."
                )
                processed_pages.layout.append(layout)
                processed_pages.ocr.append(ocr)
                continue

            # Synchronize reliable data
            reliable_boxes = all_text_boxes[is_reliable]
            reliable_objects = [
                line for i, line in enumerate(ocr.text_lines) if is_reliable[i]
            ]

            # 3. Spatial Assignment
            # Compute affinity and assign each line to the highest scoring layout block
            affinity_matrix = box_affinity(
                layout_boxes,
                reliable_boxes,
                decay_factor=params.decay_factor,
                anchor_offset=params.anchor_offset,
                direction=params.reading_direction,
            )
            best_layout_ids = np.argmax(affinity_matrix, axis=0)

            # Create a sparse assignment mask (num_layouts, num_reliable_lines)
            assignment_mask = np.zeros(
                (layout_boxes.shape[0], reliable_boxes.shape[0]), dtype=bool
            )
            assignment_mask[best_layout_ids, np.arange(reliable_boxes.shape[0])] = True

            # 4. Local Sequencing
            # Determine the reading order (columns + Y) within each block
            reading_indices = compute_local_reading_order(
                assignment_mask,
                reliable_boxes,
                direction=params.reading_direction,
                n_bins=params.n_bins,
            )

            # 5. Document Reconstruction
            ordered_ocr = OCRResult(text_lines=[], image_bbox=ocr.image_bbox)
            global_counter = 1

            for layout_idx in range(layout_boxes.shape[0]):
                # Get indices of lines assigned to this specific block
                assigned_idxs = np.flatnonzero(assignment_mask[layout_idx])

                # Sort lines by their pre-calculated local reading order
                sort_priority = np.argsort(reading_indices[layout_idx, assigned_idxs])

                for idx in assigned_idxs[sort_priority]:
                    line = reliable_objects[idx]
                    line.reading_order_ix = global_counter
                    line.layout_ix = layout_idx

                    ordered_ocr.text_lines.append(line)

                    if params.debug_plots:
                        debug_scores[id(line)] = affinity_matrix[:, idx]

                    global_counter += 1

            processed_pages.layout.append(layout)
            processed_pages.ocr.append(ordered_ocr)

        if params.debug_plots:
            self._export_debug(
                processed_pages, debug_scores, output_dir=params.debug_dir
            )

        return processed_pages

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def _export_debug(
        self, ocr_doc: OCRDocument, line_probs_dict: dict, output_dir: str
    ):
        """Encapsulated debug engine: generates diagnostic plots."""
        output_dir = output_dir or "./debug_plots"
        os.makedirs(output_dir, exist_ok=True)
        cmaps = {"layout": plt.get_cmap("tab20"), "order": plt.get_cmap("plasma")}

        def draw_poly(ax, polygon, text=None, text_color="black", **kwargs):
            poly = np.asarray(polygon)
            ax.add_patch(MplPolygon(poly, closed=True, **kwargs))
            if text is not None:
                cx, cy = poly.mean(axis=0)
                ax.text(
                    cx,
                    cy,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=text_color,
                )

        def get_style(line, cmap):
            # Orphan or invalid reading order
            if line.layout_ix is None or (line.reading_order_ix or -1) < 0:
                return {
                    "facecolor": "black",
                    "edgecolor": "red",
                    "alpha": 1.0,
                    "hatch": "////",
                    "text_color": "red",
                }

            prob_vector = line_probs_dict.get(id(line))
            alpha = 1.0
            # Safety check: ensure layout_ix is within the probability vector bounds
            if prob_vector is not None and 0 <= line.layout_ix < len(prob_vector):
                alpha = max(0.1, float(prob_vector[line.layout_ix]))

            color = cmap(line.layout_ix % cmap.N)
            return {
                "facecolor": color,
                "edgecolor": color,
                "alpha": alpha,
                "text_color": "white",
            }

        for p_idx, (layout, ocr) in enumerate(zip(ocr_doc.layout, ocr_doc.ocr)):
            bbox = layout.image_bbox
            configs = [
                (
                    "layout",
                    lambda ax, layout=layout, o=ocr: [
                        draw_poly(
                            ax,
                            quad.polygon,
                            text=str(i),
                            facecolor=cmaps["layout"](i % cmaps["layout"].N),
                            edgecolor="black",
                            alpha=0.3,
                        )
                        for i, quad in enumerate(layout.bboxes)
                    ],
                ),
                (
                    "assignment",
                    lambda ax, layout=layout, o=ocr: [
                        draw_poly(
                            ax,
                            line.polygon,
                            text=str(line.layout_ix or "?"),
                            **get_style(line, cmaps["layout"]),
                        )
                        for line in o.text_lines
                    ],
                ),
                (
                    "reading_order",
                    lambda ax, layout=layout, o=ocr: _draw_reading_flow(
                        ax, o, cmaps["order"], draw_poly
                    ),
                ),
            ]

            for suffix, draw_func in configs:
                fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
                draw_func(ax)
                ax.set_xlim(bbox[0], bbox[2])
                ax.set_ylim(bbox[1], bbox[3])
                ax.invert_yaxis()
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_title(
                    f"Page {p_idx} - {suffix.replace('_', ' ').title()}", fontsize=20
                )

                fig.savefig(
                    os.path.join(output_dir, f"page_{p_idx}_{suffix}.png"),
                    bbox_inches="tight",
                )
                fig.clear()  # Memory safety
                plt.close(fig)


def _draw_reading_flow(ax, ocr, cmap, draw_func):
    """Separate logic for the gradient flow to keep the lambda clean."""
    orders = [
        line.reading_order_ix
        for line in ocr.text_lines
        if (line.reading_order_ix or 0) > 0
    ]
    max_o = max(orders) if orders else 1
    for line in ocr.text_lines:
        if (line.reading_order_ix or -1) < 0:
            draw_func(
                ax,
                line.polygon,
                facecolor="black",
                edgecolor="black",
                alpha=0.9,
                hatch="XXX",
                text="!",
                text_color="red",
            )
        else:
            draw_func(
                ax,
                line.polygon,
                facecolor=cmap(line.reading_order_ix / max_o),
                edgecolor="black",
                alpha=0.8,
                text=str(line.reading_order_ix),
                text_color="white",
            )
