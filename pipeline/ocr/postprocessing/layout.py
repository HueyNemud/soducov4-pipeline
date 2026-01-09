from enum import Enum
from typing import Any, Dict, Iterable, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray
from surya.layout.schema import LayoutResult, LayoutBox
from pipeline.ocr.schemas import OCRResult, TextLine, OCRDocument
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon
import os

Polygon = NDArray[np.float64]


class Direction(Enum):
    ABOVE = "above"
    BELOW = "below"


# =======================
# Helper Functions
# =======================


def get_bounds(polys: Polygon) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Calcule les bornes axis-aligned (x_min, x_max, y_min, y_max) de chaque polygone.

    Parameters
    ----------
    polys : NDArray
        Polygones de forme (N, 4, 2) ou (4, 2).

    Returns
    -------
    x_min, x_max, y_min, y_max : NDArray
        Coordonnées min/max pour chaque polygone.
    """
    if polys.ndim == 2:
        polys = polys[None, :, :]
    xs, ys = polys[:, :, 0], polys[:, :, 1]
    return (
        np.min(xs, axis=1),
        np.max(xs, axis=1),
        np.min(ys, axis=1),
        np.max(ys, axis=1),
    )


def mask_poor_confidence_lines(
    lines: Iterable[TextLine], threshold: float
) -> list[TextLine]:
    """Filtre les lignes OCR avec faible confiance."""
    return [l for l in lines if l.confidence or 1.0 >= threshold]


def split_multicolumn_text_lines(
    lines: Iterable[TextLine], separators: Sequence[str] = ("|",)
) -> list[TextLine]:
    """
    Divise les lignes OCR multi-colonnes en plusieurs lignes.
    Les polygones sont redimensionnés proportionnellement à la longueur du texte.
    """
    refined = []
    for line in lines:
        sep = next((s for s in separators if s in line.text), None)
        if not sep:
            refined.append(line)
            continue
        parts = [p.strip() for p in line.text.split(sep) if p.strip()]
        if len(parts) <= 1:
            refined.append(line)
            continue

        xs = [p[0] for p in line.polygon]
        ys = [p[1] for p in line.polygon]
        min_x, max_x = min(xs), max(xs)
        total_width = max_x - min_x
        char_counts = [max(len(p), 1) for p in parts]
        total_chars = sum(char_counts)
        current_x = min_x

        for text, c in zip(parts, char_counts):
            width = total_width * c / total_chars
            refined.append(
                TextLine(
                    polygon=[
                        [current_x, ys[0]],
                        [current_x + width, ys[1]],
                        [current_x + width, ys[2]],
                        [current_x, ys[3]],
                    ],
                    text=text,
                    confidence=line.confidence,
                    chars=[],
                )
            )
            current_x += width
    return refined


# =======================
# Probabilistic Assignment
# =======================


def _compute_overlap_score(layouts, lines) -> np.ndarray:
    Ax1, Ax2, Ay1, Ay2 = get_bounds(layouts)
    Bx1, Bx2, By1, By2 = get_bounds(lines)

    # Intersection areas
    inter_w = np.maximum(
        0,
        np.minimum(Ax2[:, None], Bx2[None, :]) - np.maximum(Ax1[:, None], Bx1[None, :]),
    )
    inter_h = np.maximum(
        0,
        np.minimum(Ay2[:, None], By2[None, :]) - np.maximum(Ay1[:, None], By1[None, :]),
    )

    # On normalise par la taille de la ligne (B) pour savoir à quel point LA LIGNE est dans le layout
    line_w = np.maximum(Bx2 - Bx1, 1e-9)
    line_h = np.maximum(By2 - By1, 1e-9)

    # Score d'inclusion : 1.0 si la ligne est totalement dans le rectangle
    return (inter_w / line_w[None, :]) * (inter_h / line_h[None, :])

import numpy as np

def horizontal_priority_vertical_decay_left(layouts: np.ndarray, lines: np.ndarray, decay_factor: float = 2.0, left_fraction: float = 0.5) -> np.ndarray:
    """
    Score layout ↔ ligne privilégiant l'alignement horizontal (colonne)
    avec biais vers les layouts à gauche en utilisant seulement la moitié gauche
    de la ligne pour l'inclusion horizontale et pour calculer le centre vertical.
    Décroissance verticale selon la distance minimale au layout.

    Parameters
    ----------
    layouts : (N,4) ou (N,4,2)
        Layout boxes (x_min, y_min, x_max, y_max) ou polygones.
    lines : (M,4) ou (M,4,2)
        Text line boxes.
    decay_factor : float
        Multiplicateur pour la décroissance verticale (2× hauteur médiane ligne par défaut).
    left_fraction : float
        Fraction de la ligne utilisée pour l'inclusion horizontale et le centre vertical (0.5 = moitié gauche)

    Returns
    -------
    score : (N,M) float
        Score ∈ [0,1] par layout × ligne
    """
    # --- Convertir polygones en bbox ---
    if layouts.ndim == 3:
        Lx1 = layouts[:, :, 0].min(axis=1)
        Ly1 = layouts[:, :, 1].min(axis=1)
        Lx2 = layouts[:, :, 0].max(axis=1)
        Ly2 = layouts[:, :, 1].max(axis=1)
    else:
        Lx1, Ly1, Lx2, Ly2 = layouts.T

    if lines.ndim == 3:
        Tx1 = lines[:, :, 0].min(axis=1)
        Ty1 = lines[:, :, 1].min(axis=1)
        Tx2 = lines[:, :, 0].max(axis=1)
        Ty2 = lines[:, :, 1].max(axis=1)
    else:
        Tx1, Ty1, Tx2, Ty2 = lines.T

    # --- Moitié gauche de la ligne ---
    Tx_half = Tx1 + left_fraction * (Tx2 - Tx1)
    # Inclusion horizontale
    inter_x = np.maximum(0, np.minimum(Lx2[:, None], Tx_half[None, :]) - np.maximum(Lx1[:, None], Tx1[None, :]))
    line_width = np.maximum(Tx_half - Tx1, 1e-9)
    horiz_score = inter_x / line_width[None, :]  # signal principal favorisant les layouts à gauche

    # --- centre de la moitié gauche pour la distance verticale ---
    line_centers_y = (Ty1 + Ty2) / 2  # ou si tu veux vraiment moitié gauche verticale, reste Ty1->Ty2
    line_centers_x = (Tx1 + Tx_half) / 2  # centre horizontal moitié gauche (optionnel, utile si tu veux bias horizontal)

    # --- distance verticale minimale au layout (0 si centre à l'intérieur) ---
    dist_y = np.maximum(Ly1[:, None] - line_centers_y[None, :], 0) + \
             np.maximum(line_centers_y[None, :] - Ly2[:, None], 0)

    # --- décroissance exponentielle selon distance verticale ---
    line_heights = Ty2 - Ty1
    sigma = np.median(line_heights) * decay_factor if line_heights.size > 0 else 20.0
    decay_vertical = np.exp(-dist_y / sigma)

    # --- score final ---
    score = horiz_score * decay_vertical
    return score




def assign_lines_probabilistic(
    layouts: np.ndarray,
    lines: np.ndarray,
    direction: str = "LTR",
) -> tuple[np.ndarray, np.ndarray]:
    N, M = layouts.shape[0], lines.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=bool), np.zeros((N, M), dtype=float)

    # # Inclusion remains the strongest signal (if it's inside, it belongs)
    # inc_score = _compute_overlap_score(layouts, lines)
    
    # # Vector-Angle is our fallback for lines slightly outside or near boxes
    # valign_score = vertical_alignment_score(layouts, lines)

    # score = inc_score + (1 - inc_score) * valign_score
    score  = horizontal_priority_vertical_decay_left(layouts, lines)
    
    # Attribution : layout de score max
    best_layout_indices = np.argmax(score, axis=0)
    assigned_layouts = np.zeros((N, M), dtype=bool)
    assigned_layouts[best_layout_indices, np.arange(M)] = True

    return assigned_layouts, score

def compute_local_reading_order(
    full_assignment: np.ndarray,
    text_boxes: np.ndarray,
    direction: str = "LTR",
    n_bins: int = 400,
) -> np.ndarray:
    """
    Compute a local reading order for each layout by:
    1. Estimating column spans from horizontal text density
    2. Assigning each line to a column using its leading edge
    3. Sorting lines column-wise, then top-to-bottom

    Parameters
    ----------
    full_assignment : (n_layouts, N) bool or int
        Layout-to-text-line assignment matrix.
    text_boxes : (N, 4, 2)
        Text line polygons (x, y).
    direction : {"LTR", "RTL"}
        Reading direction.
    n_bins : int
        Horizontal density resolution.

    Returns
    -------
    reading_order : (n_layouts, N) int
        1-based reading order per layout.
    """

    n_layouts, n_lines = full_assignment.shape
    reading_order = np.zeros((n_layouts, n_lines), dtype=int)

    # ------------------------------------------------------------------
    # Geometry normalization (polygon → bounding box)
    # ------------------------------------------------------------------
    x_coords = text_boxes[:, :, 0]
    y_coords = text_boxes[:, :, 1]

    x_min = x_coords.min(axis=1)
    x_max = x_coords.max(axis=1)
    y_center = y_coords.mean(axis=1)

    # Leading edge is the most stable column anchor
    leading_x = x_min if direction == "LTR" else x_max

    for layout_id in range(n_layouts):
        idxs = np.flatnonzero(full_assignment[layout_id])
        if idxs.size == 0:
            continue

        # --------------------------------------------------------------
        # 1. Horizontal density estimation (column detection)
        # --------------------------------------------------------------
        b_min = x_min[idxs].min()
        b_max = x_max[idxs].max()
        width = max(b_max - b_min, 1.0)

        bin_width = width / n_bins
        density = np.zeros(n_bins, dtype=np.int32)

        # Project only the leading half of each line
        if direction == "LTR":
            starts = x_min[idxs]
            ends   = (x_min[idxs] + x_max[idxs]) * 0.5
        else:
            starts = (x_min[idxs] + x_max[idxs]) * 0.5
            ends   = x_max[idxs]

        s_bins = ((starts - b_min) / bin_width).astype(int)
        e_bins = ((ends   - b_min) / bin_width).astype(int)

        s_bins = np.clip(s_bins, 0, n_bins - 1)
        e_bins = np.clip(e_bins, 0, n_bins - 1)

        for s, e in zip(s_bins, e_bins):
            density[s:e + 1] += 1

        # --------------------------------------------------------------
        # 2. Column span extraction from density
        # --------------------------------------------------------------
        if np.any(density):
            threshold = max(1, np.percentile(density[density > 0], 10))
            occupied = density > threshold
        else:
            occupied = np.zeros_like(density, dtype=bool)

        # Convert occupied bins → continuous column intervals
        columns = []
        in_col = False
        start = 0

        for i, occ in enumerate(occupied):
            if occ and not in_col:
                in_col, start = True, i
            elif not occ and in_col:
                in_col = False
                columns.append((start, i))

        if in_col:
            columns.append((start, n_bins))

        if not columns:
            columns = [(0, n_bins)]

        column_spans = [
            (b_min + s * bin_width, b_min + e * bin_width)
            for s, e in columns
        ]

        # --------------------------------------------------------------
        # 3. Assign each line to the closest column (leading edge)
        # --------------------------------------------------------------
        anchors = leading_x[idxs]
        col_assign = np.empty(idxs.size, dtype=int)

        for i, x in enumerate(anchors):
            col_assign[i] = min(
                range(len(column_spans)),
                key=lambda c: max(
                    0,
                    column_spans[c][0] - x,
                    x - column_spans[c][1],
                ),
            )

        # --------------------------------------------------------------
        # 4. Reading order: column → vertical
        # --------------------------------------------------------------
        col_priority = col_assign if direction == "LTR" else -col_assign
        order = np.lexsort((y_center[idxs], col_priority))

        reading_order[layout_id, idxs[order]] = np.arange(1, idxs.size + 1)

    return reading_order

def process_probabilistic(
    page_layout: Any, 
    page_lines: Any, 
    options: Dict | None = None
) -> tuple[Polygon, NDArray[np.int_], NDArray[np.float64]]:
    """
    Transforms layout polygons and OCR lines into a local reading order.
    """
    options = options or {}

    # 1. Prepare data
    layout_boxes = np.array([b.polygon for b in page_layout.bboxes], dtype=np.float64)
    text_boxes = np.array([l.polygon for l in page_lines], dtype=np.float64)

    # 2. Run the new vector-angle assignment logic
    full_assignment, prob = assign_lines_probabilistic(
        layout_boxes, 
        text_boxes, 
        direction=options.get("direction", "LTR")
    )

    # 3. Compute reading order using the separated logic
    reading_order = compute_local_reading_order(full_assignment, text_boxes)

    return layout_boxes, reading_order, prob


# =======================
# Main PostProcessor
# =======================


class SuryaPostProcessor:
    """
    Post-traitement OCR :
    1. Nettoyage des lignes avec faible confiance.
    2. Split multicolonnes.
    3. Assignation probabiliste des lignes aux layouts.
    4. Reconstruction d'un OCRDocument structuré avec ordre de lecture global.
    5. Export debug optionnel.

    Notes
    -----
    - Utilise la logique top-to-bottom pour chaque layout.
    - Permet de gérer des layouts imbriqués (parent-child).
    """

    def __init__(self, debug_plots: bool = False, debug_dir: str | None = None):
        self._debug_plots = debug_plots
        self._debug_dir = debug_dir

    def process_ocr_output(self, surya_output: OCRDocument) -> OCRDocument:
        """
        Transforme la sortie OCR brute en un document structuré post-traité.

        Parameters
        ----------
        surya_output : OCRDocument
            Document OCR original.

        Returns
        -------
        OCRDocument
            Document post-traité avec layouts corrigés et lignes assignées.
        """
        fixed_output = OCRDocument(layout=[], ocr=[])

        for page_layout, page_textlines in zip(surya_output.layout, surya_output.ocr):
            print(f"Processing page with {len(page_layout.bboxes)} layout boxes and {len(page_textlines.text_lines)} text lines.")
            # 1. Nettoyage et split multicolonnes
            lines = mask_poor_confidence_lines(page_textlines.text_lines, 0.5)
            # lines = split_multicolumn_text_lines(lines)

            # 2. Assignation probabiliste
            layout_boxes, reading_order, line_probs_dict = (
                self._process_probabilistic_debug(page_layout, lines)
            )

            # 3. Reconstruction LayoutResult
            new_layout = LayoutResult(
                bboxes=[
                    LayoutBox(
                        label=page_layout.bboxes[i].label,
                        polygon=layout_boxes[i].tolist(),
                        top_k=page_layout.bboxes[i].top_k,
                        confidence=page_layout.bboxes[i].confidence,
                        position=page_layout.bboxes[i].position,
                    )
                    for i in range(layout_boxes.shape[0])
                ],
                image_bbox=page_layout.image_bbox,
                sliced=page_layout.sliced,
            )

            # 4. Reconstruction OCRResult
            new_ocr = OCRResult(text_lines=[], image_bbox=page_textlines.image_bbox)
            counter = 1
            for layout_idx in range(layout_boxes.shape[0]):
                assigned = np.flatnonzero(reading_order[layout_idx] > 0)
                for idx in assigned[np.argsort(reading_order[layout_idx, assigned])]:
                    line = lines[idx]
                    new_ocr.text_lines.append(
                        TextLine(
                            polygon=line.polygon,
                            text=line.text,
                            confidence=line.confidence,
                            reading_order_ix=counter,
                            layout_ix=layout_idx,
                            chars=line.chars,
                        )
                    )
                    counter += 1

            print(f"Assigned {len(new_ocr.text_lines)} lines after processing.")
            fixed_output.layout.append(new_layout)
            fixed_output.ocr.append(new_ocr)

        # 5. Export debug plots
        if self._debug_plots:
            self._export_debug(fixed_output, line_probs_dict)

        return fixed_output

    # -----------------------
    # Helper avec dict de probas
    # -----------------------
    def _process_probabilistic_debug(
    self, page_layout: Any, lines: list[TextLine]
    ) -> tuple[Polygon, NDArray[np.int_], dict]:
        """
        Version corrigée : utilise la même logique de tri que la prod.
        """
        layout_boxes = np.array(
            [b.polygon for b in page_layout.bboxes], dtype=np.float64
        )
        text_boxes = np.array([l.polygon for l in lines], dtype=np.float64)

        # 1. Même logique d'assignation
        assigned_layouts, prob_matrix = assign_lines_probabilistic(
            layout_boxes, text_boxes
        )
        
        # --- CORRECTION 1 : Utilisation de la logique de lecture unifiée ---
        reading_order = compute_local_reading_order(
            assigned_layouts, 
            text_boxes, 
            direction="LTR" # Idéalement à passer via options
        )

        # Créer le dict id(line) → vecteur de probas pour debug plots
        line_probs_dict = {id(lines[i]): prob_matrix[:, i] for i in range(len(lines))}

        return layout_boxes, reading_order, line_probs_dict

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _get_line_debug_style(self, line, prob_vector, cmap_layouts):
        """
        Détermine le style graphique d'une ligne OCR en mode debug.
        Centralise toute la logique métier debug (orphelin, alpha, couleurs).
        """
        is_orphan = (
            line.layout_ix is None
            or line.reading_order_ix is None
            or line.reading_order_ix < 0
        )

        if is_orphan:
            return {
                "facecolor": "black",
                "edgecolor": "red",
                "alpha": 1.0,
                "hatch": "////",
                "text_color": "red",
            }

        color = cmap_layouts(line.layout_ix % cmap_layouts.N)

        alpha = 1.0
        if prob_vector is not None:
            idx = min(line.layout_ix, len(prob_vector) - 1)
            alpha = max(0.1, float(prob_vector[idx]))

        return {
            "facecolor": color,
            "edgecolor": color,
            "alpha": alpha,
            "hatch": None,
            "text_color": "white",
        }

    def _draw_polygon(
        self,
        ax,
        polygon,
        *,
        facecolor,
        edgecolor,
        alpha=1.0,
        hatch=None,
        text=None,
        text_color="black",
        linewidth=1.5,
    ):
        """
        Dessine un polygone matplotlib avec texte optionnel centré.
        """
        poly = np.asarray(polygon)

        ax.add_patch(
            MplPolygon(
                poly,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                hatch=hatch,
            )
        )

        if text is not None:
            cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
            ax.text(
                float(cx),
                float(cy),
                text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    def _plot_layout_boxes(self, ax, page_layout, cmap_layouts):
        """IMAGE 1 — blocs de layout"""
        for i, box in enumerate(page_layout.bboxes):
            self._draw_polygon(
                ax,
                box.polygon,
                facecolor=cmap_layouts(i % cmap_layouts.N),
                edgecolor="black",
                alpha=0.3,
                text=str(i),
                linewidth=2,
            )

    def _plot_lines_with_confidence(self, ax, page_ocr, line_probs_dict, cmap_layouts):
        """IMAGE 2 — lignes OCR avec alpha = proba"""
        for line in page_ocr.text_lines:
            prob_vector = line_probs_dict.get(id(line))
            style = self._get_line_debug_style(line, prob_vector, cmap_layouts)

            self._draw_polygon(
                ax,
                line.polygon,
                **style,
                text=str(line.layout_ix if line.layout_ix is not None else "?"),
            )

        legend_elements = [
            Patch(
                facecolor="black",
                edgecolor="red",
                hatch="////",
                label="Orphelin (Layout / Reading order manquant)",
            )
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    def _plot_reading_order(self, ax, page_ocr, cmap_order):
        """IMAGE 3 — gradient selon l'ordre de lecture"""
        reading_orders = [
            l.reading_order_ix
            for l in page_ocr.text_lines
            if l.reading_order_ix is not None and l.reading_order_ix > 0
        ]
        max_order = max(reading_orders) if reading_orders else 1

        for line in page_ocr.text_lines:
            poly = np.asarray(line.polygon)

            if line.reading_order_ix is None or line.reading_order_ix < 0:
                self._draw_polygon(
                    ax,
                    poly,
                    facecolor="black",
                    edgecolor="black",
                    alpha=0.9,
                    hatch="XXX",
                    text="!",
                    text_color="red",
                )
            else:
                norm = line.reading_order_ix / max_order
                self._draw_polygon(
                    ax,
                    poly,
                    facecolor=cmap_order(norm),
                    edgecolor="black",
                    alpha=0.8,
                    text=str(line.reading_order_ix),
                    text_color="white",
                )

    # ------------------------------------------------------------------
    # Main debug export
    # ------------------------------------------------------------------

    def _export_debug(self, ocr_doc: "OCRDocument", line_probs_dict: dict):
        """
        Visualisation debug complète :
        1. Layout blocks
        2. Confidence & assignment
        3. Reading order flow
        """
        output_dir = self._debug_dir or "./debug_plots"
        os.makedirs(output_dir, exist_ok=True)

        cmap_layouts = plt.get_cmap("tab20")
        cmap_order = plt.get_cmap("plasma")

        for p_idx, (page_layout, page_ocr) in enumerate(
            zip(ocr_doc.layout, ocr_doc.ocr)
        ):
            bbox = page_layout.image_bbox

            # --- IMAGE 1 : Layouts ---
            fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
            self._plot_layout_boxes(ax, page_layout, cmap_layouts)
            self._finalize_plot(ax, bbox, f"Page {p_idx} - Layout Blocks")
            fig.savefig(
                os.path.join(output_dir, f"page_{p_idx}_layout.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

            # --- IMAGE 2 : Confidence ---
            fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
            self._plot_lines_with_confidence(
                ax, page_ocr, line_probs_dict, cmap_layouts
            )
            self._finalize_plot(ax, bbox, f"Page {p_idx} - Confidence & Assignment")
            fig.savefig(
                os.path.join(output_dir, f"page_{p_idx}_lines_prob.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

            # --- IMAGE 3 : Reading Order ---
            fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
            self._plot_reading_order(ax, page_ocr, cmap_order)
            self._finalize_plot(ax, bbox, f"Page {p_idx} - Reading Order Flow")
            fig.savefig(
                os.path.join(output_dir, f"page_{p_idx}_reading_order.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    def _finalize_plot(self, ax, bbox, title):
        """Formatage commun à tous les plots debug"""
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=20)
        ax.axis("off")
