"""
===========================
OCR Post-Processing components
==========================
"""

from enum import Enum, auto
from typing import Any, Dict, Tuple
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from numpy.typing import NDArray
from surya.layout.schema import LayoutResult, LayoutBox
from ..schemas import OCRResult, TextLine, OCRDocument

DEBUG = False
TAU_DEFAULT = 0.1  # valeur entre 0 et 1, représente la proportion minimale d'alignement horizontal
MAX_STRETCH_DEFAULT = 1000.0  # En % de la hauteur de la page
INCLUSION_THRESHOLD = 0.5  # Seuil d'inclusion pour l'assignation, en %


Polygon = NDArray[np.float64]  # shape (N, 4, 2)


class Direction(Enum):
    ABOVE = auto()
    BELOW = auto()


class SuryaPostProcessor:

    def __init__(self):
        pass

    def process_ocr_output(self, surya_output: OCRDocument) -> OCRDocument:

        fixed_output = OCRDocument(layout=[], ocr=[])

        for page_layout, page_textlines in zip(surya_output.layout, surya_output.ocr):
            # Masque les lignes de texte dont la certitude est trop faible
            page_lines = mask_poor_confidence_lines(
                page_textlines.text_lines, confidence_threshold=0.8
            )

            # Processus principal utilisant le code optimisé
            # new_layout_boxes: Polygon. Contains the stretched and refined layout boxes
            # reading_order: NDArray[np.int_]. Shape (n_layouts, n_textlines)
            # Contains the order of each textline in each layout box
            new_layout_boxes, reading_order = process(
                page_layout=page_layout,
                page_lines=page_lines,
                options={
                    "tau": TAU_DEFAULT,
                    "max_stretch": MAX_STRETCH_DEFAULT,
                    "threshold": INCLUSION_THRESHOLD,
                },
            )

            # Reconstruction du layout
            new_page_layout: LayoutResult = LayoutResult(
                bboxes=[
                    LayoutBox(
                        label=page_layout.bboxes[i].label,
                        polygon=new_layout_boxes[i],
                        top_k=page_layout.bboxes[i].top_k,
                        confidence=page_layout.bboxes[i].confidence,
                        position=page_layout.bboxes[i].position,
                    )
                    for i in range(new_layout_boxes.shape[0])
                ],
                image_bbox=page_layout.image_bbox,
                sliced=page_layout.sliced,
            )

            new_textlines = OCRResult(
                text_lines=[],
                image_bbox=page_textlines.image_bbox,
            )

            # Pour chaque layout box, récupérer les lignes de texte assignées dans l'ordre
            order = 1
            for j in range(new_layout_boxes.shape[0]):

                # Lignes assignées à ce layout box
                assigned_idx = np.where(reading_order[j] > 0)[0]

                if len(assigned_idx) == 0:
                    continue

                # Tri des textlines selon l'ordre de lecture
                sorted_lines = sorted(
                    assigned_idx, key=lambda idx: reading_order[j, idx]
                )

                for line_idx in sorted_lines:
                    text = page_lines[line_idx].text
                    confidence = page_lines[line_idx].confidence
                    polygon = page_lines[line_idx].polygon
                    chars = page_lines[line_idx].chars
                    order += 1

                    textline = TextLine(
                        polygon=polygon,
                        text=text,
                        confidence=confidence,
                        reading_order_ix=order,
                        layout_ix=j,
                        chars=chars,
                    )
                    new_textlines.text_lines.append(textline)

            fixed_output.layout.append(new_page_layout)
            fixed_output.ocr.append(new_textlines)
        return fixed_output


# =============================================================================
# Utilitaires géométriques
# =============================================================================


def get_bounds(poly: Polygon) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Extrait les limites min/max (x1, x2, y1, y2) d'un ensemble de polygones."""
    x = poly[:, :, 0]
    y = poly[:, :, 1]
    return (
        np.min(x, axis=1),
        np.max(x, axis=1),
        np.min(y, axis=1),
        np.max(y, axis=1),
    )


def compute_inclusion_ratios(
    container: Polygon, content: Polygon
) -> NDArray[np.float64]:
    """
    Calcule le ratio d'inclusion de 'content' à l'intérieur de 'container'.
    Retourne une matrice (N_container, N_content) des scores d'inclusion 2D.
    """
    # Broadcasting: (N, 1) vs (1, M)
    Ax1, Ax2, Ay1, Ay2 = [v[:, None] for v in get_bounds(container)]
    Bx1, Bx2, By1, By2 = [v[None, :] for v in get_bounds(content)]

    # Intersection
    inter_w = np.maximum(0, np.minimum(Ax2, Bx2) - np.maximum(Ax1, Bx1))
    inter_h = np.maximum(0, np.minimum(Ay2, By2) - np.maximum(Ay1, By1))

    # Dimensions de B (content)
    area_w = np.maximum(Bx2 - Bx1, 1e-9)
    area_h = np.maximum(By2 - By1, 1e-9)

    # Ratio d'inclusion (intersection/taille du contenu)
    return np.minimum(inter_w / area_w, inter_h / area_h)


# =============================================================================
# Layout stretching and vertical neighbors (Optimisés)
# =============================================================================
def vertical_alignment_score(
    Ax1: NDArray, Ax2: NDArray, Bx1: NDArray, Bx2: NDArray
) -> NDArray:
    """
    Le score d'alignement vertical est utilisé pour traduire numériquement à quel
    point deux boîtes 2D sont l'une au-dessus de l'autre.

    Le score considère les projections horizontales des boîtes A et B tel que :
        - [Ax1, Ax2] : projection horizontale de la boîte A
        - [Bx1, Bx2] : projection horizontale de la boîte B

    Le score est donné par score=max(ratio A ⊂ B, ratio B ⊂ A) avec :
        ratio A ⊂ B = longueur de l'intersection / longueur de A
        ratio B ⊂ A = longueur de l'intersection / longueur de B

    """
    # Broadcasting (N, 1) vs (1, M)
    inter = np.maximum(
        0,
        np.minimum(Ax2[:, None], Bx2[None, :]) - np.maximum(Ax1[:, None], Bx1[None, :]),
    )
    len_A = np.maximum(Ax2[:, None] - Ax1[:, None], 1e-9)
    len_B = np.maximum(Bx2[None, :] - Bx1[None, :], 1e-9)
    return np.maximum(inter / len_A, inter / len_B)


def find_closest_vertical_neighbors(
    L: Polygon, tau: float = 0.0, exclude_self: bool = True
) -> Dict[str, NDArray]:
    """
    Identifie les voisins verticaux immédiats (haut/bas) pour chaque polygone.
    tau : seuil minimum d'alignement horizontal pour considérer un voisin valide.
    """
    x1, x2, y1, y2 = get_bounds(L)

    # Masque d'alignement horizontal valide
    align_score = vertical_alignment_score(x1, x2, x1, x2)
    if exclude_self:
        np.fill_diagonal(align_score, -1)
    valid_h = align_score >= tau

    results = {}

    # Directions : (Nom, masque de position relative, calcul de distance)
    # Pour 'up' : le voisin B doit être au-dessus (B.y2 <= A.y1). Distance = A.y1 - B.y2
    # Pour 'down': le voisin B doit être en-dessous (B.y1 >= A.y2). Distance = B.y1 - A.y2
    directions = [
        (
            Direction.ABOVE.value,
            y2[None, :] <= y1[:, None],
            y1[:, None] - y2[None, :],
        ),
        (
            Direction.BELOW.value,
            y1[None, :] >= y2[:, None],
            y1[None, :] - y2[:, None],
        ),
    ]

    for key, pos_mask, dist_matrix in directions:
        mask = valid_h & pos_mask

        # Remplacer les distances invalides par l'infini pour l'argmin
        masked_dist = np.where(mask, dist_matrix, np.inf)

        min_dists = np.min(masked_dist, axis=1)
        min_idxs = np.argmin(masked_dist, axis=1)
        is_valid = ~np.isinf(min_dists)

        # Mappage des résultats
        results[f"{key}_idx"] = np.where(is_valid, min_idxs, -1)
        results[f"{key}_dist"] = np.where(is_valid, min_dists, np.inf)
        results[f"{key}_valid"] = is_valid

    return results


def stretch_layout(L: Polygon, tau: float = 0.1, max_stretch: float = 1000) -> Polygon:
    """Étire les polygones verticalement pour combler l'espace avec les voisins."""
    neighbors = find_closest_vertical_neighbors(L, tau, exclude_self=True)

    # Montants d'étirement (bornés par max_stretch si pas de voisin)
    s_up = np.where(
        neighbors[f"{Direction.ABOVE.value}_valid"],
        neighbors[f"{Direction.ABOVE.value}_dist"],
        max_stretch,
    )
    s_down = np.where(
        neighbors[f"{Direction.BELOW.value}_valid"],
        neighbors[f"{Direction.BELOW.value}_dist"],
        max_stretch,
    )

    L_stretched = L.copy()
    L_stretched[:, [0, 1], 1] -= s_up[:, None]  # Étirer le haut vers le haut
    L_stretched[:, [2, 3], 1] += s_down[:, None]  # Étirer le bas vers le bas
    return L_stretched


# =============================================================================
# Inclusion and assignment
# =============================================================================


# Fonction renommée pour correspondre au code original
def assign_and_find_assignable(
    T: Polygon, L: Polygon, LS: Polygon, threshold: float = 0.5
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Associe les lignes de texte aux layouts en deux passes (stricte puis étendue)."""
    N, M = L.shape[0], T.shape[0]

    # 1. Passe stricte (Layouts originaux)
    scores_orig = compute_inclusion_ratios(L, T)
    best_orig = np.argmax(scores_orig, axis=0)
    mask_assigned = scores_orig[best_orig, np.arange(M)] >= threshold

    initial_assignment = np.zeros((N, M), dtype=bool)
    initial_assignment[best_orig[mask_assigned], np.arange(M)[mask_assigned]] = True

    # 2. Passe de rattrapage (Layouts étirés) pour les lignes non assignées
    unassigned_mask = ~np.any(initial_assignment, axis=0)

    if np.any(unassigned_mask):
        scores_str = compute_inclusion_ratios(LS, T)
        # On ne regarde que les lignes non assignées
        relevant_scores = scores_str[:, unassigned_mask]
        best_str = np.argmax(relevant_scores, axis=0)

        valid_catch = (
            relevant_scores[best_str, np.arange(relevant_scores.shape[1])] >= threshold
        )

        unassigned_indices = np.where(unassigned_mask)[0]
        final_indices = unassigned_indices[valid_catch]
        best_layouts = best_str[valid_catch]

        extended_assignment = np.zeros((N, M), dtype=bool)
        extended_assignment[best_layouts, final_indices] = True
    else:
        extended_assignment = np.zeros((N, M), dtype=bool)

    return initial_assignment, extended_assignment


# Fonction renommée pour correspondre au code original
def reshape_layout_and_finalize_assignments(
    L: Polygon,
    T: Polygon,
    initial_assignment: NDArray[np.bool_],
    extended_assignment: NDArray[np.bool_],
) -> Tuple[Polygon, NDArray[np.bool_]]:
    """Ajuste la hauteur des layouts pour coller aux lignes assignées et finalise l'assignation."""
    L_refined = L.copy()
    combined = initial_assignment | extended_assignment
    N = L.shape[0]
    _, _, y1_text, y2_text = get_bounds(T)

    for i in range(N):
        lines_in_layout = combined[i]
        if np.any(lines_in_layout):
            # Limites globales des lignes contenues dans ce layout
            current_min = np.min(y1_text[lines_in_layout])
            current_max = np.max(y2_text[lines_in_layout])

            # Mise à jour des coordonnées Y
            L_refined[i, [0, 1], 1] = current_min
            L_refined[i, [2, 3], 1] = current_max

    output_assignment = combined
    return L_refined, output_assignment


# =============================================================================
# Page processing
# =============================================================================
def process(
    page_layout: Any,
    page_lines: Any,
    i: int = 1,
    options: Dict = {},
) -> Tuple[Polygon, NDArray[np.int_]]:
    """Processus complet pour une page : étirement, assignation, ordre de lecture, et debug plot."""
    layout_boxes = np.array([box.polygon for box in page_layout.bboxes])
    text_line_boxes = np.array([line.polygon for line in page_lines])

    # 0. Hyperparamètres
    tau = options.get("tau", TAU_DEFAULT)
    max_stretch = options.get("max_stretch", MAX_STRETCH_DEFAULT)
    inclusion_threshold = options.get("threshold", INCLUSION_THRESHOLD)

    # 1. Étirement
    stretched_layout_boxes = stretch_layout(
        layout_boxes, tau=tau, max_stretch=max_stretch
    )

    # 2. Assignation en deux passes (utilise les fonctions optimisées)
    initial_assignment, extended_assignment = assign_and_find_assignable(
        text_line_boxes,
        layout_boxes,
        stretched_layout_boxes,
        threshold=inclusion_threshold,
    )

    # 3. Raffinement des boîtes et finalisation de l'assignation
    corrected_layout_boxes, output_assignment = reshape_layout_and_finalize_assignments(
        layout_boxes, text_line_boxes, initial_assignment, extended_assignment
    )
    full_assignment = output_assignment  # ASSIGNABLE est maintenant vide

    # --- 4. Calcul de l'ordre de lecture
    # n : nombre de layouts, m : nombre de lignes de texte
    n, m = full_assignment.shape
    reading_order = np.zeros((n, m), dtype=int)
    _, _, line_tops, _ = get_bounds(text_line_boxes)

    for j in range(n):
        assigned_idx = np.where(full_assignment[j])[0]
        if len(assigned_idx) == 0:
            continue

        # Trier les lignes assignées par leur position Y (top-to-bottom)
        sorted_idx = assigned_idx[np.argsort(line_tops[assigned_idx])]
        for order, line_idx in enumerate(sorted_idx, start=1):
            reading_order[j, line_idx] = order

    # --- 5. Debug plotting (Conservé pour la vérification)
    if DEBUG:
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        ax.set_xlim(0, 6000)
        ax.set_ylim(0, 8000)
        # Assurer une palette de couleurs large pour les layouts
        colors = plt.cm.get_cmap("tab20", n)
        for j in range(n):
            assigned_idx = np.where(full_assignment[j])[0]
            for line_idx in assigned_idx:
                # Dessine le rectangle de la ligne de texte avec la couleur du layout
                poly = text_line_boxes[line_idx]
                # Utilise les coordonnées min/max pour dessiner le rectangle
                x1, x2, y1, y2 = (
                    np.min(poly[:, 0]),
                    np.max(poly[:, 0]),
                    np.min(poly[:, 1]),
                    np.max(poly[:, 1]),
                )
                rect = Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=colors(j),
                    linewidth=1,
                )
                ax.add_patch(rect)
        plt.savefig(f"debug_layout_page_{i}.png")
        plt.close(fig)

    return corrected_layout_boxes, reading_order


def mask_poor_confidence_lines(lines: Any, confidence_threshold: float = 0.5) -> Any:
    """Filtre les lignes de texte dont la confiance est inférieure au seuil."""
    filtered_lines = [line for line in lines if line.confidence >= confidence_threshold]
    return filtered_lines
