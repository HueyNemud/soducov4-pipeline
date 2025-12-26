from math import ceil
import numpy as np

from pipeline.ocr.schemas import OCRDocument


class Scorer:
    def __init__(self, min_chars: int = 10) -> None:
        self.min_chars = min_chars  # Seuil minimum de caractères

    def compute_and_assign(self, doc: OCRDocument) -> None:
        for page in doc.ocr:
            text_lines = [line.text for line in page.text_lines]
            spuriousness_scores = self.compute(text_lines)
            for line, score in zip(page.text_lines, spuriousness_scores):
                line.spuriousness = score

    def compute(self, text_lines: list[str]) -> np.ndarray:
        n = len(text_lines)
        if n == 0:
            return np.array([])

        # Longueurs des lignes
        lengths = np.array([len(line.strip()) for line in text_lines])

        # ---------------------------
        # Seuil auto-adaptatif
        # ---------------------------

        non_empty = lengths[lengths > 0]
        if len(non_empty) == 0:
            threshold_low = self.min_chars
        else:
            median_len = np.median(non_empty)
            threshold_low = max(self.min_chars, ceil(0.25 * median_len))

        # Sécurité
        threshold_low = max(threshold_low, 1)

        # ---------------------------
        # Score intrinsèque
        # ---------------------------

        intrinsic = np.zeros(n)

        # Ligne vide → max spurious
        intrinsic[lengths == 0] = 1.0

        # Lignes courtes
        mask = (lengths > 0) & (lengths < threshold_low)
        intrinsic[mask] = 1.0 - (lengths[mask] / threshold_low)

        # intrinsic = self._apply_incremental_penalty(intrinsic)
        # return intrinsic

        # Smoothing ciblé
        scores = np.zeros(n)

        for i in range(n):
            if lengths[i] >= threshold_low:
                scores[i] = 0.0
                continue

            neighbors = intrinsic[max(0, i - 1) : min(n, i + 2)]
            scores[i] = neighbors.mean()

        scores = self._apply_incremental_penalty(scores)

        return scores

    def _apply_incremental_penalty(self, intrinsic: np.ndarray) -> np.ndarray:
        penalized = intrinsic.copy()

        run_length = 0

        for i in range(len(penalized)):
            if penalized[i] > 0:
                run_length += 1
                # pénalité incrémentale bornée
                penalized[i] = 1 - (1 - penalized[i]) ** run_length
            else:
                run_length = 0

        return penalized


if __name__ == "__main__":
    # Exemple d'utilisation
    lines = [
        "Negocians, Marchands et Courtiers.",
        "10",
        "Bar, épicier, R. du Foin, 306. - Thermes.",
        "Bar, Md. de meubles, R. St. Honoré, 1352. - Butte des Moul.",
    ]

    scorer = Scorer()
    scores = scorer.compute(lines)

    for line, score in zip(lines, scores):
        print(f"Line: {line!r}, Spuriousness Score: {score:.4f}")
