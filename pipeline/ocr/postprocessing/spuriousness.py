from math import ceil
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from pipeline.ocr.schemas import OCRDocument


class SpuriousnessScorer:
    """
    Scorer that identifies 'spurious' OCR lines (artifacts, noise, or fragments).

    It combines three distinct signals:
    1.  Intrinsic Length: How short is the line relative to the page average?
    2.  Local Context: Are the neighboring lines also suspicious? (Smoothing)
    3.  Sequence Decay: Is this line part of a long run of garbage? (Reinforcement)
    """

    @dataclass
    class Parameters:
        """
        Hyper-parameters for the scoring algorithm.

        Attributes:
            min_chars: Absolute noise floor. Any line below this is highly suspect.
            adaptive_ratio: Percentage of the median page length used to calculate
                the adaptive threshold (e.g., 0.25 means 1/4 of the typical line).
        """

        min_chars: int = Field(default=10, ge=1)
        adaptive_ratio: float = Field(default=0.25, ge=0.0, le=1.0)

    def compute_and_assign(self, doc: OCRDocument, params: Parameters) -> None:
        """
        Processes an OCRDocument and assigns a spuriousness score to every text line.

        Args:
            doc: The document to enrich.
            params: Scoring parameters. Uses class defaults if None.
        """

        for page in doc.ocr:
            if not page.text_lines:
                continue

            # Vectorized compute call
            texts = [line.text for line in page.text_lines]
            scores = self.compute(texts, params)

            for line, score in zip(page.text_lines, scores):
                line.spuriousness = float(score)

    def compute(self, text_lines: list[str], params: Parameters) -> np.ndarray:
        """
        Calculates spuriousness scores using a vectorized pipeline.

        Algorithm Steps:
        ----------------
        1. Adaptive Thresholding: Determine the 'noise floor' based on page statistics.
           This mechanism uses a dual-safety approach:
           - min_chars: Acts as an absolute floor. Any line below this is suspect,
             regardless of the page's density.
           - adaptive_ratio: Acts as a contextual ceiling. It scales the threshold
             up for documents with very long lines (e.g., full-page articles),
             ensuring that fragments which are technically longer than min_chars
             but significantly shorter than the page average are caught.
           The final threshold is max(min_chars, adaptive_ratio * median_length).

        2. Intrinsic Scoring: Measure the raw distance of each line to that floor.
           Calculates a base score in [0.0, 1.0] proportional to how far below
           the threshold a line falls.

        3. Contextual Smoothing: Apply a 1D convolution to average scores with neighbors.
           In directory scans, spurious lines are rarely isolated; they typically
           appear in clusters. This often happens when the OCR detects truncated
           text from an adjacent page's columns appearing in the scan margins.
           Smoothing ensures that a small fragment surrounded by other noise
           is heavily penalized.

        4. Reinforcement: Apply a penalty for consecutive suspicious lines.
           Increases the confidence of the score for repetitive noise sequences
           (like a vertical strip of truncated margin text) using a non-linear
           run-length penalty.
        """
        if not text_lines:
            return np.array([], dtype=np.float64)

        # 1. Feature Extraction: Line Lengths
        lengths = np.array([len(line.strip()) for line in text_lines], dtype=np.int32)

        # 2. Adaptive Threshold Calculation
        # We calculate the median length of non-empty lines to understand the
        # document's scale (e.g., dense text vs. sparse lists).
        non_empty = lengths[lengths > 0]
        page_median = np.median(non_empty) if non_empty.size > 0 else params.min_chars

        # The threshold is the maximum of the absolute floor and the adaptive ratio.
        threshold = max(params.min_chars, ceil(params.adaptive_ratio * page_median), 1)

        # 3. Intrinsic Score Calculation
        # Lines longer than threshold get 0.0. Empty lines get 1.0.
        # Lines in between get a linear score based on their distance to threshold.
        intrinsic = np.clip(1.0 - (lengths / threshold), 0.0, 1.0)

        # 4. Spatial Smoothing (Moving Average)
        # We use a 3-tap kernel [1/3, 1/3, 1/3] to average each line with its
        # immediate predecessor and successor. Spurious lines often appear in clusters.
        kernel = np.ones(3) / 3.0
        scores = np.convolve(intrinsic, kernel, mode="same")

        # 5. Threshold Immunity
        # Explicitly reset scores for lines that are clearly 'long enough'.
        scores[lengths >= threshold] = 0.0

        # 6. Run-length Reinforcement
        return self._apply_incremental_penalty(scores)

    def _apply_incremental_penalty(self, scores: np.ndarray) -> np.ndarray:
        """
        Increases the confidence of the score for consecutive suspicious blocks.

        Uses the formula: P_new = 1 - (1 - P_initial)^run_length
        This boosts a score of 0.5 to 0.75 on the 2nd line, 0.875 on the 3rd, etc.
        """
        penalized = scores.copy()
        run_length = 0

        for i, s in enumerate(penalized):
            if s > 0:
                run_length += 1
                penalized[i] = 1.0 - (1.0 - s) ** run_length
            else:
                run_length = 0

        return penalized
