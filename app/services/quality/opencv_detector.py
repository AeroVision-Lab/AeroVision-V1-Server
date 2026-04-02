"""
OpenCV-based image quality detector.

Covers rules that can be evaluated purely with classical CV metrics:

    Rules1.1.1  Blurry / out-of-focus        (Laplacian variance)
    Rules1.1.2  Oversharpen / noisy           (edge-to-noise ratio)
    Rules1.2.1  Bad contrast                  (histogram std-dev)
    Rules1.2.3  Overexposed / underexposed    (bright/dark pixel ratio)
    Rules1.3.1  Bad colour                    (HSV saturation)
    Rules1.7.1  Horizon unlevel               (Hough line angle)
    Rules1.10.1 Vignetting                    (centre-to-edge brightness)
    Rules1.10.2 Hazy / foggy                  (dark-channel prior)
    Rules1.10.3 Purple fringing               (hue mask in edge regions)

Each metric is mapped to a [0, 1] score where **1 = perfect**.
Violations are emitted when a score falls below its threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from app.core import get_logger
from app.schemas.quality import QualityDetails, RuleViolation

logger = get_logger("services.quality.opencv")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OpenCVResult:
    """All outputs from the OpenCV detector."""

    violations: list[RuleViolation] = field(default_factory=list)
    dimension_scores: QualityDetails | None = None
    # individual raw scores (kept for hybrid merger / debugging)
    sharpness_score: float = 1.0
    noise_score: float = 1.0
    contrast_score: float = 1.0
    exposure_score: float = 1.0
    color_score: float = 1.0
    horizon_score: float = 1.0
    vignetting_score: float = 1.0
    haze_score: float = 1.0
    purple_fringe_score: float = 1.0


# ---------------------------------------------------------------------------
# Thresholds (all tuneable)
# ---------------------------------------------------------------------------

# Rules1.1.1  – Laplacian variance below this → blurry
_SHARPNESS_BLUR_THRESHOLD = 0.25
# Rules1.1.2  – above this → oversharpen; also uses noise score
_SHARPNESS_OVER_THRESHOLD = 0.90
_NOISE_OVER_THRESHOLD = 0.30          # low noise score = high noise
# Rules1.2.1  – histogram std-dev
_CONTRAST_LOW_THRESHOLD = 0.25
_CONTRAST_HIGH_THRESHOLD = 0.92
# Rules1.2.3  – pixel ratio outside [5, 250]
_EXPOSURE_DARK_THRESHOLD = 0.25       # too many dark pixels
_EXPOSURE_BRIGHT_THRESHOLD = 0.25     # too many bright pixels
# Rules1.3.1  – mean HSV saturation
_COLOR_SAT_THRESHOLD = 0.18
# Rules1.7.1  – dominant near-horizontal line angle (degrees)
_HORIZON_TILT_DEG = 5.0
# Rules1.10.1 – centre/edge brightness ratio
_VIGNETTING_THRESHOLD = 0.30
# Rules1.10.2 – dark-channel prior score
_HAZE_THRESHOLD = 0.30
# Rules1.10.3 – purple-fringe pixel density
_PURPLE_FRINGE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class OpenCVDetector:
    """
    Runs all OpenCV-based quality checks on a PIL Image and returns an
    ``OpenCVResult`` containing violations and per-dimension scores.
    """

    def detect(self, image: Image.Image) -> OpenCVResult:
        """
        Run all checks.

        Args:
            image: PIL Image (any mode; converted internally as needed).

        Returns:
            OpenCVResult with violations and dimension scores.
        """
        # Convert to numpy BGR (standard for OpenCV)
        bgr = self._to_bgr(image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        result = OpenCVResult()

        # --- run every metric -----------------------------------------------
        result.sharpness_score = self._score_sharpness(gray)
        result.noise_score = self._score_noise(gray)
        result.contrast_score = self._score_contrast(gray)
        result.exposure_score = self._score_exposure(gray)
        result.color_score = self._score_color(hsv)
        result.horizon_score = self._score_horizon(gray)
        result.vignetting_score = self._score_vignetting(gray)
        result.haze_score = self._score_haze(bgr)
        result.purple_fringe_score = self._score_purple_fringe(bgr, gray)

        # --- map scores → violations ----------------------------------------
        self._check_rules(result)

        # --- build QualityDetails for backward compat -----------------------
        result.dimension_scores = QualityDetails(
            sharpness=result.sharpness_score,
            exposure=result.exposure_score,
            composition=result.horizon_score,       # best proxy from OpenCV
            noise=result.noise_score,
            color=result.color_score,
        )

        logger.debug(
            "OpenCV scores: sharpness=%.2f noise=%.2f contrast=%.2f "
            "exposure=%.2f color=%.2f horizon=%.2f vignetting=%.2f "
            "haze=%.2f purple=%.2f  violations=%d",
            result.sharpness_score, result.noise_score, result.contrast_score,
            result.exposure_score, result.color_score, result.horizon_score,
            result.vignetting_score, result.haze_score, result.purple_fringe_score,
            len(result.violations),
        )

        return result

    # -----------------------------------------------------------------------
    # Metric helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _to_bgr(image: Image.Image) -> np.ndarray:
        """Convert PIL Image → uint8 BGR ndarray."""
        rgb = image.convert("RGB")
        return cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _score_sharpness(gray: np.ndarray) -> float:
        """
        Rules1.1.1 – Laplacian variance normalised to [0, 1].

        The raw variance is mapped through a sigmoid-like cap at 1000 so the
        score saturates gracefully on very sharp images.
        """
        var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # cap at 1000; typical sharp photo ~200-800, blurry <50
        return float(np.clip(var / 1000.0, 0.0, 1.0))

    @staticmethod
    def _score_noise(gray: np.ndarray) -> float:
        """
        Rules1.1.2 – estimate noise via median-blur residual.

        High residual = high noise → low score.
        """
        blurred = cv2.medianBlur(gray, 5)
        residual = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
        mean_residual = float(residual.mean())
        # typical clean image ~1-3; noisy ~8-20
        return float(np.clip(1.0 - mean_residual / 15.0, 0.0, 1.0))

    @staticmethod
    def _score_contrast(gray: np.ndarray) -> float:
        """
        Rules1.2.1 – histogram standard deviation normalised to [0, 1].

        Very low std-dev = flat image (low contrast).
        Very high std-dev = extreme contrast (bimodal histogram).
        Score is highest in the 35-75 std-dev range.
        """
        std = float(gray.std())
        # ideal band [35, 75]; outside this band score degrades
        low_score = float(np.clip(std / 35.0, 0.0, 1.0))
        high_score = float(np.clip(1.0 - (std - 75.0) / 55.0, 0.0, 1.0)) if std > 75 else 1.0
        return min(low_score, high_score)

    @staticmethod
    def _score_exposure(gray: np.ndarray) -> float:
        """
        Rules1.2.3 – ratio of pixels in a "good" brightness range [15, 240].

        Both over-exposed (many pixels ~255) and under-exposed (many pixels ~0)
        images score low.
        """
        total = gray.size
        good = int(np.sum((gray >= 15) & (gray <= 240)))
        return float(good / total)

    @staticmethod
    def _score_color(hsv: np.ndarray) -> float:
        """
        Rules1.3.1 – mean HSV saturation normalised to [0, 1].

        Very desaturated images score low.  Night/dusk scenes with intentional
        low saturation will also score low – Qwen handles context-awareness.
        """
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        return float(np.clip(sat.mean() / 0.35, 0.0, 1.0))

    @staticmethod
    def _score_horizon(gray: np.ndarray) -> float:
        """
        Rules1.7.1 – detect dominant near-horizontal lines via HoughLinesP.

        Returns a score in [0, 1] where 1 = level, 0 = ≥10° tilt.
        """
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=gray.shape[1] // 4,
            maxLineGap=20,
        )
        if lines is None or len(lines) == 0:
            return 1.0  # no strong lines found – assume OK

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle_deg = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            # Keep only near-horizontal lines (angle < 30°)
            if angle_deg < 30:
                angles.append(angle_deg)

        if not angles:
            return 1.0

        median_tilt = float(np.median(angles))
        # score: 0° tilt → 1.0 ; 10° tilt → 0.0
        return float(np.clip(1.0 - median_tilt / 10.0, 0.0, 1.0))

    @staticmethod
    def _score_vignetting(gray: np.ndarray) -> float:
        """
        Rules1.10.1 – compare centre brightness vs. corner brightness.

        Returns 1.0 when the difference is small.
        """
        h, w = gray.shape
        ch, cw = h // 4, w // 4

        centre = gray[ch: h - ch, cw: w - cw]
        centre_mean = float(centre.mean())

        # average of four corner patches (10% of each dim)
        ph, pw = max(h // 10, 10), max(w // 10, 10)
        corners = [
            gray[:ph, :pw],
            gray[:ph, w - pw:],
            gray[h - ph:, :pw],
            gray[h - ph:, w - pw:],
        ]
        corner_mean = float(np.mean([c.mean() for c in corners]))

        if centre_mean < 1.0:
            return 1.0  # avoid div-by-zero on black frames
        ratio = corner_mean / centre_mean
        # severe vignetting: ratio < 0.5 → score 0 ; no vignetting: ratio ~1 → score 1
        return float(np.clip((ratio - 0.4) / 0.5, 0.0, 1.0))

    @staticmethod
    def _score_haze(bgr: np.ndarray) -> float:
        """
        Rules1.10.2 – simplified dark-channel prior for haze estimation.

        A hazy image has a high dark-channel value (>= ~100 in 0-255 range).
        Returns 1.0 when dark-channel mean is low (clear image).
        """
        # Dark channel: minimum across channels, then minimum in local patch
        dark = np.min(bgr, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(dark, kernel)
        dc_mean = float(dark_channel.mean())
        # clear image: dc_mean ~5-20 ; hazy: dc_mean ~60-130
        return float(np.clip(1.0 - (dc_mean - 10.0) / 100.0, 0.0, 1.0))

    @staticmethod
    def _score_purple_fringe(bgr: np.ndarray, gray: np.ndarray) -> float:
        """
        Rules1.10.3 – purple fringe detection.

        Measures the density of purple/violet pixels that coincide with
        high-contrast edges (typical signature of chromatic aberration).
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Purple hue in OpenCV HSV: roughly 120-160 (out of 180)
        purple_mask = (
            (h >= 120) & (h <= 160) &
            (s >= 40) &
            (v >= 40)
        ).astype(np.uint8)

        # Limit to edge regions
        edges = cv2.Canny(gray, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        edge_purple = purple_mask & (dilated_edges > 0)

        density = float(edge_purple.sum()) / max(float(dilated_edges.sum()), 1.0)
        # density > 0.05 starts to look like fringing
        return float(np.clip(1.0 - density / 0.05, 0.0, 1.0))

    # -----------------------------------------------------------------------
    # Rule mapping
    # -----------------------------------------------------------------------

    @staticmethod
    def _check_rules(result: OpenCVResult) -> None:
        """Populate result.violations based on computed scores."""
        v = result.violations

        # ---- Rules1.1.1  Blurry / out-of-focus ----------------------------
        if result.sharpness_score < _SHARPNESS_BLUR_THRESHOLD:
            severity = "critical" if result.sharpness_score < 0.10 else "major"
            v.append(RuleViolation(
                rule_id="Rules1.1.1",
                rule_name="模糊/虚焦",
                severity=severity,
                confidence=round(1.0 - result.sharpness_score, 3),
                description=(
                    f"图片清晰度不足（Laplacian 得分 {result.sharpness_score:.2f}），"
                    "飞机轮廓和细节无法清晰呈现"
                ),
                source="opencv",
            ))

        # ---- Rules1.1.2  Oversharpen / noisy --------------------------------
        if result.noise_score < _NOISE_OVER_THRESHOLD and result.sharpness_score > _SHARPNESS_OVER_THRESHOLD:
            v.append(RuleViolation(
                rule_id="Rules1.1.2",
                rule_name="锐化过度/噪点",
                severity="minor",
                confidence=round(1.0 - result.noise_score, 3),
                description=(
                    f"图片噪点明显（噪点得分 {result.noise_score:.2f}），"
                    "可能由高 ISO 或过度锐化导致"
                ),
                source="opencv",
            ))

        # ---- Rules1.2.1  Bad contrast ---------------------------------------
        if result.contrast_score < _CONTRAST_LOW_THRESHOLD:
            v.append(RuleViolation(
                rule_id="Rules1.2.1",
                rule_name="对比度不足",
                severity="minor",
                confidence=round(1.0 - result.contrast_score, 3),
                description=(
                    f"图片对比度过低（得分 {result.contrast_score:.2f}），"
                    "飞机主体与背景区分度不足"
                ),
                source="opencv",
            ))
        elif result.contrast_score < _CONTRAST_HIGH_THRESHOLD and result.contrast_score > 0.88:
            # near the upper boundary – flag as high contrast
            v.append(RuleViolation(
                rule_id="Rules1.2.1",
                rule_name="对比度过高",
                severity="minor",
                confidence=round(1.0 - result.contrast_score, 3),
                description=(
                    f"图片对比度过高（得分 {result.contrast_score:.2f}），"
                    "高光和阴影区域细节损失"
                ),
                source="opencv",
            ))

        # ---- Rules1.2.3  Overexposed / underexposed -------------------------
        if result.exposure_score < _EXPOSURE_DARK_THRESHOLD:
            severity = "critical" if result.exposure_score < 0.10 else "major"
            v.append(RuleViolation(
                rule_id="Rules1.2.3",
                rule_name="曝光异常",
                severity=severity,
                confidence=round(1.0 - result.exposure_score, 3),
                description=(
                    f"图片曝光异常（得分 {result.exposure_score:.2f}），"
                    "整体偏暗或偏亮，主体细节丢失"
                ),
                source="opencv",
            ))

        # ---- Rules1.3.1  Bad colour -----------------------------------------
        if result.color_score < _COLOR_SAT_THRESHOLD:
            v.append(RuleViolation(
                rule_id="Rules1.3.1",
                rule_name="颜色不佳",
                severity="minor",
                confidence=round(1.0 - result.color_score, 3),
                description=(
                    f"图片色彩饱和度过低（得分 {result.color_score:.2f}），"
                    "色彩偏差明显或过于不自然"
                ),
                source="opencv",
            ))

        # ---- Rules1.7.1  Horizon unlevel ------------------------------------
        if result.horizon_score < (1.0 - _HORIZON_TILT_DEG / 10.0):
            v.append(RuleViolation(
                rule_id="Rules1.7.1",
                rule_name="地平线不正",
                severity="minor",
                confidence=round(1.0 - result.horizon_score, 3),
                description=(
                    f"地平线倾斜（得分 {result.horizon_score:.2f}），"
                    f"未通过垂直参照物校正"
                ),
                source="opencv",
            ))

        # ---- Rules1.10.1  Vignetting ----------------------------------------
        if result.vignetting_score < _VIGNETTING_THRESHOLD:
            v.append(RuleViolation(
                rule_id="Rules1.10.1",
                rule_name="暗角",
                severity="minor",
                confidence=round(1.0 - result.vignetting_score, 3),
                description=(
                    f"图片四角存在明显暗角（得分 {result.vignetting_score:.2f}），"
                    "镜头光学素质导致，后期未改善"
                ),
                source="opencv",
            ))

        # ---- Rules1.10.2  Hazy / foggy --------------------------------------
        if result.haze_score < _HAZE_THRESHOLD:
            v.append(RuleViolation(
                rule_id="Rules1.10.2",
                rule_name="薄雾/能见度不佳",
                severity="minor",
                confidence=round(1.0 - result.haze_score, 3),
                description=(
                    f"图片存在雾霾或能见度不足（得分 {result.haze_score:.2f}），"
                    "天气原因导致，后期未改善"
                ),
                source="opencv",
            ))

        # ---- Rules1.10.3  Purple fringe -------------------------------------
        if result.purple_fringe_score < _PURPLE_FRINGE_THRESHOLD:
            v.append(RuleViolation(
                rule_id="Rules1.10.3",
                rule_name="紫边",
                severity="minor",
                confidence=round(1.0 - result.purple_fringe_score, 3),
                description=(
                    f"图片存在紫边（得分 {result.purple_fringe_score:.2f}），"
                    "镜头色散导致，后期未改善"
                ),
                source="opencv",
            ))
