from __future__ import annotations

import scipy.interpolate
import scipy.ndimage
import numpy as np


# ============================================================
# Normalization helpers
# ============================================================

def normalize_to_unit(array: np.ndarray) -> np.ndarray:
    """Linearly map all values to [0, 1]; return a zero array if constant."""
    a_min = array.min()
    a_max = array.max()
    if a_max > a_min:
        return (array - a_min) / (a_max - a_min)
    return np.zeros_like(array, dtype=np.float32)


def normalize_to_unit_robust(
    array: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Robustly map values to [0, 1] using global percentiles."""
    arr = np.asarray(array, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo = float(np.percentile(finite, lower_percentile))
    hi = float(np.percentile(finite, upper_percentile))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ============================================================
# Parameter accessor
# ============================================================

def get_param(params: dict | None, key: str, default):
    """Read a user parameter with a safe fallback."""
    if params is None:
        return default
    return params.get(key, default)


def normalize_positive_weights(
    weight_dict: dict[str, float],
    fallback: dict[str, float],
) -> dict[str, float]:
    """Normalize non-negative weights; fall back to defaults if their sum is zero."""
    cleaned = {k: max(0.0, float(v)) for k, v in weight_dict.items()}
    total = sum(cleaned.values())
    if total <= 1e-12:
        cleaned = {k: max(0.0, float(v)) for k, v in fallback.items()}
        total = sum(cleaned.values())
    if total <= 1e-12:
        n = max(1, len(cleaned))
        return {k: 1.0 / n for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


# ============================================================
# Spatial / interpolation utilities
# ============================================================

def interpolate_to_shape(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resample a 2D array to (target_h, target_w) via bicubic spline interpolation.

    Both source and destination grids are normalized to [0, 1] so the
    resampling is scale-invariant. This handles arbitrary upsampling and
    downsampling ratios in both dimensions without explicit padding or trimming.
    """
    src_h, src_w = array.shape

    if src_h < 2 or src_w < 2:
        return np.resize(array, (target_h, target_w))

    y_src = np.linspace(0.0, 1.0, src_h)
    x_src = np.linspace(0.0, 1.0, src_w)
    y_dst = np.linspace(0.0, 1.0, target_h)
    x_dst = np.linspace(0.0, 1.0, target_w)

    ky = min(3, src_h - 1)
    kx = min(3, src_w - 1)

    interp = scipy.interpolate.RectBivariateSpline(y_src, x_src, array, kx=kx, ky=ky)
    return interp(y_dst, x_dst)


def row_to_2d(row: np.ndarray, target_size: int) -> np.ndarray:
    """
    Stretch a 1-D temporal feature vector into a square (N, N) spatial map.

    The vector is normalized to [0, 1], interpolated to N points, then
    replicated across N rows. The result encodes temporal evolution as a
    horizontally-varying, vertically-uniform pattern in the spatial grid.
    """
    row_norm   = normalize_to_unit(row)
    row_interp = interpolate_to_shape(row_norm.reshape(1, -1), 1, target_size)[0]
    return np.tile(row_interp, (target_size, 1))


def slice_band(array: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """
    Extract a fractional sub-band from a 2-D spectrogram along the frequency axis.

    Parameters:
        array: (K, T) spectrogram, K = frequency bins
        band:  (lo, hi) normalized fractions in [0, 1]

    Returns:
        Sub-array of shape (hi_bin - lo_bin, T)
    """
    k  = array.shape[0]
    lo = int(round(band[0] * k))
    hi = max(lo + 1, int(round(band[1] * k)))
    return array[lo:hi, :]


def resize_float_image_to_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a floating-point image to an arbitrary size with bicubic interpolation."""
    width  = max(1, int(width))
    height = max(1, int(height))
    image  = np.asarray(image, dtype=np.float64)
    if image.shape[0] == height and image.shape[1] == width:
        return image

    zoom_y = height / max(1, image.shape[0])
    zoom_x = width  / max(1, image.shape[1])
    if image.ndim == 2:
        resized = scipy.ndimage.zoom(image, (zoom_y, zoom_x), order=3)
    else:
        resized = scipy.ndimage.zoom(image, (zoom_y, zoom_x, 1.0), order=3)

    return resized[:height, :width].astype(np.float64)


def resize_float_image_to_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a floating-point image to a square of given side length."""
    size = max(1, int(size))
    return resize_float_image_to_size(image, size, size)
