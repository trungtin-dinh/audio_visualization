from __future__ import annotations

import io

import matplotlib.cm as matplotlib_cm
import numpy as np
from PIL import Image

from audio_io import LIBROSA_AVAILABLE
if LIBROSA_AVAILABLE:
    import librosa

from utils import interpolate_to_shape, normalize_to_unit
from grids import reconstruct_channel_and_spectrum, spectrum_to_centered_magnitude_phase


# ============================================================
# Colormap application
# ============================================================

def apply_colormap(values: np.ndarray, colormap_name: str) -> np.ndarray:
    """Apply a matplotlib colormap to a (H, W) float array in [0, 1].

    Returns:
        (H, W, 3) uint8 RGB image
    """
    cmap = matplotlib_cm.get_cmap(colormap_name)
    rgba = cmap(values)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


# ============================================================
# Audio signal visualizations
# ============================================================

def waveform_to_display_image(waveform: np.ndarray, width: int = 512, height: int = 80) -> np.ndarray:
    """
    Render the waveform as an (H, W, 3) uint8 oscillogram on a dark background.

    Each pixel column x corresponds to one downsampled time index.
    A vertical bar is drawn from the zero-amplitude midline to the normalized
    amplitude value, producing a standard time-domain plot.
    """
    canvas   = np.full((height, width, 3), 25, dtype=np.uint8)
    x_idx    = np.linspace(0, len(waveform) - 1, width).astype(int)
    samples  = waveform[x_idx]
    max_amp  = np.abs(waveform).max()
    amp_norm = samples / max(1e-8, max_amp)
    mid      = height // 2
    y_coords = np.clip((mid - amp_norm * (height // 2 - 4)).astype(int), 0, height - 1)
    for x_px, y_px in enumerate(y_coords):
        canvas[min(mid, y_px):max(mid, y_px) + 1, x_px] = [80, 160, 255]
    return canvas


def spectrogram_to_display_image(
    waveform: np.ndarray, width: int = 512, height: int = 160
) -> np.ndarray:
    """
    Compute and render a log-magnitude spectrogram as an (H, W, 3) uint8 image.

    Uses n_fft = 1024 (hop = 256) with a Hann window: a general-purpose
    trade-off between time and frequency resolution for display.
    Frequency axis is flipped so low frequencies appear at the bottom,
    matching standard spectrogram conventions.
    """
    if not LIBROSA_AVAILABLE:
        return np.zeros((height, width, 3), dtype=np.uint8)
    stft    = librosa.stft(waveform, n_fft=1024, hop_length=256, window="hann")
    log_mag = normalize_to_unit(np.log1p(np.abs(stft)))[::-1, :]
    return apply_colormap(interpolate_to_shape(log_mag, height, width), "inferno")


# ============================================================
# Fourier display helpers
# ============================================================

def fourier_grid_to_display_image(
    grid: np.ndarray, colormap: str, width: int = 512, height: int = 512
) -> np.ndarray:
    """
    Render a 2D Fourier grid (magnitude or phase) as an (H, W, 3) uint8 image.

    For phase grids in (−π, π], the field is normalized to [0, 1] before
    applying the colormap (circular colormaps such as "twilight" are recommended).
    For magnitude grids in [0, 1], the field is passed through directly.
    """
    normalized = normalize_to_unit(grid)
    return apply_colormap(interpolate_to_shape(normalized, height, width), colormap)


def output_image_fourier_to_display_images(
    image_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the classical 2D Fourier transform of the final output image.

    For RGB outputs, the transform is computed on a luminance image so that the
    displayed magnitude and phase summarize the spatial structure of the final
    image itself, not the local spectra used internally for block generation.

    Returns:
        magnitude display image  (H, W, 3) uint8, colormap "viridis"
        phase display image      (H, W, 3) uint8, colormap "twilight"
    """
    image = np.asarray(image_rgb, dtype=np.float64)
    gray  = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
             if image.ndim == 3 else image)
    gray  = normalize_to_unit(gray)
    F         = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(F))
    phase     = np.angle(F)
    return (
        fourier_grid_to_display_image(magnitude, "viridis"),
        fourier_grid_to_display_image(phase,     "twilight"),
    )


# ============================================================
# PNG encoding
# ============================================================

def image_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    """Encode an RGB image as PNG bytes for Streamlit download buttons."""
    buffer = io.BytesIO()
    Image.fromarray(image_rgb.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()
