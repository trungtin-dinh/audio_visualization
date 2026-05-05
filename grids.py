from __future__ import annotations

import numpy as np

from utils import (
    interpolate_to_shape,
    normalize_to_unit,
    row_to_2d,
    slice_band,
)
from features import get_magnitude_weights, get_phase_weights, get_rgb_bands


# ============================================================
# Magnitude grid
# ============================================================

def build_magnitude_grid(
    features: dict,
    target_size: int,
    band: tuple[float, float] | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """
    Build a (N, N) Fourier magnitude grid from user-weighted audio features.

    Each feature is log-compressed, normalized to [0, 1], and bilinearly
    resampled to (N, N) before being blended by the weight vector returned
    by get_magnitude_weights(). The STFT weight is split equally among all
    active STFT resolutions.
    """
    weights     = get_magnitude_weights(params)
    resolutions = features["stft_resolutions"]
    w_per_stft  = weights["stft"] / max(1, len(resolutions))
    components: list[tuple[float, np.ndarray]] = []

    for n_fft in resolutions:
        mag = features[f"mag_{n_fft}"]
        if band is not None:
            mag = slice_band(mag, band)
        log_mag = normalize_to_unit(np.log1p(mag))
        components.append((w_per_stft, interpolate_to_shape(log_mag, target_size, target_size)))

    cwt_mag = features["cwt_magnitude"]
    if band is not None:
        cwt_mag = slice_band(cwt_mag, band)
    components.append((weights["cwt"], interpolate_to_shape(normalize_to_unit(np.log1p(cwt_mag)), target_size, target_size)))

    mel = features["mel"]
    if band is not None:
        mel = slice_band(mel, band)
    components.append((weights["mel"], interpolate_to_shape(normalize_to_unit(np.log1p(mel)), target_size, target_size)))

    chroma = normalize_to_unit(np.abs(features["chroma"]))
    components.append((weights["chroma"], interpolate_to_shape(chroma, target_size, target_size)))

    mfcc = normalize_to_unit(np.abs(features["mfcc"]))
    components.append((weights["mfcc"], interpolate_to_shape(mfcc, target_size, target_size)))

    rms_2d = row_to_2d(features["rms"], target_size)
    components.append((weights["rms"], rms_2d))

    combined = sum(w * arr for w, arr in components)
    return normalize_to_unit(combined)


# ============================================================
# Phase grid
# ============================================================

def build_phase_grid(
    features: dict,
    target_size: int,
    band: tuple[float, float] | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """
    Build a (N, N) Fourier phase grid from user-weighted audio features.

    STFT phases are unwrapped along both axes before interpolation to prevent
    phase-wrapping discontinuities from producing high-frequency spatial
    artefacts. The final grid is wrapped back into (−π, π].
    """
    resolutions   = features["stft_resolutions"]
    has_cwt_phase = features.get("cwt_phase") is not None
    weights       = get_phase_weights(params, has_cwt_phase=has_cwt_phase)

    n_fft_mid  = 1024 if 1024 in resolutions else resolutions[len(resolutions) // 2]
    phase_mid  = features[f"phase_{n_fft_mid}"]
    if band is not None:
        phase_mid = slice_band(phase_mid, band)
    grid_mid = interpolate_to_shape(np.unwrap(np.unwrap(phase_mid, axis=0), axis=1), target_size, target_size)

    n_fft_fine  = 512 if 512 in resolutions else resolutions[0]
    phase_fine  = features[f"phase_{n_fft_fine}"]
    if band is not None:
        phase_fine = slice_band(phase_fine, band)
    grid_fine = interpolate_to_shape(np.unwrap(np.unwrap(phase_fine, axis=0), axis=1), target_size, target_size)

    if has_cwt_phase:
        cwt_phase = features["cwt_phase"]
        if band is not None:
            cwt_phase = slice_band(cwt_phase, band)
        grid_cwt = interpolate_to_shape(np.unwrap(np.unwrap(cwt_phase, axis=0), axis=1), target_size, target_size)
    else:
        grid_cwt = np.zeros((target_size, target_size), dtype=np.float64)

    onset_2d    = row_to_2d(features["onset_strength"],   target_size) * (np.pi / 2.0)
    centroid_2d = row_to_2d(features["spectral_centroid"], target_size) * np.pi
    zcr_2d      = row_to_2d(features["zcr"],               target_size) * (np.pi / 4.0)

    phase_combined = (
        weights["stft_mid"]  * grid_mid
        + weights["stft_fine"] * grid_fine
        + weights["cwt"]       * grid_cwt
        + weights["onset"]     * onset_2d
        + weights["centroid"]  * centroid_2d
        + weights["zcr"]       * zcr_2d
    )

    return (phase_combined + np.pi) % (2.0 * np.pi) - np.pi


# ============================================================
# Hermitian symmetry enforcement
# ============================================================

def enforce_hermitian_symmetry(Z: np.ndarray) -> np.ndarray:
    """
    Project a complex (N, N) array onto the subspace of Hermitian-symmetric
    matrices, guaranteeing that its 2D inverse DFT is real-valued.

    A 2D DFT F of a real-valued image f satisfies:

        F[m, n] = conj(F[(-m) mod N, (-n) mod N])   ∀ m, n

    For an arbitrary complex Z, the unique nearest Hermitian-symmetric
    matrix in the Frobenius norm is:

        Z_sym[m, n] = (Z[m, n] + conj(Z[(-m) mod N, (-n) mod N])) / 2

    Computation:
        1. Z_f[m, n] = Z[N−1−m, N−1−n]         (flip both axes)
        2. Z_r = roll(roll(Z_f, 1, axis=0), 1, axis=1)
           → Z_r[m, n] = Z[(-m) mod N, (-n) mod N]
        3. Z_sym = (Z + conj(Z_r)) / 2

    Verification:
        DC (m=n=0): Z_sym[0,0] = (Z[0,0] + conj(Z[0,0])) / 2 = Re(Z[0,0]) ∈ ℝ ✓
        Pair (m,n): Z_sym[-m,-n] = (Z[-m,-n] + conj(Z[m,n])) / 2
                                  = conj(Z_sym[m,n]) ✓

    After symmetrization, Im(IFFT2(Z_sym)) is at floating-point precision only.
    """
    Z_r = np.roll(np.roll(Z[::-1, ::-1], 1, axis=0), 1, axis=1)
    return (Z + np.conj(Z_r)) / 2.0


# ============================================================
# Image reconstruction from magnitude and phase grids
# ============================================================

def reconstruct_channel_raw_and_spectrum(
    magnitude_2d: np.ndarray,
    phase_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct one spatial channel without local min-max normalization.

    This variant is used for section-by-section synthesis so that all blocks
    can be normalized only once at the final-image level.

    Returns:
        f:     (N, N) float64 unnormalized spatial channel (Re(IFFT2(Z_sym)))
        Z_sym: (N, N) Hermitian-symmetric complex spectrum passed to IFFT2
    """
    Z     = magnitude_2d * np.exp(1j * phase_2d)
    Z_sym = enforce_hermitian_symmetry(Z)
    f     = np.real(np.fft.ifft2(Z_sym))
    return f.astype(np.float64), Z_sym


def reconstruct_channel_and_spectrum(
    magnitude_2d: np.ndarray,
    phase_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct one spatial channel and return the exact Hermitian spectrum
    used by np.fft.ifft2.

    Pipeline:
        1. Z[u,v] = M̃[u,v] · exp(j · Φ̃[u,v])
        2. Z_sym  = enforce_hermitian_symmetry(Z)
        3. f[x,y] = IFFT2(Z_sym)
        4. Return normalize_to_unit(Re(f)) and Z_sym

    Parameters:
        magnitude_2d: (N, N) float array in [0, 1]
        phase_2d:     (N, N) float array in (−π, π]

    Returns:
        channel: (N, N) float32 array in [0, 1]
        Z_sym:   (N, N) complex Hermitian spectrum passed to np.fft.ifft2
    """
    Z     = magnitude_2d * np.exp(1j * phase_2d)
    Z_sym = enforce_hermitian_symmetry(Z)
    f     = np.real(np.fft.ifft2(Z_sym))
    return normalize_to_unit(f), Z_sym


def spectrum_to_centered_magnitude_phase(
    Z_sym: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the exact spectrum used by np.fft.ifft2 into display grids.

    np.fft.ifft2 expects the DC/origin coefficient at index (0, 0). For a
    standard Fourier visualization, the origin is moved to the center with
    fftshift. The returned magnitude and phase therefore show the actual
    Hermitian-symmetrized spectrum used for reconstruction, but displayed in
    the conventional centered layout.
    """
    Z_centered         = np.fft.fftshift(Z_sym)
    magnitude_centered = np.abs(Z_centered)
    phase_centered     = np.angle(Z_centered)
    return magnitude_centered, phase_centered


# ============================================================
# Unnormalized multi-channel patch for sectioned synthesis
# ============================================================

def audio_to_image_float(
    features: dict,
    target_size: int,
    output_mode: str,
    params: dict | None = None,
) -> np.ndarray:
    """
    Generate an unnormalized floating-point image patch from extracted features.

    Returns a (N, N, 3) float64 array. Global normalization is deferred to
    finalize_sectioned_image() so that all section patches share the same scale.
    """
    if output_mode == "Grayscale":
        mag_2d   = build_magnitude_grid(features, target_size, band=None, params=params)
        phase_2d = build_phase_grid(features, target_size, band=None, params=params)
        channel, _ = reconstruct_channel_raw_and_spectrum(mag_2d, phase_2d)
        return np.stack([channel, channel, channel], axis=2)

    channels = []
    for band in get_rgb_bands(params):
        mag_2d   = build_magnitude_grid(features, target_size, band=band, params=params)
        phase_2d = build_phase_grid(features, target_size, band=band, params=params)
        channel, _ = reconstruct_channel_raw_and_spectrum(mag_2d, phase_2d)
        channels.append(channel)

    return np.stack(channels, axis=2)
