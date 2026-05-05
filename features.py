from __future__ import annotations

import numpy as np
import scipy.signal

from audio_io import LIBROSA_AVAILABLE
if LIBROSA_AVAILABLE:
    import librosa
    import librosa.feature
    import librosa.onset

from config import (
    CWT_MAX_SAMPLES,
    CWT_MORLET_W,
    CWT_N_SCALES,
    N_MELS,
    N_MFCC,
    STFT_N_FFT_MAX,
    STFT_N_FFT_MIN,
    W_CHROMA,
    W_CWT_MAG,
    W_MEL,
    W_MFCC,
    W_PHASE_CENTROID,
    W_PHASE_CWT,
    W_PHASE_ONSET,
    W_PHASE_STFT_1024,
    W_PHASE_STFT_512,
    W_PHASE_ZCR,
    W_RMS,
    W_STFT_TOTAL,
)
from utils import get_param, normalize_positive_weights


# ============================================================
# SciPy-compatible CWT helpers
# (scipy.signal.cwt, morlet2, and ricker were removed in SciPy 1.15)
# ============================================================

def morlet2_compat(M: int, s: float, w: float = CWT_MORLET_W) -> np.ndarray:
    """
    Return a complex Morlet wavelet compatible with the removed scipy.signal.morlet2 API.

    The Morlet wavelet is:
        ψ(t) = π^{-1/4} · s^{-1/2} · exp(jωt/s) · exp(−t²/(2s²))
    where ω = w (central frequency parameter, default 6.0).
    """
    M = int(max(1, M))
    s = float(max(1e-12, s))
    x = np.arange(M, dtype=np.float64) - (M - 1.0) / 2.0
    x_scaled = x / s
    return (np.pi ** -0.25) * np.sqrt(1.0 / s) * np.exp(1j * w * x_scaled) * np.exp(-0.5 * x_scaled ** 2)


def ricker_compat(M: int, a: float) -> np.ndarray:
    """Return a Ricker (Mexican-hat) wavelet compatible with scipy.signal.ricker."""
    M = int(max(1, M))
    a = float(max(1e-12, a))
    x   = np.arange(M, dtype=np.float64) - (M - 1.0) / 2.0
    xsq = x ** 2
    asq = a ** 2
    amplitude = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    return amplitude * (1.0 - xsq / asq) * np.exp(-xsq / (2.0 * asq))


def cwt_compat(data: np.ndarray, wavelet_fn, widths: np.ndarray) -> np.ndarray:
    """
    Local replacement for the removed scipy.signal.cwt.

    For each width, the signal is convolved with the time-reversed complex
    conjugate of the corresponding wavelet, matching the behavior documented
    for scipy.signal.cwt before its removal.
    """
    data   = np.asarray(data)
    widths = np.asarray(widths, dtype=np.float64)
    outputs = []
    for width in widths:
        wavelet_length = min(int(np.ceil(10.0 * width)), data.size)
        wavelet_length = max(1, wavelet_length)
        wavelet = np.asarray(wavelet_fn(wavelet_length, width))
        coeff = scipy.signal.convolve(
            data, np.conj(wavelet)[::-1], mode="same", method="auto",
        )
        outputs.append(coeff)
    return np.asarray(outputs)


# ============================================================
# STFT resolution list
# ============================================================

def get_stft_resolutions(n_samples: int, params: dict | None = None) -> list[int]:
    """
    Return valid powers-of-two STFT window sizes under user-controlled bounds.

    The returned list contains every power of two k in [stft_min, upper] where
    upper = min(stft_max, n_samples // 2). This guarantees that each window fits
    at least twice inside the signal.
    """
    stft_min = int(get_param(params, "stft_n_fft_min", STFT_N_FFT_MIN))
    stft_max = int(get_param(params, "stft_n_fft_max", STFT_N_FFT_MAX))

    stft_min = int(2 ** round(np.log2(max(64, stft_min))))
    stft_max = int(2 ** round(np.log2(max(stft_min, stft_max))))

    upper = min(stft_max, max(stft_min, n_samples // 2))
    resolutions = [2 ** k for k in range(6, 15) if stft_min <= 2 ** k <= upper]
    return resolutions if resolutions else [stft_min]


# ============================================================
# Weight accessors
# ============================================================

def get_magnitude_weights(params: dict | None) -> dict[str, float]:
    """Return normalized user-controlled magnitude feature weights."""
    defaults = {
        "stft": W_STFT_TOTAL, "cwt": W_CWT_MAG, "mel": W_MEL,
        "chroma": W_CHROMA, "mfcc": W_MFCC, "rms": W_RMS,
    }
    values = {
        "stft":   get_param(params, "mag_weight_stft",   W_STFT_TOTAL),
        "cwt":    get_param(params, "mag_weight_cwt",    W_CWT_MAG),
        "mel":    get_param(params, "mag_weight_mel",    W_MEL),
        "chroma": get_param(params, "mag_weight_chroma", W_CHROMA),
        "mfcc":   get_param(params, "mag_weight_mfcc",   W_MFCC),
        "rms":    get_param(params, "mag_weight_rms",    W_RMS),
    }
    return normalize_positive_weights(values, defaults)


def get_phase_weights(params: dict | None, has_cwt_phase: bool) -> dict[str, float]:
    """
    Return normalized user-controlled phase feature weights.

    When has_cwt_phase is False (Ricker wavelet), the CWT weight is
    redistributed proportionally to the two STFT phase sources.
    """
    defaults = {
        "stft_mid": W_PHASE_STFT_1024, "stft_fine": W_PHASE_STFT_512,
        "cwt":      W_PHASE_CWT,       "onset":     W_PHASE_ONSET,
        "centroid": W_PHASE_CENTROID,   "zcr":       W_PHASE_ZCR,
    }
    values = {
        "stft_mid":  get_param(params, "phase_weight_stft_mid",  W_PHASE_STFT_1024),
        "stft_fine": get_param(params, "phase_weight_stft_fine", W_PHASE_STFT_512),
        "cwt":       get_param(params, "phase_weight_cwt",       W_PHASE_CWT),
        "onset":     get_param(params, "phase_weight_onset",     W_PHASE_ONSET),
        "centroid":  get_param(params, "phase_weight_centroid",  W_PHASE_CENTROID),
        "zcr":       get_param(params, "phase_weight_zcr",       W_PHASE_ZCR),
    }

    if not has_cwt_phase:
        cwt_weight  = max(0.0, float(values.get("cwt", 0.0)))
        values["cwt"] = 0.0
        stft_total  = max(1e-12, max(0.0, values["stft_mid"]) + max(0.0, values["stft_fine"]))
        values["stft_mid"]  += cwt_weight * max(0.0, values["stft_mid"])  / stft_total
        values["stft_fine"] += cwt_weight * max(0.0, values["stft_fine"]) / stft_total

    return normalize_positive_weights(values, defaults)


def get_rgb_bands(params: dict | None) -> list[tuple[float, float]]:
    """Return user-controlled low/mid/high frequency bands for RGB synthesis."""
    low_end    = float(get_param(params, "rgb_low_end",    1.0 / 3.0))
    high_start = float(get_param(params, "rgb_high_start", 2.0 / 3.0))
    low_end    = float(np.clip(low_end,    0.05, 0.90))
    high_start = float(np.clip(high_start, 0.10, 0.95))
    if high_start <= low_end + 0.05:
        mid        = 0.5 * (low_end + high_start)
        low_end    = max(0.05, mid - 0.025)
        high_start = min(0.95, mid + 0.025)
    return [(0.0, low_end), (low_end, high_start), (high_start, 1.0)]


def apply_rgb_balance(image: np.ndarray, params: dict | None) -> np.ndarray:
    """Apply user-controlled per-channel RGB gains."""
    gains = np.array([
        float(get_param(params, "rgb_balance_r", 1.0)),
        float(get_param(params, "rgb_balance_g", 1.0)),
        float(get_param(params, "rgb_balance_b", 1.0)),
    ], dtype=np.float64)
    return np.asarray(image, dtype=np.float64) * gains.reshape(1, 1, 3)


def apply_global_image_adjustments(
    image: np.ndarray, params: dict | None, is_grayscale: bool = False
) -> np.ndarray:
    """Apply brightness, contrast, gamma and saturation after global normalization."""
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)

    contrast   = float(get_param(params, "contrast_strength",  1.0))
    brightness = float(get_param(params, "brightness_factor",  1.0))
    gamma      = float(get_param(params, "gamma_correction",   0.85))
    saturation = float(get_param(params, "saturation_factor",  1.0))

    img = np.clip(0.5 + contrast * (img - 0.5), 0.0, 1.0) * brightness
    img = np.clip(img, 0.0, 1.0)

    if gamma > 1e-6:
        img = img ** gamma

    if not is_grayscale and img.ndim == 3 and img.shape[2] == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        img  = gray[:, :, None] + saturation * (img - gray[:, :, None])

    return np.clip(img, 0.0, 1.0)


# ============================================================
# Feature extraction
# ============================================================

def extract_features(
    waveform: np.ndarray,
    sr: int,
    wavelet_type: str = "Morlet",
    params: dict | None = None,
    step_callback=None,
) -> dict:
    """
    Extract a comprehensive feature set from a mono waveform.

    The features are grouped into four categories, all of which contribute to
    the 2D Fourier magnitude and phase grids.

    Multi-resolution STFT magnitudes and phases:
        A separate STFT is computed for each power-of-two window size between
        STFT_N_FFT_MIN and min(STFT_N_FFT_MAX, len(waveform) // 2).
        Hop length = n_fft // 4 (75% overlap, standard for spectral analysis).
        Using multiple resolutions simultaneously avoids the single-resolution
        Heisenberg trade-off: fine-time STFTs resolve onsets; fine-frequency
        STFTs resolve harmonic partials.

    Continuous Wavelet Transform (CWT):
        The waveform is downsampled to CWT_MAX_SAMPLES samples before
        transform to bound computation time (O(N · S · max_scale)).
        Morlet (analytic, complex): both magnitude and instantaneous phase are
            extracted.  The Morlet wavelet ψ(t) = π^{-1/4} · exp(jω₀t) ·
            exp(−t²/2) with ω₀ = CWT_MORLET_W provides simultaneous time and
            frequency localization, optimal under the uncertainty principle.
        Ricker (real, symmetric): only magnitude is extracted (no analytic
            signal → no meaningful instantaneous phase). The Ricker wavelet
            is the second derivative of a Gaussian; it is optimal for detecting
            transients and sharp spectral peaks.
        Scales are log-spaced from 1 to min(512, N_cwt // 2) to cover the
        full dynamic range of the signal uniformly on a logarithmic axis.

    Perceptual representations:
        Mel spectrogram (N_MELS = 128 filters): maps the linear frequency
            axis to the mel scale, a perceptually uniform scale approximating
            the logarithmic pitch perception of the cochlea above ≈ 1 000 Hz.
        Chroma (12 bins): sums energy across all octaves for each pitch class
            (C, C#, …, B), capturing harmonic content independently of octave.
        MFCC (N_MFCC = 20 coefficients): the discrete cosine transform of the
            log mel spectrogram. Coefficients encode the spectral envelope
            (timbre) in a compact decorrelated representation.

    Temporal descriptors (1-D over time):
        spectral_centroid: frequency-weighted mean of the power spectrum,
            correlated with perceived brightness.
        spectral_bandwidth: standard deviation of the power spectrum around
            the centroid, correlated with spectral spread.
        spectral_rolloff: the frequency below which 85% of total spectral
            energy is concentrated; a summary statistic for high-frequency content.
        spectral_flatness: ratio of geometric to arithmetic mean of the power
            spectrum (Wiener entropy). Near 0 for tonal signals; near 1 for
            white noise.
        rms: root mean square frame energy; captures loudness dynamics.
        zcr: zero-crossing rate; high for noisy/percussive signals, low for
            tonal/smooth signals.
        onset_strength: the mean of the positive first-order difference of
            the log mel spectrogram. Peaks sharply at note onsets and
            rhythmic events.

    Parameters:
        waveform:     (N,) float32 mono signal
        sr:           sample rate in Hz
        wavelet_type: "Morlet" or "Ricker (Mexican hat)"

    Returns:
        dict mapping feature names to numpy arrays, plus the list of STFT
        resolutions used under the key "stft_resolutions"
    """
    features: dict = {}
    hop_default = 256   # shared hop length for perceptual features (≈11.6 ms at 22 050 Hz)

    # --- Multi-resolution STFT ---
    resolutions = get_stft_resolutions(len(waveform), params=params)
    features["stft_resolutions"] = resolutions

    for n_fft in resolutions:
        if step_callback is not None:
            step_callback(f"STFT  N={n_fft}")
        hop  = n_fft // 4
        stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop, window="hann", center=True)
        features[f"mag_{n_fft}"]   = np.abs(stft)
        features[f"phase_{n_fft}"] = np.angle(stft)

    # --- CWT ---
    cwt_max_samples = max(512, int(get_param(params, "cwt_max_samples", CWT_MAX_SAMPLES)))
    cwt_n_scales    = max(8,   int(get_param(params, "cwt_n_scales",    CWT_N_SCALES)))
    n_mels          = max(16,  int(get_param(params, "n_mels",          N_MELS)))
    n_mfcc          = max(4,   int(get_param(params, "n_mfcc",          N_MFCC)))

    step_cwt    = max(1, len(waveform) // cwt_max_samples)
    waveform_cwt = waveform[::step_cwt].copy()
    n_cwt        = len(waveform_cwt)

    if step_callback is not None:
        step_callback(f"CWT  ({wavelet_type},  S={cwt_n_scales})")

    max_scale  = min(512, n_cwt // 2)
    cwt_scales = np.geomspace(1.0, max(1.0, float(max_scale)), num=cwt_n_scales)

    if wavelet_type == "Morlet":
        wavelet_fn = lambda M, s: morlet2_compat(M, s, w=CWT_MORLET_W)
    else:
        wavelet_fn = ricker_compat

    cwt_coeffs = cwt_compat(waveform_cwt, wavelet_fn, cwt_scales)

    features["cwt_magnitude"] = np.abs(cwt_coeffs)
    features["cwt_phase"]     = np.angle(cwt_coeffs) if wavelet_type == "Morlet" else None

    # --- Perceptual features ---
    if step_callback is not None:
        step_callback(f"Mel spectrogram  (B={n_mels})")
    features["mel"] = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, hop_length=hop_default,
    )

    if step_callback is not None:
        step_callback("Chroma")
    features["chroma"] = librosa.feature.chroma_stft(
        y=waveform, sr=sr, hop_length=hop_default,
    )

    if step_callback is not None:
        step_callback(f"MFCC  (C={n_mfcc})")
    features["mfcc"] = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=n_mfcc, hop_length=hop_default,
    )

    # --- Temporal descriptors ---
    if step_callback is not None:
        step_callback("Temporal descriptors")
    features["spectral_centroid"]  = librosa.feature.spectral_centroid(y=waveform, sr=sr, hop_length=hop_default)[0]
    features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(y=waveform, sr=sr, hop_length=hop_default)[0]
    features["spectral_rolloff"]   = librosa.feature.spectral_rolloff(y=waveform, sr=sr, hop_length=hop_default)[0]
    features["spectral_flatness"]  = librosa.feature.spectral_flatness(y=waveform, hop_length=hop_default)[0]
    features["rms"]                = librosa.feature.rms(y=waveform, hop_length=hop_default)[0]
    features["zcr"]                = librosa.feature.zero_crossing_rate(waveform, hop_length=hop_default)[0]
    features["onset_strength"]     = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop_default)

    if step_callback is not None:
        step_callback("Building Fourier grids")

    return features
