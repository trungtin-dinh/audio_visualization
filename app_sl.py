from __future__ import annotations

import html
import io
import re
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import matplotlib.cm as matplotlib_cm
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")


# ============================================================
# Optional imports
# ============================================================

try:
    import librosa
    import librosa.feature
    import librosa.onset
    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    LIBROSA_AVAILABLE = False


# ============================================================
# Constants
# ============================================================

# DEFAULT_AUDIO_SR = 22 050 Hz: standard mono sample rate. The Nyquist
# frequency sr/2 = 11 025 Hz covers the full perceptual range of music and
# speech. Higher rates (e.g. 44 100 Hz) would double memory usage and
# compute time with no audible benefit after downmixing to mono.
DEFAULT_AUDIO_SR: int = 22_050

# MAX_RECORD_SECONDS = 60: caps the in-browser recording at 60 × 22 050 =
# 1 323 000 samples. Beyond this, feature extraction would take tens of
# seconds on a single CPU core, exceeding user-interaction latency budgets.
MAX_RECORD_SECONDS: int = 60

# STFT_N_FFT_MIN = 256: smallest DFT window. Frequency resolution =
# sr / N = 22 050 / 256 ≈ 86 Hz per bin. Smaller windows would make
# individual semitones (≈ 26 Hz at A4) unresolvable.
STFT_N_FFT_MIN: int = 256

# STFT_N_FFT_MAX = 8 192: largest DFT window. Frequency resolution ≈ 2.7 Hz
# per bin. Larger windows require a signal at least 8 192 samples long
# (≈ 0.37 s) to produce a single frame; beyond this the temporal axis of
# the spectrogram collapses.
STFT_N_FFT_MAX: int = 8_192

# CWT_N_SCALES = 64: 64 log-spaced scales. On a geometric grid from scale 1
# to scale 512, 64 steps span log2(512) = 9 octaves with one scale per
# ~0.14 octave, matching the frequency resolution of the mel filterbank.
CWT_N_SCALES: int = 64

# CWT_MORLET_W = 6.0: Morlet central frequency parameter. For w ≥ 5 the
# DC leak is exp(−w²/2) ≤ exp(−12.5) ≈ 3.7×10⁻⁶, making the wavelet
# analytically valid for instantaneous phase computation. w = 6 is the
# conventional choice in neuroscience and audio analysis literature.
CWT_MORLET_W: float = 6.0

# CWT_MAX_SAMPLES = 44 100: the waveform is downsampled to this many samples
# before CWT. At DEFAULT_AUDIO_SR this equals 2 s of audio.  CWT cost is
# O(N · S · max_scale), so halving N reduces cost by ~2×. Two seconds
# is sufficient to capture the broad spectro-temporal envelope that CWT
# targets; fine temporal detail is already covered by the STFTs.
CWT_MAX_SAMPLES: int = 44_100

# N_MELS = 128: matches the number of auditory critical bands between 0 Hz
# and sr/2 Hz. The mel scale compresses the frequency axis logarithmically
# above ≈ 1 000 Hz, reflecting the reduced pitch discrimination of the
# cochlea at high frequencies. 128 bins is the empirical standard in
# music information retrieval.
N_MELS: int = 128

# N_MFCC = 20: cepstral liftering studies show that MFCC coefficients 1–13
# capture speech timbre and 14–20 add fine texture; beyond 20 the marginal
# information content per coefficient decays rapidly and becomes
# recording-condition-dependent.
N_MFCC: int = 20

# Image geometry
IMAGE_SIZE_MIN:     int = 64
IMAGE_SIZE_MAX:     int = 512
IMAGE_SIZE_DEFAULT: int = 256
# IMAGE_SIZE_STEP = 16: step of 16 keeps all sizes divisible by 16,
# which aligns with typical GPU/SIMD tile widths and ensures symmetric
# Hermitian indices (N//2 is always an integer).
IMAGE_SIZE_STEP: int = 16

# Magnitude grid blend weights — must sum exactly to 1.0.
# STFT (0.45): primary representation; accounts for multiple resolutions.
# CWT  (0.15): adds multi-scale temporal structure orthogonal to STFT.
# Mel  (0.18): perceptually weighted frequency axis.
# Chroma (0.09): pitch-class content, octave-invariant.
# MFCC (0.09): coarse spectral envelope.
# RMS  (0.04): loudness envelope as a spatial amplitude modulation.
W_STFT_TOTAL: float = 0.45
W_CWT_MAG:    float = 0.15
W_MEL:        float = 0.18
W_CHROMA:     float = 0.09
W_MFCC:       float = 0.09
W_RMS:        float = 0.04
# Sum = 0.45 + 0.15 + 0.18 + 0.09 + 0.09 + 0.04 = 1.00

# Phase grid blend weights — must sum to 1.0 (Morlet) or be renormalized
# (Ricker, where CWT phase is unavailable).
# STFT 1024 (0.30): best overall time–frequency balance; primary source.
# STFT 512  (0.20): higher time resolution; captures fast transients.
# CWT Morlet (0.20): instantaneous phase at multiple scales.
# Onset (0.15): phase jumps at rhythmic events.
# Centroid (0.10): pitch brightness variation encodes melodic contour.
# ZCR (0.05): noisiness encodes timbre texture.
W_PHASE_STFT_1024: float = 0.30
W_PHASE_STFT_512:  float = 0.20
W_PHASE_CWT:       float = 0.20   # set to 0 and redistributed for Ricker
W_PHASE_ONSET:     float = 0.15
W_PHASE_CENTROID:  float = 0.10
W_PHASE_ZCR:       float = 0.05
# Morlet total: 0.30+0.20+0.20+0.15+0.10+0.05 = 1.00
# Ricker total (redistribute CWT weight 0.20 to STFT): 0.40+0.30+0+0.15+0.10+0.05 = 1.00

WAVELET_OPTIONS     = ["Morlet", "Ricker (Mexican hat)"]
OUTPUT_MODE_OPTIONS = ["Grayscale", "Colors", "Colors + black drawing", "Mix"]
SECTION_LAYOUT_OPTIONS = [
    "None",
    "Chronological treemap",
    "Clockwise circular slices",
    "Concentric circles",
    "Concentric squares",
    "Vertical strips",
    "Horizontal strips",
]

# Sectioned image synthesis
MIN_SECTION_SAMPLES: int = 16_384
MIN_BLOCK_SIDE: int = 32
DEFAULT_SECTIONS: int = 32
MAX_SECTIONS_UI: int = 64

COLORMAP_OPTIONS    = [
    "inferno", "magma", "plasma", "viridis",
    "twilight", "hsv", "coolwarm", "turbo",
    "Blues", "Greens", "Reds", "Purples",
]
AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]

# Default open-source music sample.  The file is 14 s long and released as
# CC0 on Wikimedia Commons, which is long enough to make the default section
# count reach 32 with the current MIN_SECTION_SAMPLES rule.
DEFAULT_AUDIO_TITLE = "Phonk sample.ogg"
DEFAULT_AUDIO_DESCRIPTION = "8-bar drift phonk instrumental at 140 BPM, CC0, Wikimedia Commons"
DEFAULT_AUDIO_URL = "https://upload.wikimedia.org/wikipedia/commons/2/2c/Phonk_sample.ogg"

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$",  "right": "$",  "display": False},
]


# ============================================================
# Portfolio links
# ============================================================

PORTFOLIO_LINKS = [
    {
        "platform": "Streamlit",
        "label":    "trungtin-dinh",
        "url":      "https://share.streamlit.io/user/trungtin-dinh",
        "icon_url": "https://cdn.simpleicons.org/streamlit/FF4B4B",
    },
    {
        "platform": "GitHub",
        "label":    "trungtin-dinh",
        "url":      "https://github.com/trungtin-dinh",
        "icon_url": "https://cdn.simpleicons.org/github/FFFFFF",
    },
    {
        "platform": "LinkedIn",
        "label":    "Trung-Tin Dinh",
        "url":      "https://www.linkedin.com/in/trung-tin-dinh/",
        "icon_url": "https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg",
    },
    {
        "platform": "Hugging Face",
        "label":    "trungtindinh",
        "url":      "https://huggingface.co/trungtindinh",
        "icon_url": "https://cdn.simpleicons.org/huggingface/FFD21E",
    },
    {
        "platform": "Medium",
        "label":    "@trungtin.dinh",
        "url":      "https://medium.com/@trungtin.dinh",
        "icon_url": "https://cdn.simpleicons.org/medium/FFFFFF",
    },
    {
        "platform": "CV FR",
        "label":    "CV FR",
        "url":      "http://e.pc.cd/t2ly6alK",
        "icon_url": "https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg",
    },
    {
        "platform": "CV EN",
        "label":    "CV EN",
        "url":      "http://e.pc.cd/KjMotalK",
        "icon_url": "https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg",
    },
]


def render_portfolio_links() -> None:
    parts = []
    for item in PORTFOLIO_LINKS:
        show_label = item["platform"] in {"CV FR", "CV EN"}
        link_class = "portfolio-link with-label" if show_label else "portfolio-link icon-only"
        title = (
            f"Open {item['platform']}: {item['label']}"
            if not show_label
            else f"Open {item['platform']}"
        )
        label_html = (
            f'<span class="portfolio-label">{html.escape(item["label"])}</span>'
            if show_label
            else ""
        )
        parts.append(
            f'<a class="{link_class}" '
            f'href="{html.escape(item["url"], quote=True)}" '
            f'target="_blank" rel="noopener noreferrer" '
            f'title="{html.escape(title, quote=True)}" '
            f'aria-label="{html.escape(title, quote=True)}">'
            f'<img class="portfolio-icon" '
            f'src="{html.escape(item["icon_url"], quote=True)}" '
            f'alt="{html.escape(item["platform"], quote=True)} icon">'
            f'{label_html}'
            f'</a>'
        )
    st.markdown(
        f'<div class="portfolio-link-row">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Documentation loading
# ============================================================

def read_markdown_file(path: str) -> str:
    """Read a local Markdown file; return a placeholder if missing."""
    fp = Path(path)
    if not fp.exists():
        return (
            "## Documentation unavailable\n\n"
            "The file `" + path + "` was not found in the app directory."
        )
    return fp.read_text(encoding="utf-8")


DOCUMENTATION_fr = read_markdown_file("documentation_fr.md")
DOCUMENTATION_en = read_markdown_file("documentation_en.md")


def split_markdown_by_h2(markdown_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    parts = re.split(r"(?m)^##\s+", markdown_text.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        title = lines[0].strip()
        if title.lower() in {"table des matières", "table of contents"}:
            continue
        sections[title] = "## " + part
    if not sections:
        sections["Documentation"] = markdown_text
    return sections


DOC_FR_SECTIONS = split_markdown_by_h2(DOCUMENTATION_fr)
DOC_EN_SECTIONS = split_markdown_by_h2(DOCUMENTATION_en)
DOC_FR_TITLES   = list(DOC_FR_SECTIONS.keys())
DOC_EN_TITLES   = list(DOC_EN_SECTIONS.keys())


# ============================================================
# Default audio
# ============================================================

@st.cache_data(show_spinner=False)
def load_default_audio() -> tuple[np.ndarray | None, int | None, bytes | None]:
    """
    Load the default open-source music sample.

    The default file is downloaded from Wikimedia Commons and decoded at its
    original sample rate. It is intentionally longer than the previous librosa
    trumpet example so that the default number of sections can reach 32 when
    the image size is large enough.

    Returns:
        waveform:    (N,) float32 mono signal, or None on failure
        sr:          original sample rate in Hz, or None
        audio_bytes: raw bytes for st.audio playback, or None
    """
    if not LIBROSA_AVAILABLE:
        return None, None, None
    try:
        request = urllib.request.Request(
            DEFAULT_AUDIO_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            audio_bytes = response.read()
        waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        max_samples = int(MAX_RECORD_SECONDS * sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        return waveform, sr, audio_bytes
    except Exception:
        return None, None, None


# ============================================================
# Audio loading
# ============================================================

@st.cache_data(show_spinner=False)
def load_audio(audio_bytes: bytes) -> tuple[np.ndarray | None, int | None]:
    """
    Decode raw audio bytes into a mono waveform at its original sample rate.

    Parameters:
        audio_bytes: raw file content in any format supported by librosa

    Returns:
        waveform: (N,) float32 mono array in [-1, 1]
        sr:       original sample rate in Hz
    """
    if not LIBROSA_AVAILABLE:
        return None, None
    waveform, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=None,
        mono=True,
    )
    # Enforce recording duration cap at the original sample rate.
    max_samples = int(MAX_RECORD_SECONDS * sr)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    return waveform, sr


# ============================================================
# Multi-resolution STFT window sizes
# ============================================================

def get_stft_resolutions(n_samples: int) -> list[int]:
    """
    Return all powers-of-two STFT window sizes between STFT_N_FFT_MIN and
    min(STFT_N_FFT_MAX, n_samples // 2).

    The upper cap ensures at least two non-overlapping frames exist in the
    signal (n_fft ≤ n_samples / 2 guarantees T ≥ 2 frames with default
    hop = n_fft // 4, giving T ≈ 4n_samples / n_fft ≥ 8).

    Parameters:
        n_samples: number of audio samples in the waveform

    Returns:
        list of valid n_fft values (ascending), never empty
    """
    upper = min(STFT_N_FFT_MAX, max(STFT_N_FFT_MIN, n_samples // 2))
    resolutions = [
        2 ** k
        for k in range(8, 14)   # 2^8=256 … 2^13=8192
        if STFT_N_FFT_MIN <= 2 ** k <= upper
    ]
    return resolutions if resolutions else [STFT_N_FFT_MIN]




# ============================================================
# SciPy-compatible CWT helpers
# ============================================================

def morlet2_compat(M: int, s: float, w: float = CWT_MORLET_W) -> np.ndarray:
    """
    Return a complex Morlet wavelet compatible with the removed
    scipy.signal.morlet2 API.

    SciPy removed scipy.signal.cwt, scipy.signal.morlet2 and
    scipy.signal.ricker in version 1.15.  Keeping these local helpers avoids
    pinning SciPy to an old version and avoids adding a new PyWavelets
    dependency only for this educational visualization.
    """
    M = int(max(1, M))
    s = float(max(1e-12, s))
    x = np.arange(M, dtype=np.float64) - (M - 1.0) / 2.0
    x_scaled = x / s
    return (np.pi ** -0.25) * np.sqrt(1.0 / s) * np.exp(1j * w * x_scaled) * np.exp(-0.5 * x_scaled**2)


def ricker_compat(M: int, a: float) -> np.ndarray:
    """Return a Ricker / Mexican-hat wavelet compatible with scipy.signal.ricker."""
    M = int(max(1, M))
    a = float(max(1e-12, a))
    x = np.arange(M, dtype=np.float64) - (M - 1.0) / 2.0
    xsq = x**2
    asq = a**2
    amplitude = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    return amplitude * (1.0 - xsq / asq) * np.exp(-xsq / (2.0 * asq))


def cwt_compat(data: np.ndarray, wavelet_fn, widths: np.ndarray) -> np.ndarray:
    """
    Local replacement for the removed scipy.signal.cwt.

    For each width, the signal is convolved with the time-reversed complex
    conjugate of the corresponding wavelet, matching the behavior documented
    for scipy.signal.cwt before its removal.
    """
    data = np.asarray(data)
    widths = np.asarray(widths, dtype=np.float64)
    outputs = []

    for width in widths:
        wavelet_length = min(int(np.ceil(10.0 * width)), data.size)
        wavelet_length = max(1, wavelet_length)
        wavelet = np.asarray(wavelet_fn(wavelet_length, width))
        coeff = scipy.signal.convolve(
            data,
            np.conj(wavelet)[::-1],
            mode="same",
            method="auto",
        )
        outputs.append(coeff)

    return np.asarray(outputs)


# ============================================================
# Feature extraction
# ============================================================

def extract_features(waveform: np.ndarray, sr: int, wavelet_type: str = "Morlet") -> dict:
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
            extracted.  The Morlet wavelet W(t) = π^{-1/4} · exp(jω₀t) ·
            exp(−t²/2) with ω₀ = CWT_MORLET_W provides simultaneous time and
            frequency localization, optimal under the uncertainty principle.
        Ricker (real, symmetric): only magnitude is extracted (no analytic
            signal → no meaningful instantaneous phase).  The Ricker wavelet
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
            log mel spectrogram.  Coefficients encode the spectral envelope
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
            the log mel spectrogram.  Peaks sharply at note onsets and
            rhythmic events.

    Parameters:
        waveform:     (N,) float32 mono signal at DEFAULT_AUDIO_SR
        sr:           sample rate in Hz
        wavelet_type: "Morlet" or "Ricker (Mexican hat)"

    Returns:
        dict mapping feature names to numpy arrays, plus the list of STFT
        resolutions used under the key "stft_resolutions"
    """
    features: dict = {}
    hop_default = 256   # shared hop length for perceptual features (≈11.6 ms at 22050 Hz)

    # --- Multi-resolution STFT ---
    resolutions = get_stft_resolutions(len(waveform))
    features["stft_resolutions"] = resolutions

    for n_fft in resolutions:
        hop = n_fft // 4
        stft = librosa.stft(
            waveform, n_fft=n_fft, hop_length=hop, window="hann", center=True,
        )
        features[f"mag_{n_fft}"]   = np.abs(stft)
        features[f"phase_{n_fft}"] = np.angle(stft)

    # --- CWT ---
    # Downsample waveform for CWT to bound computation time
    step_cwt = max(1, len(waveform) // CWT_MAX_SAMPLES)
    waveform_cwt = waveform[::step_cwt].copy()
    n_cwt = len(waveform_cwt)

    max_scale = min(512, n_cwt // 2)
    cwt_scales = np.geomspace(1.0, max(1.0, float(max_scale)), num=CWT_N_SCALES)

    if wavelet_type == "Morlet":
        wavelet_fn = lambda M, s: morlet2_compat(M, s, w=CWT_MORLET_W)
    else:
        wavelet_fn = ricker_compat

    cwt_coeffs = cwt_compat(waveform_cwt, wavelet_fn, cwt_scales)
    # cwt_coeffs: (CWT_N_SCALES, n_cwt), complex for Morlet, real for Ricker

    features["cwt_magnitude"] = np.abs(cwt_coeffs)
    features["cwt_phase"] = np.angle(cwt_coeffs) if wavelet_type == "Morlet" else None

    # --- Perceptual features ---
    features["mel"] = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=N_MELS, hop_length=hop_default,
    )
    features["chroma"] = librosa.feature.chroma_stft(
        y=waveform, sr=sr, hop_length=hop_default,
    )
    features["mfcc"] = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=N_MFCC, hop_length=hop_default,
    )

    # --- Temporal descriptors ---
    features["spectral_centroid"] = librosa.feature.spectral_centroid(
        y=waveform, sr=sr, hop_length=hop_default,
    )[0]
    features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(
        y=waveform, sr=sr, hop_length=hop_default,
    )[0]
    features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
        y=waveform, sr=sr, hop_length=hop_default,
    )[0]
    features["spectral_flatness"] = librosa.feature.spectral_flatness(
        y=waveform, hop_length=hop_default,
    )[0]
    features["rms"] = librosa.feature.rms(
        y=waveform, hop_length=hop_default,
    )[0]
    features["zcr"] = librosa.feature.zero_crossing_rate(
        waveform, hop_length=hop_default,
    )[0]
    features["onset_strength"] = librosa.onset.onset_strength(
        y=waveform, sr=sr, hop_length=hop_default,
    )

    return features


# ============================================================
# Grid construction utilities
# ============================================================

def normalize_to_unit(array: np.ndarray) -> np.ndarray:
    """Linearly map all values to [0, 1]; return zero array if constant."""
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


def interpolate_to_shape(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resample a 2D array to (target_h, target_w) via bicubic spline interpolation.

    Both source and destination grids are normalized to [0, 1] so the
    resampling is scale-invariant.  This handles arbitrary upsampling and
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
    replicated across N rows.  The result encodes temporal evolution as a
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


# ============================================================
# 2D Fourier magnitude grid
# ============================================================

def build_magnitude_grid(
    features: dict,
    target_size: int,
    band: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Build a (N, N) Fourier magnitude grid M̃[u, v] from multi-resolution
    audio features.

    The grid is a weighted sum of six independent feature sources.  Each
    source is log-compressed (log(1 + x)) where applicable, normalized to
    [0, 1], and bicubically interpolated to (N, N) before blending.

    Source 1 — Multi-resolution STFT log-magnitudes (total weight W_STFT_TOTAL = 0.45):
        The W_STFT_TOTAL budget is distributed equally across all available
        STFT resolutions. For a two-minute signal this yields up to 6 STFTs
        (n_fft ∈ {256, 512, 1024, 2048, 4096, 8192}), each contributing
        0.45 / 6 = 0.075. Equal weights treat all time–frequency scales
        symmetrically; no prior assumption is made about which scale is most
        informative for a given audio class.

    Source 2 — CWT log-magnitude (weight W_CWT_MAG = 0.15):
        |W[s, t]| after logarithmic compression.  The CWT magnitude encodes
        the energy at scale s and time t in a way that is complementary to
        the STFT: at low frequencies the CWT has higher frequency resolution
        and lower time resolution than the STFT (and vice versa at high
        frequencies), so the blend fills the resolution gap.

    Source 3 — Log-mel spectrogram (weight W_MEL = 0.18):
        The mel filterbank integrates STFT power across triangular filters on
        the mel frequency axis, producing a representation weighted by human
        auditory sensitivity.  The 20% higher weight relative to MFCC/Chroma
        reflects the fact that mel magnitude directly encodes spectral energy,
        while MFCC and Chroma are derived statistics.

    Source 4 — Chroma (weight W_CHROMA = 0.09):
        12-bin pitch-class energy, summed across octaves.  Captures the
        harmonic / tonal structure of the signal independently of register.
        Its 2D (pitch × time) layout maps naturally onto the spatial grid.

    Source 5 — MFCC absolute values (weight W_MFCC = 0.09):
        |c_i[t]|, i = 0 … N_MFCC − 1.  Absolute values ensure all cepstral
        coefficients contribute positively (negative coefficients encode
        spectral valleys, not absence of energy).

    Source 6 — RMS energy envelope (weight W_RMS = 0.04):
        Tiled (N, N) temporal map.  Encodes loudness dynamics as a spatially
        uniform amplitude modulation along the time axis.  The low weight (4%)
        reflects that RMS duplicates information already present in the
        magnitude spectrograms but at coarser frequency resolution.

    When band ≠ None, all spectrograms are sliced to the fractional sub-band
    [lo, hi) along the frequency axis before interpolation, restricting the
    grid to one frequency register for the RGB channel decomposition.

    Parameters:
        features:    dict from extract_features()
        target_size: N, output grid side length in pixels
        band:        (lo, hi) normalized frequency band, or None for full band

    Returns:
        (N, N) float32 array in [0, 1] representing M̃[u, v]
    """
    resolutions = features["stft_resolutions"]
    w_per_stft  = W_STFT_TOTAL / len(resolutions)
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
    cwt_log = normalize_to_unit(np.log1p(cwt_mag))
    components.append((W_CWT_MAG, interpolate_to_shape(cwt_log, target_size, target_size)))

    mel = features["mel"]
    if band is not None:
        mel = slice_band(mel, band)
    mel_log = normalize_to_unit(np.log1p(mel))
    components.append((W_MEL, interpolate_to_shape(mel_log, target_size, target_size)))

    chroma = normalize_to_unit(np.abs(features["chroma"]))
    components.append((W_CHROMA, interpolate_to_shape(chroma, target_size, target_size)))

    mfcc = normalize_to_unit(np.abs(features["mfcc"]))
    components.append((W_MFCC, interpolate_to_shape(mfcc, target_size, target_size)))

    rms_2d = row_to_2d(features["rms"], target_size)
    components.append((W_RMS, rms_2d))

    combined = sum(w * arr for w, arr in components)
    return normalize_to_unit(combined)


# ============================================================
# 2D Fourier phase grid
# ============================================================

def build_phase_grid(
    features: dict,
    target_size: int,
    band: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Build a (N, N) Fourier phase grid Φ̃[u, v] from multi-resolution audio features.

    The phase grid is assembled from up to six weighted contributions.
    Because phase is a circular quantity in (−π, π], simple arithmetic
    averaging can introduce wrap-around bias.  To mitigate this, the STFT and
    CWT sources are unwrapped before interpolation (removing the 2π
    discontinuities) so the interpolated field varies smoothly.  The
    temporal modulation terms (onset, centroid, ZCR) are scaled to lie
    in [0, π/2], [0, π], and [0, π/4] respectively and act as additive
    phase offsets rather than independent phase references.  The final
    combined field is re-wrapped to (−π, π] by the modulo operation.

    Contributions:

    1. STFT instantaneous phase at n_fft = 1024 (weight W_PHASE_STFT_1024 = 0.30):
       Φ[k, t] = ∠S_{1024}[k, t] ∈ (−π, π].  Unwrapped along both the
       frequency axis k (to remove octave-boundary jumps) and the time axis t
       (to remove frame-boundary jumps), then bicubically interpolated to
       (N, N), then re-wrapped.

    2. STFT instantaneous phase at n_fft = 512 (weight W_PHASE_STFT_512 = 0.20):
       Same pipeline at the finer-time resolution STFT.  The 512-sample window
       resolves faster phase evolution (attack transients) that the 1024-window
       smears over two frames.

    3. CWT instantaneous phase — Morlet only (weight W_PHASE_CWT = 0.20):
       ∠W[s, t] for the analytic Morlet wavelet.  CWT phase is well-defined
       because the Morlet wavelet is analytic (imaginary part ≈ 0 at DC),
       so ∠W represents the true instantaneous phase of the signal bandpassed
       at scale s.  Unavailable for Ricker (real-valued wavelet): the 0.20
       weight is redistributed to the two STFT sources instead, raising them
       to 0.40 and 0.30 respectively.

    4. Onset strength modulation (weight W_PHASE_ONSET = 0.15):
       onset_strength[t] ≥ 0, normalized to [0, 1] and scaled to [0, π/2].
       Tiled to (N, N).  Peaks at rhythmic events create spatially localized
       phase transitions in the image, analogous to the discontinuities at
       object boundaries that dominate visual perception.

    5. Spectral centroid modulation (weight W_PHASE_CENTROID = 0.10):
       spectral_centroid[t] normalized to [0, 1] and scaled to [0, π].
       Encodes the instantaneous brightness (pitch register) of each frame as
       a time-varying phase offset, mapping melodic contour into a slowly-
       varying spatial phase gradient.

    6. Zero-crossing rate modulation (weight W_PHASE_ZCR = 0.05):
       zcr[t] normalized to [0, 1] and scaled to [0, π/4].  High ZCR (noisy
       or percussive signals) adds fine-scale phase perturbations; low ZCR
       (tonal signals) leaves the phase smooth.  The small weight (5%)
       prevents ZCR from dominating over the structured phase sources.

    Parameters:
        features:    dict from extract_features()
        target_size: N, output grid side length in pixels
        band:        (lo, hi) normalized frequency band, or None for full band

    Returns:
        (N, N) float32 array in (−π, π] representing Φ̃[u, v]
    """
    resolutions  = features["stft_resolutions"]
    has_cwt_phase = features.get("cwt_phase") is not None

    # STFT 1024 phase
    n_fft_mid = 1024 if 1024 in resolutions else resolutions[len(resolutions) // 2]
    phase_mid = features[f"phase_{n_fft_mid}"]
    if band is not None:
        phase_mid = slice_band(phase_mid, band)
    unwrapped_mid = np.unwrap(np.unwrap(phase_mid, axis=0), axis=1)
    grid_mid = interpolate_to_shape(unwrapped_mid, target_size, target_size)

    # STFT 512 phase (finest available, or fallback to smallest)
    n_fft_fine = 512 if 512 in resolutions else resolutions[0]
    phase_fine = features[f"phase_{n_fft_fine}"]
    if band is not None:
        phase_fine = slice_band(phase_fine, band)
    unwrapped_fine = np.unwrap(np.unwrap(phase_fine, axis=0), axis=1)
    grid_fine = interpolate_to_shape(unwrapped_fine, target_size, target_size)

    # Blend weights — redistribute CWT share if unavailable
    if has_cwt_phase:
        w_mid, w_fine = W_PHASE_STFT_1024, W_PHASE_STFT_512
        w_cwt = W_PHASE_CWT
    else:
        # Distribute W_PHASE_CWT = 0.20 proportionally to the two STFT sources
        total_stft = W_PHASE_STFT_1024 + W_PHASE_STFT_512   # = 0.50
        w_mid  = W_PHASE_STFT_1024 + W_PHASE_CWT * (W_PHASE_STFT_1024 / total_stft)
        w_fine = W_PHASE_STFT_512  + W_PHASE_CWT * (W_PHASE_STFT_512  / total_stft)
        w_cwt  = 0.0

    # CWT phase (Morlet only)
    if has_cwt_phase:
        cwt_phase = features["cwt_phase"]
        if band is not None:
            cwt_phase = slice_band(cwt_phase, band)
        cwt_unwrapped = np.unwrap(np.unwrap(cwt_phase, axis=0), axis=1)
        grid_cwt = interpolate_to_shape(cwt_unwrapped, target_size, target_size)
    else:
        grid_cwt = np.zeros((target_size, target_size), dtype=np.float64)

    # Temporal modulation maps
    onset_2d    = row_to_2d(features["onset_strength"],    target_size) * (np.pi / 2.0)
    centroid_2d = row_to_2d(features["spectral_centroid"], target_size) * np.pi
    zcr_2d      = row_to_2d(features["zcr"],               target_size) * (np.pi / 4.0)

    phase_combined = (
        w_mid                  * grid_mid
        + w_fine               * grid_fine
        + w_cwt                * grid_cwt
        + W_PHASE_ONSET        * onset_2d
        + W_PHASE_CENTROID     * centroid_2d
        + W_PHASE_ZCR          * zcr_2d
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

    Parameters:
        Z: (N, N) complex array

    Returns:
        (N, N) Hermitian-symmetric complex array
    """
    Z_r = np.roll(np.roll(Z[::-1, ::-1], 1, axis=0), 1, axis=1)
    return (Z + np.conj(Z_r)) / 2.0


# ============================================================
# Image reconstruction
# ============================================================

def apply_colormap(values: np.ndarray, colormap_name: str) -> np.ndarray:
    """Apply a matplotlib colormap to a (H, W) float array in [0, 1].

    Returns:
        (H, W, 3) uint8 RGB image
    """
    cmap = matplotlib_cm.get_cmap(colormap_name)
    rgba = cmap(values)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def reconstruct_channel_raw_and_spectrum(
    magnitude_2d: np.ndarray,
    phase_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct one spatial channel without local min-max normalization.

    This is used for section-by-section synthesis so that all blocks can be
    normalized only once at the final-image level.
    """
    Z = magnitude_2d * np.exp(1j * phase_2d)
    Z_sym = enforce_hermitian_symmetry(Z)
    f = np.real(np.fft.ifft2(Z_sym))
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


def reconstruct_channel(magnitude_2d: np.ndarray, phase_2d: np.ndarray) -> np.ndarray:
    """
    Reconstruct a single spatial channel from a 2D Fourier magnitude and phase.

    This compatibility wrapper returns only the reconstructed channel.  The
    actual Hermitian spectrum used by np.fft.ifft2 is available through
    reconstruct_channel_and_spectrum().
    """
    channel, _ = reconstruct_channel_and_spectrum(magnitude_2d, phase_2d)
    return channel


def spectrum_to_centered_magnitude_phase(Z_sym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the exact spectrum used by np.fft.ifft2 into display grids.

    np.fft.ifft2 expects the DC/origin coefficient at index (0, 0).  For a
    standard Fourier visualization, the origin is moved to the center with
    fftshift.  The returned magnitude and phase therefore show the actual
    Hermitian-symmetrized spectrum used for reconstruction, but displayed in
    the conventional centered layout.
    """
    Z_centered = np.fft.fftshift(Z_sym)
    magnitude_centered = np.abs(Z_centered)
    phase_centered = np.angle(Z_centered)
    return magnitude_centered, phase_centered


def audio_to_image_float(
    features: dict,
    target_size: int,
    output_mode: str,
) -> np.ndarray:
    """
    Generate an unnormalized floating-point image patch from extracted features.

    Unlike audio_to_image(), this function does not apply a local [0, 1]
    normalization after each IFFT. It is therefore suitable for sectioned
    synthesis, where the final assembled image is normalized globally.
    """
    if output_mode == "Grayscale":
        mag_2d_raw = build_magnitude_grid(features, target_size, band=None)
        phase_2d_raw = build_phase_grid(features, target_size, band=None)
        channel, _ = reconstruct_channel_raw_and_spectrum(mag_2d_raw, phase_2d_raw)
        return np.stack([channel, channel, channel], axis=2)

    bands = [(0.00, 1/3), (1/3, 2/3), (2/3, 1.00)]
    channels = []
    for band in bands:
        mag_2d_raw = build_magnitude_grid(features, target_size, band=band)
        phase_2d_raw = build_phase_grid(features, target_size, band=band)
        channel, _ = reconstruct_channel_raw_and_spectrum(mag_2d_raw, phase_2d_raw)
        channels.append(channel)

    return np.stack(channels, axis=2)


def audio_to_image(
    features: dict,
    target_size: int,
    output_mode: str,
    colormap_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an image from pre-extracted audio features.

    Grayscale mode:
        Full-band magnitude and phase grids → IFFT2 → colormap → (N, N, 3) RGB.

    RGB mode:
        The frequency axis [0, 1) is partitioned into three equal bands:
            Low  → R : [0.00, 0.33)  — bass / lower mid-range
            Mid  → G : [0.33, 0.67)  — upper mid-range
            High → B : [0.67, 1.00)  — treble / presence
        Each band follows the full pipeline independently, producing one
        spatial channel that encodes the energy distribution of that frequency
        register.  The three channels are stacked as R, G, B.

    Returns:
        image_rgb:      (N, N, 3) uint8 generated image
        mag_2d_used:    (N, N) centered magnitude of the Hermitian spectrum
                        actually passed to np.fft.ifft2
        phase_2d_used:  (N, N) centered phase of the Hermitian spectrum
                        actually passed to np.fft.ifft2

        In RGB mode, the displayed magnitude and phase correspond to the
        low-frequency band used to reconstruct the R channel.
    """
    if output_mode == "Grayscale":
        mag_2d_raw   = build_magnitude_grid(features, target_size, band=None)
        phase_2d_raw = build_phase_grid(features, target_size, band=None)
        channel, Z_sym = reconstruct_channel_and_spectrum(mag_2d_raw, phase_2d_raw)
        mag_2d_used, phase_2d_used = spectrum_to_centered_magnitude_phase(Z_sym)
        gray = (channel * 255).astype(np.uint8)
        image_gray_rgb = np.stack([gray, gray, gray], axis=2)
        return image_gray_rgb, mag_2d_used, phase_2d_used

    bands    = [(0.00, 1/3), (1/3, 2/3), (2/3, 1.00)]
    channels = []
    mag_2d_ref = phase_2d_ref = None

    for i, band in enumerate(bands):
        mag_2d_raw   = build_magnitude_grid(features, target_size, band=band)
        phase_2d_raw = build_phase_grid(features, target_size, band=band)
        channel, Z_sym = reconstruct_channel_and_spectrum(mag_2d_raw, phase_2d_raw)
        channels.append((channel * 255).astype(np.uint8))
        if i == 0:
            mag_2d_ref, phase_2d_ref = spectrum_to_centered_magnitude_phase(Z_sym)

    return np.stack(channels, axis=2), mag_2d_ref, phase_2d_ref


# ============================================================
# Sectioned image synthesis helpers
# ============================================================

def compute_max_sections(n_samples: int, target_size: int) -> int:
    """
    Compute a dynamic upper bound for the number of temporal sections.

    The limit is expressed in samples rather than seconds. Each section should
    contain enough samples for the largest STFT window and the CWT/MFCC/onset
    descriptors to remain meaningful, while each visual block should remain
    readable in the final square image.
    """
    k_time = max(1, int(n_samples) // MIN_SECTION_SAMPLES)
    k_space = max(1, (int(target_size) // MIN_BLOCK_SIDE) ** 2)
    return max(1, min(k_time, k_space, MAX_SECTIONS_UI))


def split_waveform_into_sections(waveform: np.ndarray, n_sections: int) -> list[np.ndarray]:
    """Split a waveform into n chronological sections with nearly equal sample counts."""
    n_sections = max(1, int(n_sections))
    boundaries = np.linspace(0, len(waveform), n_sections + 1, dtype=int)
    sections: list[np.ndarray] = []
    for i in range(n_sections):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        if end <= start:
            end = min(len(waveform), start + 1)
        sections.append(waveform[start:end].copy())
    return sections


def recursive_chronological_layout(
    x: int,
    y: int,
    w: int,
    h: int,
    section_start: int,
    n_sections: int,
) -> list[dict[str, int]]:
    """
    Build a deterministic chronological equal-area treemap layout.

    The section list is always split into chronological halves. The current
    rectangle is split along its longest side, with size proportional to the
    number of sections in each half. A square tie is split vertically. This
    gives one universal rule for every possible section count.
    """
    if n_sections <= 1:
        return [{"section": section_start, "x": x, "y": y, "w": max(1, w), "h": max(1, h)}]

    n_first = (n_sections + 1) // 2
    n_second = n_sections - n_first

    if w >= h:
        w_first = int(round(w * n_first / n_sections))
        w_first = min(max(1, w_first), w - 1)
        first = recursive_chronological_layout(x, y, w_first, h, section_start, n_first)
        second = recursive_chronological_layout(
            x + w_first, y, w - w_first, h, section_start + n_first, n_second
        )
    else:
        h_first = int(round(h * n_first / n_sections))
        h_first = min(max(1, h_first), h - 1)
        first = recursive_chronological_layout(x, y, w, h_first, section_start, n_first)
        second = recursive_chronological_layout(
            x, y + h_first, w, h - h_first, section_start + n_first, n_second
        )

    return first + second


def fit_square_patch_to_rect_float(patch: np.ndarray, rect_w: int, rect_h: int) -> np.ndarray:
    """Fit a floating square patch to a rectangular block by centered crop/resize."""
    rect_w = max(1, int(rect_w))
    rect_h = max(1, int(rect_h))
    patch = np.asarray(patch, dtype=np.float64)
    h, w = patch.shape[:2]

    crop_w = min(rect_w, w)
    crop_h = min(rect_h, h)
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    cropped = patch[y0:y0 + crop_h, x0:x0 + crop_w]

    if cropped.shape[1] == rect_w and cropped.shape[0] == rect_h:
        return cropped.astype(np.float64)

    return resize_float_image_to_size(cropped, rect_w, rect_h)


def fit_square_patch_to_rect(patch: np.ndarray, rect_w: int, rect_h: int) -> np.ndarray:
    """
    Fit a generated square patch to a rectangular block.

    The local image is computed at the longest block edge. If the block is
    rectangular, the square patch is center-cropped to the target rectangle.
    If an integer rounding edge case occurs, PIL is used only as a final resize
    safeguard.
    """
    rect_w = max(1, int(rect_w))
    rect_h = max(1, int(rect_h))
    patch = np.asarray(patch)
    h, w = patch.shape[:2]

    crop_w = min(rect_w, w)
    crop_h = min(rect_h, h)
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    cropped = patch[y0:y0 + crop_h, x0:x0 + crop_w]

    if cropped.shape[1] == rect_w and cropped.shape[0] == rect_h:
        return cropped.astype(np.uint8)

    pil_img = Image.fromarray(cropped.astype(np.uint8))
    pil_img = pil_img.resize((rect_w, rect_h), Image.Resampling.BICUBIC)
    return np.asarray(pil_img, dtype=np.uint8)


def draw_block_borders(image: np.ndarray, rectangles: list[dict[str, int]]) -> np.ndarray:
    """Draw thin visible borders so the section-by-section structure is explicit."""
    out = image.copy()
    border = max(1, out.shape[0] // 256)
    color = np.array([245, 245, 245], dtype=np.uint8)
    for rect in rectangles:
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        x1 = min(out.shape[1], x + w)
        y1 = min(out.shape[0], y + h)
        out[y:y + border, x:x1] = color
        out[max(y, y1 - border):y1, x:x1] = color
        out[y:y1, x:x + border] = color
        out[y:y1, max(x, x1 - border):x1] = color
    return out


def resize_float_image_to_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a floating RGB image to an arbitrary size with bicubic interpolation."""
    width = max(1, int(width))
    height = max(1, int(height))
    image = np.asarray(image, dtype=np.float64)
    if image.shape[0] == height and image.shape[1] == width:
        return image

    zoom_y = height / max(1, image.shape[0])
    zoom_x = width / max(1, image.shape[1])
    if image.ndim == 2:
        resized = scipy.ndimage.zoom(image, (zoom_y, zoom_x), order=3)
    else:
        resized = scipy.ndimage.zoom(image, (zoom_y, zoom_x, 1.0), order=3)

    return resized[:height, :width].astype(np.float64)


def resize_float_image_to_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a floating RGB image to a square size."""
    size = max(1, int(size))
    return resize_float_image_to_size(image, size, size)


def resize_image_to_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB image to a square size using bicubic interpolation."""
    size = max(1, int(size))
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[0] == size and image.shape[1] == size:
        return image
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(pil_img, dtype=np.uint8)


def generate_section_patch(
    section: np.ndarray,
    sr: int,
    patch_size: int,
    output_mode: str,
    wavelet_type: str,
) -> np.ndarray:
    """Generate one unnormalized floating-point square patch from one audio section."""
    patch_size = max(8, int(patch_size))
    features = extract_features(section, sr, wavelet_type=wavelet_type)
    return audio_to_image_float(
        features=features,
        target_size=patch_size,
        output_mode=output_mode,
    )


def finalize_sectioned_image(canvas_float: np.ndarray, output_mode: str) -> np.ndarray:
    """
    Apply one global robust normalization after all sections are assembled.

    This avoids local per-section normalization, preserves stronger intensity
    differences between sections, and gives a less flat final result.
    """
    canvas_float = np.asarray(canvas_float, dtype=np.float64)

    if output_mode == "Grayscale":
        gray = normalize_to_unit_robust(canvas_float[:, :, 0], 1.0, 99.0)
        image = np.stack([gray, gray, gray], axis=2)
    else:
        channels = [
            normalize_to_unit_robust(canvas_float[:, :, c], 1.0, 99.0)
            for c in range(3)
        ]
        image = np.stack(channels, axis=2)

    # Slight contrast lift after global normalization. This is intentionally
    # global, not section-wise, so it increases relief without equalizing blocks.
    image = np.clip(image, 0.0, 1.0) ** 0.85
    return (image * 255.0).round().astype(np.uint8)




def otsu_threshold_unit(values: np.ndarray) -> float:
    """
    Compute Otsu's threshold on values normalized to [0, 1].

    This local implementation avoids adding scikit-image as a dependency.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.5

    arr = np.clip(arr, 0.0, 1.0)
    if arr.max() <= arr.min():
        return float(arr.min())

    hist, bin_edges = np.histogram(arr, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5

    prob = hist / total
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    between_class_variance = np.zeros_like(centers)
    valid = denom > 1e-12
    between_class_variance[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]

    return float(centers[int(np.argmax(between_class_variance))])


def apply_black_drawing_from_grayscale(color_image: np.ndarray, grayscale_image: np.ndarray) -> np.ndarray:
    """
    Use a grayscale-generated image as a sparse binary drawing mask over a color image.

    Step 1:
        The grayscale image is binarized by Otsu's method.

    Step 2:
        Between the two Otsu classes, the minority class is selected as the
        candidate drawing layer.

    Step 3:
        The candidate class is split again by its own median gray value, so
        approximately half of the candidate pixels are discarded. The preserved
        half is the more extreme side of the original histogram:
            - if the low Otsu class is the minority, keep only its darker half;
            - if the high Otsu class is the minority, keep only its brighter half.

    The final sparse mask is rendered in black on top of the color image.
    """
    color = np.asarray(color_image, dtype=np.uint8).copy()
    gray_rgb = np.asarray(grayscale_image, dtype=np.float64)

    if gray_rgb.ndim == 3:
        gray = 0.299 * gray_rgb[:, :, 0] + 0.587 * gray_rgb[:, :, 1] + 0.114 * gray_rgb[:, :, 2]
    else:
        gray = gray_rgb

    gray = normalize_to_unit(gray)
    threshold = otsu_threshold_unit(gray)
    low_mask = gray <= threshold
    high_mask = gray > threshold

    low_count = int(np.count_nonzero(low_mask))
    high_count = int(np.count_nonzero(high_mask))

    if low_count == 0 and high_count == 0:
        return color
    if low_count == 0:
        candidate_mask = high_mask
        keep_high_extreme = True
    elif high_count == 0:
        candidate_mask = low_mask
        keep_high_extreme = False
    elif low_count <= high_count:
        candidate_mask = low_mask
        keep_high_extreme = False
    else:
        candidate_mask = high_mask
        keep_high_extreme = True

    candidate_values = gray[candidate_mask]
    if candidate_values.size == 0:
        return color

    # Split the selected Otsu class into two parts with approximately equal
    # pixel counts. This keeps only half of the original black drawing pixels.
    second_threshold = float(np.median(candidate_values))
    if keep_high_extreme:
        drawing_mask = candidate_mask & (gray >= second_threshold)
    else:
        drawing_mask = candidate_mask & (gray <= second_threshold)

    # Degenerate fallback: if the selected class is constant, median splitting
    # may keep all pixels. In that case, keep exactly the most extreme half by
    # rank, while preserving the intended dark/bright side.
    if np.count_nonzero(drawing_mask) >= candidate_values.size:
        candidate_indices = np.flatnonzero(candidate_mask.ravel())
        candidate_gray = gray.ravel()[candidate_indices]
        n_keep = max(1, candidate_indices.size // 2)
        if keep_high_extreme:
            keep_local = np.argpartition(candidate_gray, -n_keep)[-n_keep:]
        else:
            keep_local = np.argpartition(candidate_gray, n_keep - 1)[:n_keep]
        drawing_flat = np.zeros(gray.size, dtype=bool)
        drawing_flat[candidate_indices[keep_local]] = True
        drawing_mask = drawing_flat.reshape(gray.shape)

    color[drawing_mask] = 0
    return color


def apply_grayscale_mix_to_color(color_image: np.ndarray, grayscale_image: np.ndarray) -> np.ndarray:
    """
    Use a grayscale-generated image as a multiplicative coefficient map over a color image.

    The color image provides the RGB base. The grayscale image is converted to
    a coefficient field C(x, y) in [0, 1], then each color channel is multiplied
    by this same coefficient:

        R_out = C · R
        G_out = C · G
        B_out = C · B

    This keeps the chromatic structure from the Colors mode while using the
    grayscale reconstruction as a luminance/contrast modulation.
    """
    color = np.asarray(color_image, dtype=np.float64)
    gray_rgb = np.asarray(grayscale_image, dtype=np.float64)

    if gray_rgb.ndim == 3:
        gray = 0.299 * gray_rgb[:, :, 0] + 0.587 * gray_rgb[:, :, 1] + 0.114 * gray_rgb[:, :, 2]
    else:
        gray = gray_rgb

    coeff = normalize_to_unit(gray)
    mixed = color * coeff[:, :, None]
    return np.clip(mixed, 0.0, 255.0).round().astype(np.uint8)


def build_layout_index_map(target_size: int, n_sections: int, section_layout: str) -> np.ndarray | None:
    """
    Build a dense section-index map for mask-based layouts.

    The treemap layout is handled separately because its blocks may have
    different rectangle sizes. All other layouts return an (N, N) integer map
    whose value at each pixel is the chronological section index assigned to
    that pixel.
    """
    n = int(target_size)
    k = max(1, int(n_sections))

    if section_layout == "Chronological treemap":
        return None

    y, x = np.indices((n, n), dtype=np.float64)

    if section_layout == "Clockwise circular slices":
        center = (n - 1.0) / 2.0
        # Clockwise angle measured from the vertical upward direction.
        theta = np.arctan2(x - center, center - y)
        theta = np.where(theta < 0.0, theta + 2.0 * np.pi, theta)
        return np.clip(np.floor(theta / (2.0 * np.pi) * k).astype(int), 0, k - 1)

    if section_layout == "Concentric circles":
        center = (n - 1.0) / 2.0
        radius = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        radius_max = max(1.0, float(radius.max()))
        return np.clip(np.floor(radius / radius_max * k).astype(int), 0, k - 1)

    if section_layout == "Concentric squares":
        center = (n - 1.0) / 2.0
        radius = np.maximum(np.abs(x - center), np.abs(y - center))
        radius_max = max(1.0, float(radius.max()))
        return np.clip(np.floor(radius / radius_max * k).astype(int), 0, k - 1)

    if section_layout == "Vertical strips":
        return np.clip(np.floor(x / max(1.0, float(n)) * k).astype(int), 0, k - 1)

    if section_layout == "Horizontal strips":
        return np.clip(np.floor(y / max(1.0, float(n)) * k).astype(int), 0, k - 1)

    return None


def generate_sectioned_image(
    waveform: np.ndarray,
    sr: int,
    target_size: int,
    output_mode: str,
    wavelet_type: str,
    n_sections: int,
    section_layout: str = "None",
    progress_callback=None,
) -> np.ndarray:
    """
    Generate a final square image by processing temporal sections sequentially.

    Available section-combination layouts:
        - None: the whole signal generates one image, with no temporal sectioning.
        - Chronological treemap: the recursive equal-area block layout.
          Each local patch is computed near its target block size.
        - Clockwise circular slices: chronological angular sectors around the
          center. Each section patch is computed at half the final side length
          (one quarter of the final pixel count), then resized and cropped by
          its angular mask.
        - Concentric circles: chronological circular rings with the first
          section at the center and later sections moving outward.
        - Concentric squares: chronological square rings with the first
          section at the center and later sections moving outward.
        - Vertical strips: chronological left-to-right rectangular strips.
        - Horizontal strips: chronological top-to-bottom rectangular strips.

    No explicit contour is drawn between sections. Boundaries are visible only
    through the discontinuity between independently generated section images.
    """
    target_size = int(target_size)
    n_sections = max(1, int(n_sections))

    if output_mode in {"Colors + black drawing", "Mix"}:
        total_steps = 2 if section_layout == "None" else 2 * n_sections

        def color_progress(done: int, total: int) -> None:
            if progress_callback is not None:
                progress_callback(done, total_steps)

        def grayscale_progress(done: int, total: int) -> None:
            if progress_callback is not None:
                progress_callback(total + done, total_steps)

        color_image = generate_sectioned_image(
            waveform=waveform,
            sr=sr,
            target_size=target_size,
            output_mode="Colors",
            wavelet_type=wavelet_type,
            n_sections=n_sections,
            section_layout=section_layout,
            progress_callback=color_progress,
        )
        grayscale_image = generate_sectioned_image(
            waveform=waveform,
            sr=sr,
            target_size=target_size,
            output_mode="Grayscale",
            wavelet_type=wavelet_type,
            n_sections=n_sections,
            section_layout=section_layout,
            progress_callback=grayscale_progress,
        )

        if output_mode == "Colors + black drawing":
            return apply_black_drawing_from_grayscale(color_image, grayscale_image)

        return apply_grayscale_mix_to_color(color_image, grayscale_image)

    section_layout = section_layout if section_layout in SECTION_LAYOUT_OPTIONS else "None"

    if section_layout == "None":
        patch = generate_section_patch(
            section=waveform,
            sr=sr,
            patch_size=target_size,
            output_mode=output_mode,
            wavelet_type=wavelet_type,
        )
        if progress_callback is not None:
            progress_callback(1, 1)
        return finalize_sectioned_image(patch, output_mode)

    sections = split_waveform_into_sections(waveform, n_sections)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.float64)

    if section_layout == "Chronological treemap":
        rectangles = recursive_chronological_layout(0, 0, target_size, target_size, 0, n_sections)

        for idx, rect in enumerate(rectangles):
            section = sections[rect["section"]]
            local_size = max(8, int(max(rect["w"], rect["h"])))
            patch = generate_section_patch(
                section=section,
                sr=sr,
                patch_size=local_size,
                output_mode=output_mode,
                wavelet_type=wavelet_type,
            )
            patch_rect = fit_square_patch_to_rect_float(patch, rect["w"], rect["h"])

            y0, x0 = rect["y"], rect["x"]
            y1, x1 = y0 + rect["h"], x0 + rect["w"]
            canvas[y0:y1, x0:x1] = patch_rect[:rect["h"], :rect["w"]]

            if progress_callback is not None:
                progress_callback(idx + 1, n_sections)

        return finalize_sectioned_image(canvas, output_mode)

    index_map = build_layout_index_map(target_size, n_sections, section_layout)
    if index_map is None:
        raise ValueError(f"Unsupported section layout: {section_layout}")

    if section_layout == "Clockwise circular slices":
        patch_size = max(8, target_size // 2)
    else:
        patch_size = target_size

    for idx, section in enumerate(sections):
        patch = generate_section_patch(
            section=section,
            sr=sr,
            patch_size=patch_size,
            output_mode=output_mode,
            wavelet_type=wavelet_type,
        )
        patch_full = resize_float_image_to_square(patch, target_size)
        mask = index_map == idx
        canvas[mask] = patch_full[mask]

        if progress_callback is not None:
            progress_callback(idx + 1, n_sections)

    return finalize_sectioned_image(canvas, output_mode)

def output_image_fourier_to_display_images(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the classical 2D Fourier transform of the final output image.

    For RGB outputs, the transform is computed on a luminance image so that the
    displayed magnitude and phase summarize the spatial structure of the final
    image itself, not the local spectra used internally for block generation.
    """
    image = np.asarray(image_rgb, dtype=np.float64)
    if image.ndim == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image
    gray = normalize_to_unit(gray)
    F = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(F))
    phase = np.angle(F)
    return (
        fourier_grid_to_display_image(magnitude, "viridis"),
        fourier_grid_to_display_image(phase, "twilight"),
    )


def image_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    """Encode an RGB image as PNG bytes for Streamlit download buttons."""
    buffer = io.BytesIO()
    Image.fromarray(image_rgb.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================
# Visualization helpers
# ============================================================

def waveform_to_display_image(waveform: np.ndarray, width: int = 512, height: int = 80) -> np.ndarray:
    """
    Render the waveform as an (H, W, 3) uint8 oscillogram on a dark background.

    Each pixel column x corresponds to one downsampled time index.
    A vertical bar is drawn from the zero-amplitude midline to the
    normalized amplitude value, producing a standard time-domain plot.
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


def spectrogram_to_display_image(waveform: np.ndarray, width: int = 512, height: int = 160) -> np.ndarray:
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


def fourier_grid_to_display_image(grid: np.ndarray, colormap: str, width: int = 512, height: int = 512) -> np.ndarray:
    """
    Render a 2D Fourier grid (magnitude or phase) as an (H, W, 3) uint8 image.

    For phase grids in (−π, π], the field is normalized to [0, 1] before
    applying the colormap (circular colormaps like "twilight" are recommended).
    For magnitude grids in [0, 1], the field is passed through directly.
    """
    normalized = normalize_to_unit(grid)
    return apply_colormap(interpolate_to_shape(normalized, height, width), colormap)


# ============================================================
# Page configuration and CSS
# ============================================================

def configure_page() -> None:
    st.set_page_config(
        page_title="Audio Visualization",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 100%;
            padding-top: 3.10rem;
            padding-left: 1.35rem;
            padding-right: 1.35rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stTabs"] [role="tablist"] {
            margin-top: 0rem;
            gap: 0.35rem;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            padding-top: 0.45rem;
            padding-bottom: 0.45rem;
        }
        div[data-testid="stButton"] > button {
            border-radius: 0.30rem;
            min-height: 2.40rem;
            white-space: normal;
            text-align: center;
        }
        div[data-testid="stButton"] > button[kind="primary"] { font-weight: 650; }
        h2 {
            text-align: center;
            border: 1px solid rgba(49, 51, 63, 0.20);
            border-radius: 0.30rem;
            padding: 0.55rem 0.75rem;
            margin-top: 0.25rem;
            margin-bottom: 1.00rem;
            background: rgba(49, 51, 63, 0.04);
        }
        .small-muted { color: #6b7280; font-size: 0.88rem; }

        /* ---- Portfolio link row ---- */
        .portfolio-link-row {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 0.42rem;
            min-height: 2.35rem;
            margin: 0 0 -2.65rem 0;
            padding-right: 0.15rem;
            position: relative;
            z-index: 20;
        }
        .portfolio-link,
        .portfolio-link:visited {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            height: 2rem !important;
            border: 1px solid rgba(250, 250, 250, 0.22) !important;
            border-radius: 0.45rem !important;
            color: inherit !important;
            text-decoration: none !important;
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            line-height: 1 !important;
            background: rgba(255, 255, 255, 0.03) !important;
            white-space: nowrap !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }
        .portfolio-link:hover {
            border-color: rgb(255, 75, 75) !important;
            color: rgb(255, 75, 75) !important;
            background: rgba(255, 75, 75, 0.08) !important;
            text-decoration: none !important;
        }
        .portfolio-link.icon-only,
        .portfolio-link.icon-only:visited {
            width: 2rem !important; min-width: 2rem !important;
            max-width: 2rem !important; padding: 0 !important; gap: 0 !important;
        }
        .portfolio-link.with-label,
        .portfolio-link.with-label:visited {
            width: auto !important; padding: 0 0.58rem !important; gap: 0.38rem !important;
        }
        .portfolio-icon {
            display: block !important;
            width: 1.12rem !important; height: 1.12rem !important;
            min-width: 1.12rem !important; max-width: 1.12rem !important;
            object-fit: contain !important; flex: 0 0 auto !important;
            margin: 0 !important; padding: 0 !important; border: 0 !important;
        }
        .portfolio-label { display: inline-block !important; }
        .portfolio-link.icon-only .portfolio-label {
            display: none !important; width: 0 !important;
            min-width: 0 !important; max-width: 0 !important;
            margin: 0 !important; padding: 0 !important; overflow: hidden !important;
        }
        @media (max-width: 1180px) {
            .portfolio-link-row {
                justify-content: flex-start;
                flex-wrap: wrap;
                margin-bottom: 0.65rem;
                padding-right: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Session state
# ============================================================

def init_session_state() -> None:
    defaults: dict = {
        "audio_bytes":   None,
        "using_default": False,
        "audio_source":  "Default sample",
        "last_audio_context": None,
        "results":       None,
        "doc_fr_title":  DOC_FR_TITLES[0],
        "doc_en_title":  DOC_EN_TITLES[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_audio() -> None:
    """Invalidate stored audio and all downstream results."""
    st.session_state.audio_bytes   = None
    st.session_state.using_default = False
    st.session_state.last_audio_context = None
    st.session_state.results       = None


def clear_results() -> None:
    st.session_state.results = None


def set_doc_section(state_key: str, title: str) -> None:
    st.session_state[state_key] = title


# ============================================================
# Shared render helpers
# ============================================================

def render_image_output(label: str, image: np.ndarray | None, caption: str = "") -> None:
    st.markdown(f"**{label}**")
    if image is None:
        st.empty()
    else:
        st.image(image, width="stretch")
        if caption:
            st.markdown(f'<p class="small-muted">{caption}</p>', unsafe_allow_html=True)


def render_documentation_tab(titles: list[str], sections: dict[str, str], state_key: str) -> None:
    left_col, right_col = st.columns([1, 3], gap="large")
    with left_col:
        for title in titles:
            is_active = st.session_state[state_key] == title
            st.button(
                title,
                key=f"{state_key}_{title}",
                type="primary" if is_active else "secondary",
                width="stretch",
                on_click=set_doc_section,
                args=(state_key, title),
            )
    with right_col:
        st.markdown(sections[st.session_state[state_key]])


# ============================================================
# App tab
# ============================================================

def render_app_tab() -> None:
    if not LIBROSA_AVAILABLE:
        st.error("librosa is not installed. Please add `librosa` to requirements.txt.")
        return

    # --------------------------------------------------------
    # Initialise default audio bytes only when the default source is selected
    # --------------------------------------------------------
    if (
        st.session_state.audio_source == "Default sample"
        and st.session_state.audio_bytes is None
    ):
        def_waveform, def_sr, def_bytes = load_default_audio()
        if def_bytes is not None:
            st.session_state.audio_bytes   = def_bytes
            st.session_state.using_default = True

    # --------------------------------------------------------
    # Three-column layout
    # col1: universal input + Run + collapsed parameters
    # col2: output image + download
    # col3: diagnostic plots in expanders
    # --------------------------------------------------------
    col1, col2, col3 = st.columns([1.0, 1.35, 1.35], gap="large")

    waveform_preview = None
    sr_preview = None
    if st.session_state.audio_bytes is not None:
        try:
            waveform_preview, sr_preview = load_audio(st.session_state.audio_bytes)
        except Exception:
            waveform_preview, sr_preview = None, None

    # ---- Column 1: universal input + Run + parameters ----
    with col1:
        with st.container(border=True):
            st.markdown("#### Input audio signal")

            source_choice = st.radio(
                "Audio source",
                options=["Default sample", "Upload file", "Record audio"],
                horizontal=True,
                key="audio_source",
                on_change=clear_audio,
            )

            if source_choice == "Default sample":
                def_waveform, def_sr, def_bytes = load_default_audio()
                if def_bytes is not None:
                    if st.session_state.audio_bytes != def_bytes:
                        st.session_state.audio_bytes = def_bytes
                        st.session_state.using_default = True
                        st.session_state.last_audio_context = None
                        st.session_state.results = None
                        st.rerun()
                    st.caption(f"Default sample: {DEFAULT_AUDIO_TITLE} — {DEFAULT_AUDIO_DESCRIPTION}")
                    st.audio(def_bytes)
                else:
                    st.error("The default audio sample could not be loaded.")
                    st.session_state.audio_bytes = None
                    st.session_state.using_default = False

            elif source_choice == "Upload file":
                uploaded_file = st.file_uploader(
                    "Upload an audio file",
                    type=AUDIO_TYPES,
                    key="audio_upload",
                )
                if uploaded_file is not None:
                    uploaded_bytes = uploaded_file.getvalue()
                    if st.session_state.audio_bytes != uploaded_bytes:
                        st.session_state.audio_bytes = uploaded_bytes
                        st.session_state.using_default = False
                        st.session_state.last_audio_context = None
                        st.session_state.results = None
                        st.rerun()
                    st.audio(uploaded_bytes)
                else:
                    st.info("Upload an audio file to use a personal signal.")
                    st.session_state.audio_bytes = None
                    st.session_state.using_default = False

            else:
                try:
                    recorded = st.audio_input(
                        "Record audio",
                        key="audio_record",
                    )
                    if recorded is not None:
                        recorded_bytes = recorded.getvalue()
                        if st.session_state.audio_bytes != recorded_bytes:
                            st.session_state.audio_bytes = recorded_bytes
                            st.session_state.using_default = False
                            st.session_state.last_audio_context = None
                            st.session_state.results = None
                            st.rerun()
                        st.audio(recorded_bytes)
                    else:
                        st.info("Record an audio signal with your microphone.")
                        st.session_state.audio_bytes = None
                        st.session_state.using_default = False
                except AttributeError:
                    st.warning("Audio recording requires a newer Streamlit version.")
                    st.session_state.audio_bytes = None
                    st.session_state.using_default = False

        run_clicked = st.button(
            "Run",
            type="primary",
            width="stretch",
            disabled=(st.session_state.audio_bytes is None),
        )

        with st.expander("Parameters", expanded=False):
            target_size = st.slider(
                "Output image size (px)",
                min_value=IMAGE_SIZE_MIN,
                max_value=IMAGE_SIZE_MAX,
                value=IMAGE_SIZE_DEFAULT,
                step=IMAGE_SIZE_STEP,
                key="target_size",
                on_change=clear_results,
            )

            output_mode = st.radio(
                "Output mode",
                options=OUTPUT_MODE_OPTIONS,
                index=3,
                key="output_mode",
                horizontal=True,
                on_change=clear_results,
            )

            section_layout = st.selectbox(
                "Section layout",
                options=SECTION_LAYOUT_OPTIONS,
                index=0,
                key="section_layout",
                on_change=clear_results,
                help=(
                    "Choose how chronological audio sections are combined into "
                    "the final square image."
                ),
            )

            if waveform_preview is None or sr_preview is None:
                n_sections = 1
                st.info(
                    "Choose the default sample, upload an audio file, or record audio "
                    "to enable the section slider."
                )
            else:
                n_samples_preview = len(waveform_preview)
                sr_context = int(sr_preview)
                k_max = compute_max_sections(n_samples_preview, target_size)
                audio_context = (
                    st.session_state.audio_source,
                    len(st.session_state.audio_bytes or b""),
                    int(n_samples_preview),
                    sr_context,
                    int(target_size),
                    int(k_max),
                )

                # Streamlit keeps widget values in session state. When the input
                # signal changes, k_max can become smaller than the previous slider
                # value. Using a context-dependent key recreates the slider safely
                # and prevents the "value outside bounds" exception.
                slider_context = "_".join(str(item).replace(" ", "_") for item in audio_context)
                n_sections_key = f"n_sections_{slider_context}"
                n_sections_default = min(DEFAULT_SECTIONS, k_max)
                st.session_state.last_audio_context = audio_context

                if section_layout == "None":
                    n_sections = 1
                    st.slider(
                        "Number of sections",
                        min_value=1,
                        max_value=max(1, k_max),
                        value=1,
                        step=1,
                        key=f"n_sections_disabled_{slider_context}",
                        disabled=True,
                        help=(
                            "Sectioning is disabled when Section layout is set to None. "
                            "The whole signal is used to generate a single image."
                        ),
                    )
                else:
                    n_sections = st.slider(
                        "Number of sections",
                        min_value=1,
                        max_value=k_max,
                        value=n_sections_default,
                        step=1,
                        key=n_sections_key,
                        on_change=clear_results,
                        help=(
                            "The signal is split into chronological sections. Each section "
                            "generates one visible block of the final image. The maximum "
                            "depends on the number of samples and the output image size."
                        ),
                    )

                st.caption(
                    f"Samples: {n_samples_preview} · SR: {sr_context} Hz · "
                    f"Dynamic maximum sections: {k_max}"
                )

            wavelet_type = st.selectbox(
                "CWT wavelet",
                options=WAVELET_OPTIONS,
                index=0,
                key="wavelet_type",
                on_change=clear_results,
                help=(
                    "Morlet: analytic complex wavelet — contributes both magnitude "
                    "and instantaneous phase to the Fourier grids.\n"
                    "Ricker: real-valued wavelet — contributes magnitude only; "
                    "phase weight is redistributed to the STFT sources."
                ),
            )


    # ---- Run computation ----
    if run_clicked and st.session_state.audio_bytes is not None:
        with st.spinner("Loading audio…"):
            waveform, sr = load_audio(st.session_state.audio_bytes)

        if waveform is None or sr is None:
            st.error("Could not decode the audio file.")
        else:
            k_max_runtime = compute_max_sections(len(waveform), target_size)
            n_sections_runtime = 1 if section_layout == "None" else min(max(1, int(n_sections)), k_max_runtime)

            with col2:
                progress_text = st.empty()
                progress_bar = st.progress(0.0)

            def update_progress(done: int, total: int) -> None:
                with col2:
                    progress_text.markdown(
                        f'<p class="small-muted">Computing section {done}/{total}…</p>',
                        unsafe_allow_html=True,
                    )
                    progress_bar.progress(done / max(1, total))

            with st.spinner("Generating section-by-section image…"):
                image_rgb = generate_sectioned_image(
                    waveform=waveform,
                    sr=sr,
                    target_size=target_size,
                    output_mode=output_mode,
                    wavelet_type=wavelet_type,
                    n_sections=n_sections_runtime,
                    section_layout=section_layout,
                    progress_callback=update_progress,
                )

            with col2:
                progress_bar.empty()
                progress_text.empty()

            duration = len(waveform) / sr
            input_waveform_display = waveform_to_display_image(waveform)
            input_spectrogram_display = spectrogram_to_display_image(waveform)
            output_mag_display, output_phase_display = output_image_fourier_to_display_images(image_rgb)

            st.session_state.results = {
                "input_waveform_display": input_waveform_display,
                "input_spectrogram_display": input_spectrogram_display,
                "output_mag_display": output_mag_display,
                "output_phase_display": output_phase_display,
                "generated_image": image_rgb,
                "png_bytes": image_to_png_bytes(image_rgb),
                "duration": duration,
                "sr": sr,
                "n_samples": len(waveform),
                "n_sections": n_sections_runtime,
                "section_layout": section_layout,
                "output_mode": output_mode,
            }

    results = st.session_state.results

    # ---- Column 2: output only ----
    with col2:
        if results is None:
            st.info("Generated image will appear here after you click **Run**.")
        else:
            render_image_output("Generated image", results["generated_image"])
            st.download_button(
                "Download image (PNG)",
                data=results["png_bytes"],
                file_name="output.png",
                mime="image/png",
                width="stretch",
            )

    # ---- Column 3: diagnostic plots ----
    with col3:
        if results is None:
            with st.expander("Input signal plots", expanded=False):
                st.info("Input plots will appear after computation.")
            with st.expander("Fourier plots of the output image", expanded=False):
                st.info("Output-image Fourier plots will appear after computation.")
        else:
            with st.expander("Input signal plots", expanded=False):
                render_image_output("Waveform", results["input_waveform_display"])
                render_image_output("Log-magnitude spectrogram", results["input_spectrogram_display"])
                st.markdown(
                    f'<p class="small-muted">'
                    f'Duration: {results["duration"]:.2f} s &nbsp;·&nbsp; '
                    f'SR: {results["sr"]} Hz &nbsp;·&nbsp; '
                    f'Samples: {results["n_samples"]} &nbsp;·&nbsp; '
                    f'Sections: {results["n_sections"]} &nbsp;·&nbsp; '
                    f'Layout: {html.escape(results.get("section_layout", "None"))}'
                    f'</p>',
                    unsafe_allow_html=True,
                )

            with st.expander("Fourier plots of the output image", expanded=False):
                render_image_output("2D Fourier magnitude of the output image", results["output_mag_display"])
                render_image_output("2D Fourier phase of the output image", results["output_phase_display"])


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    configure_page()
    init_session_state()
    render_portfolio_links()

    app_tab, doc_fr_tab, doc_en_tab = st.tabs([
        "App",
        "Documentation FR",
        "Documentation EN",
    ])

    with app_tab:
        render_app_tab()

    with doc_fr_tab:
        render_documentation_tab(DOC_FR_TITLES, DOC_FR_SECTIONS, "doc_fr_title")

    with doc_en_tab:
        render_documentation_tab(DOC_EN_TITLES, DOC_EN_SECTIONS, "doc_en_title")


if __name__ == "__main__":
    main()
