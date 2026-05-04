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
import scipy.cluster.vq
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

try:
    from skimage import segmentation as skimage_seg
    SKIMAGE_AVAILABLE = True
except Exception:
    skimage_seg = None
    SKIMAGE_AVAILABLE = False

try:
    from sklearn.cluster import MeanShift, estimate_bandwidth
    SKLEARN_AVAILABLE = True
except Exception:
    MeanShift = None
    estimate_bandwidth = None
    SKLEARN_AVAILABLE = False


# ============================================================
# Constants
# ============================================================

# DEFAULT_AUDIO_SR = 22_050 Hz: standard mono sample rate. The Nyquist
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
IMAGE_SIZE_MAX:     int = 1024
IMAGE_SIZE_DEFAULT: int = 512
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

WAVELET_OPTIONS        = ["Morlet", "Ricker (Mexican hat)"]
OUTPUT_MODE_OPTIONS    = ["Grayscale", "Colors", "Black mix", "Luma mix", "Watershed"]
SEGMENTATION_METHODS   = ["Watershed", "K-means", "SLIC", "Felzenszwalb", "Mean-shift"]
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
        "label": "CV FR",
        "url": "https://e.pcloud.link/publink/show?code=XZX81iZss7g3iD9fGJXmPRRGSi7LBTvLcgX",
        "icon_url": "https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg",        
    },
    {
        "platform": "CV EN",
        "label": "CV EN",
        "url": "https://e.pcloud.link/publink/show?code=XZ581iZBQvbu1mFKjziunF9lblghze8OXkk",
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

def get_stft_resolutions(n_samples: int, params: dict | None = None) -> list[int]:
    """
    Return valid powers-of-two STFT window sizes under user-controlled bounds.
    """
    stft_min = int(get_param(params, "stft_n_fft_min", STFT_N_FFT_MIN))
    stft_max = int(get_param(params, "stft_n_fft_max", STFT_N_FFT_MAX))

    stft_min = int(2 ** round(np.log2(max(64, stft_min))))
    stft_max = int(2 ** round(np.log2(max(stft_min, stft_max))))

    upper = min(stft_max, max(stft_min, n_samples // 2))
    resolutions = [
        2 ** k
        for k in range(6, 15)   # 64 … 16384, then filtered by user limits
        if stft_min <= 2 ** k <= upper
    ]
    return resolutions if resolutions else [stft_min]





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

def extract_features(waveform: np.ndarray, sr: int, wavelet_type: str = "Morlet", params: dict | None = None, step_callback=None) -> dict:
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
    resolutions = get_stft_resolutions(len(waveform), params=params)
    features["stft_resolutions"] = resolutions

    for n_fft in resolutions:
        if step_callback is not None:
            step_callback(f"STFT  N={n_fft}")
        hop = n_fft // 4
        stft = librosa.stft(
            waveform, n_fft=n_fft, hop_length=hop, window="hann", center=True,
        )
        features[f"mag_{n_fft}"]   = np.abs(stft)
        features[f"phase_{n_fft}"] = np.angle(stft)

    # --- CWT ---
    cwt_max_samples = int(get_param(params, "cwt_max_samples", CWT_MAX_SAMPLES))
    cwt_n_scales = int(get_param(params, "cwt_n_scales", CWT_N_SCALES))
    n_mels = int(get_param(params, "n_mels", N_MELS))
    n_mfcc = int(get_param(params, "n_mfcc", N_MFCC))
    cwt_max_samples = max(512, cwt_max_samples)
    cwt_n_scales = max(8, cwt_n_scales)
    n_mels = max(16, n_mels)
    n_mfcc = max(4, n_mfcc)
    step_cwt = max(1, len(waveform) // cwt_max_samples)
    waveform_cwt = waveform[::step_cwt].copy()
    n_cwt = len(waveform_cwt)

    if step_callback is not None:
        step_callback(f"CWT  ({wavelet_type},  S={cwt_n_scales})")
    max_scale = min(512, n_cwt // 2)
    cwt_scales = np.geomspace(1.0, max(1.0, float(max_scale)), num=cwt_n_scales)

    if wavelet_type == "Morlet":
        wavelet_fn = lambda M, s: morlet2_compat(M, s, w=CWT_MORLET_W)
    else:
        wavelet_fn = ricker_compat

    cwt_coeffs = cwt_compat(waveform_cwt, wavelet_fn, cwt_scales)

    features["cwt_magnitude"] = np.abs(cwt_coeffs)
    features["cwt_phase"] = np.angle(cwt_coeffs) if wavelet_type == "Morlet" else None

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

    if step_callback is not None:
        step_callback("Building Fourier grids")

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



def get_param(params: dict | None, key: str, default):
    """Read a user parameter with a safe fallback."""
    if params is None:
        return default
    return params.get(key, default)


def normalize_positive_weights(weight_dict: dict[str, float], fallback: dict[str, float]) -> dict[str, float]:
    """Normalize non-negative weights; fall back to defaults if their sum is zero."""
    cleaned = {key: max(0.0, float(value)) for key, value in weight_dict.items()}
    total = sum(cleaned.values())
    if total <= 1e-12:
        cleaned = {key: max(0.0, float(value)) for key, value in fallback.items()}
        total = sum(cleaned.values())
    if total <= 1e-12:
        n = max(1, len(cleaned))
        return {key: 1.0 / n for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}


def get_magnitude_weights(params: dict | None) -> dict[str, float]:
    """Return normalized user-controlled magnitude feature weights."""
    defaults = {
        "stft": W_STFT_TOTAL,
        "cwt": W_CWT_MAG,
        "mel": W_MEL,
        "chroma": W_CHROMA,
        "mfcc": W_MFCC,
        "rms": W_RMS,
    }
    values = {
        "stft": get_param(params, "mag_weight_stft", W_STFT_TOTAL),
        "cwt": get_param(params, "mag_weight_cwt", W_CWT_MAG),
        "mel": get_param(params, "mag_weight_mel", W_MEL),
        "chroma": get_param(params, "mag_weight_chroma", W_CHROMA),
        "mfcc": get_param(params, "mag_weight_mfcc", W_MFCC),
        "rms": get_param(params, "mag_weight_rms", W_RMS),
    }
    return normalize_positive_weights(values, defaults)


def get_phase_weights(params: dict | None, has_cwt_phase: bool) -> dict[str, float]:
    """Return normalized user-controlled phase feature weights."""
    defaults = {
        "stft_mid": W_PHASE_STFT_1024,
        "stft_fine": W_PHASE_STFT_512,
        "cwt": W_PHASE_CWT,
        "onset": W_PHASE_ONSET,
        "centroid": W_PHASE_CENTROID,
        "zcr": W_PHASE_ZCR,
    }
    values = {
        "stft_mid": get_param(params, "phase_weight_stft_mid", W_PHASE_STFT_1024),
        "stft_fine": get_param(params, "phase_weight_stft_fine", W_PHASE_STFT_512),
        "cwt": get_param(params, "phase_weight_cwt", W_PHASE_CWT),
        "onset": get_param(params, "phase_weight_onset", W_PHASE_ONSET),
        "centroid": get_param(params, "phase_weight_centroid", W_PHASE_CENTROID),
        "zcr": get_param(params, "phase_weight_zcr", W_PHASE_ZCR),
    }

    if not has_cwt_phase:
        cwt_weight = max(0.0, float(values.get("cwt", 0.0)))
        values["cwt"] = 0.0
        stft_total = max(1e-12, max(0.0, values["stft_mid"]) + max(0.0, values["stft_fine"]))
        values["stft_mid"] += cwt_weight * max(0.0, values["stft_mid"]) / stft_total
        values["stft_fine"] += cwt_weight * max(0.0, values["stft_fine"]) / stft_total

    return normalize_positive_weights(values, defaults)


def get_rgb_bands(params: dict | None) -> list[tuple[float, float]]:
    """Return user-controlled low/mid/high frequency bands for RGB synthesis."""
    low_end = float(get_param(params, "rgb_low_end", 1.0 / 3.0))
    high_start = float(get_param(params, "rgb_high_start", 2.0 / 3.0))
    low_end = float(np.clip(low_end, 0.05, 0.90))
    high_start = float(np.clip(high_start, 0.10, 0.95))
    if high_start <= low_end + 0.05:
        mid = 0.5 * (low_end + high_start)
        low_end = max(0.05, mid - 0.025)
        high_start = min(0.95, mid + 0.025)
    return [(0.0, low_end), (low_end, high_start), (high_start, 1.0)]


def apply_rgb_balance(image: np.ndarray, params: dict | None) -> np.ndarray:
    """Apply user-controlled RGB channel gains."""
    gains = np.array([
        float(get_param(params, "rgb_balance_r", 1.0)),
        float(get_param(params, "rgb_balance_g", 1.0)),
        float(get_param(params, "rgb_balance_b", 1.0)),
    ], dtype=np.float64)
    return np.asarray(image, dtype=np.float64) * gains.reshape(1, 1, 3)


def apply_global_image_adjustments(image: np.ndarray, params: dict | None, is_grayscale: bool = False) -> np.ndarray:
    """Apply brightness, contrast, gamma and saturation after global normalization."""
    img = np.asarray(image, dtype=np.float64)
    img = np.clip(img, 0.0, 1.0)

    contrast = float(get_param(params, "contrast_strength", 1.0))
    brightness = float(get_param(params, "brightness_factor", 1.0))
    gamma = float(get_param(params, "gamma_correction", 0.85))
    saturation = float(get_param(params, "saturation_factor", 1.0))

    img = 0.5 + contrast * (img - 0.5)
    img = img * brightness
    img = np.clip(img, 0.0, 1.0)

    if gamma > 1e-6:
        img = img ** gamma

    if not is_grayscale and img.ndim == 3 and img.shape[2] == 3:
        gray = (
            0.299 * img[:, :, 0]
            + 0.587 * img[:, :, 1]
            + 0.114 * img[:, :, 2]
        )
        img = gray[:, :, None] + saturation * (img - gray[:, :, None])

    return np.clip(img, 0.0, 1.0)



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
    params: dict | None = None,
) -> np.ndarray:
    """
    Build a (N, N) Fourier magnitude grid from user-weighted audio features.
    """
    weights = get_magnitude_weights(params)
    resolutions = features["stft_resolutions"]
    w_per_stft = weights["stft"] / max(1, len(resolutions))
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
    components.append((weights["cwt"], interpolate_to_shape(cwt_log, target_size, target_size)))

    mel = features["mel"]
    if band is not None:
        mel = slice_band(mel, band)
    mel_log = normalize_to_unit(np.log1p(mel))
    components.append((weights["mel"], interpolate_to_shape(mel_log, target_size, target_size)))

    chroma = normalize_to_unit(np.abs(features["chroma"]))
    components.append((weights["chroma"], interpolate_to_shape(chroma, target_size, target_size)))

    mfcc = normalize_to_unit(np.abs(features["mfcc"]))
    components.append((weights["mfcc"], interpolate_to_shape(mfcc, target_size, target_size)))

    rms_2d = row_to_2d(features["rms"], target_size)
    components.append((weights["rms"], rms_2d))

    combined = sum(w * arr for w, arr in components)
    return normalize_to_unit(combined)


# ============================================================
# 2D Fourier phase grid
# ============================================================

# ============================================================
# 2D Fourier phase grid
# ============================================================

def build_phase_grid(
    features: dict,
    target_size: int,
    band: tuple[float, float] | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """
    Build a (N, N) Fourier phase grid from user-weighted audio features.
    """
    resolutions = features["stft_resolutions"]
    has_cwt_phase = features.get("cwt_phase") is not None
    weights = get_phase_weights(params, has_cwt_phase=has_cwt_phase)

    n_fft_mid = 1024 if 1024 in resolutions else resolutions[len(resolutions) // 2]
    phase_mid = features[f"phase_{n_fft_mid}"]
    if band is not None:
        phase_mid = slice_band(phase_mid, band)
    unwrapped_mid = np.unwrap(np.unwrap(phase_mid, axis=0), axis=1)
    grid_mid = interpolate_to_shape(unwrapped_mid, target_size, target_size)

    n_fft_fine = 512 if 512 in resolutions else resolutions[0]
    phase_fine = features[f"phase_{n_fft_fine}"]
    if band is not None:
        phase_fine = slice_band(phase_fine, band)
    unwrapped_fine = np.unwrap(np.unwrap(phase_fine, axis=0), axis=1)
    grid_fine = interpolate_to_shape(unwrapped_fine, target_size, target_size)

    if has_cwt_phase:
        cwt_phase = features["cwt_phase"]
        if band is not None:
            cwt_phase = slice_band(cwt_phase, band)
        cwt_unwrapped = np.unwrap(np.unwrap(cwt_phase, axis=0), axis=1)
        grid_cwt = interpolate_to_shape(cwt_unwrapped, target_size, target_size)
    else:
        grid_cwt = np.zeros((target_size, target_size), dtype=np.float64)

    onset_2d = row_to_2d(features["onset_strength"], target_size) * (np.pi / 2.0)
    centroid_2d = row_to_2d(features["spectral_centroid"], target_size) * np.pi
    zcr_2d = row_to_2d(features["zcr"], target_size) * (np.pi / 4.0)

    phase_combined = (
        weights["stft_mid"] * grid_mid
        + weights["stft_fine"] * grid_fine
        + weights["cwt"] * grid_cwt
        + weights["onset"] * onset_2d
        + weights["centroid"] * centroid_2d
        + weights["zcr"] * zcr_2d
    )

    return (phase_combined + np.pi) % (2.0 * np.pi) - np.pi


# ============================================================
# Hermitian symmetry enforcement
# ============================================================

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
    params: dict | None = None,
) -> np.ndarray:
    """
    Generate an unnormalized floating-point image patch from extracted features.
    """
    if output_mode == "Grayscale":
        mag_2d_raw = build_magnitude_grid(features, target_size, band=None, params=params)
        phase_2d_raw = build_phase_grid(features, target_size, band=None, params=params)
        channel, _ = reconstruct_channel_raw_and_spectrum(mag_2d_raw, phase_2d_raw)
        return np.stack([channel, channel, channel], axis=2)

    channels = []
    for band in get_rgb_bands(params):
        mag_2d_raw = build_magnitude_grid(features, target_size, band=band, params=params)
        phase_2d_raw = build_phase_grid(features, target_size, band=band, params=params)
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
        mag_2d_raw   = build_magnitude_grid(features, target_size, band=None, params=None)
        phase_2d_raw = build_phase_grid(features, target_size, band=None, params=None)
        channel, Z_sym = reconstruct_channel_and_spectrum(mag_2d_raw, phase_2d_raw)
        mag_2d_used, phase_2d_used = spectrum_to_centered_magnitude_phase(Z_sym)
        gray = (channel * 255).astype(np.uint8)
        image_gray_rgb = np.stack([gray, gray, gray], axis=2)
        return image_gray_rgb, mag_2d_used, phase_2d_used

    bands    = [(0.00, 1/3), (1/3, 2/3), (2/3, 1.00)]
    channels = []
    mag_2d_ref = phase_2d_ref = None

    for i, band in enumerate(bands):
        mag_2d_raw   = build_magnitude_grid(features, target_size, band=band, params=None)
        phase_2d_raw = build_phase_grid(features, target_size, band=band, params=None)
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
    params: dict | None = None,
    step_callback=None,
) -> np.ndarray:
    """Generate one unnormalized floating-point square patch from one audio section."""
    patch_size = max(8, int(patch_size))
    features = extract_features(section, sr, wavelet_type=wavelet_type, params=params, step_callback=step_callback)
    if step_callback is not None:
        step_callback("IFFT2 reconstruction")
    return audio_to_image_float(
        features=features,
        target_size=patch_size,
        output_mode=output_mode,
        params=params,
    )


def finalize_sectioned_image(canvas_float: np.ndarray, output_mode: str, params: dict | None = None) -> np.ndarray:
    """
    Apply one global robust normalization after all sections are assembled,
    followed by user-controlled brightness/contrast/gamma/saturation.
    """
    canvas_float = np.asarray(canvas_float, dtype=np.float64)

    lower = float(get_param(params, "robust_lower_percentile", 1.0))
    upper = float(get_param(params, "robust_upper_percentile", 99.0))
    if upper <= lower + 0.1:
        upper = min(100.0, lower + 0.1)

    if output_mode == "Grayscale":
        gray = normalize_to_unit_robust(canvas_float[:, :, 0], lower, upper)
        image = np.stack([gray, gray, gray], axis=2)
        image = apply_global_image_adjustments(image, params, is_grayscale=True)
    else:
        normalization_mode = str(get_param(params, "rgb_normalization_mode", "Per-channel"))
        if normalization_mode == "Shared":
            image = normalize_to_unit_robust(canvas_float, lower, upper)
        else:
            channels = [
                normalize_to_unit_robust(canvas_float[:, :, c], lower, upper)
                for c in range(3)
            ]
            image = np.stack(channels, axis=2)

        image = apply_rgb_balance(image, params)
        image = apply_global_image_adjustments(image, params, is_grayscale=False)

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


def apply_black_drawing_from_grayscale(
    color_image: np.ndarray,
    grayscale_image: np.ndarray,
    params: dict | None = None,
) -> np.ndarray:
    """
    Use a grayscale-generated image as a sparse black drawing mask over a color image.
    """
    color = np.asarray(color_image, dtype=np.uint8).copy()
    gray_rgb = np.asarray(grayscale_image, dtype=np.float64)

    if gray_rgb.ndim == 3:
        gray = 0.299 * gray_rgb[:, :, 0] + 0.587 * gray_rgb[:, :, 1] + 0.114 * gray_rgb[:, :, 2]
    else:
        gray = gray_rgb

    gray = normalize_to_unit(gray)
    smooth_sigma = float(get_param(params, "ink_smoothing_sigma", 0.0))
    if smooth_sigma > 0:
        gray = scipy.ndimage.gaussian_filter(gray, sigma=smooth_sigma)

    threshold = otsu_threshold_unit(gray)
    low_mask = gray <= threshold
    high_mask = gray > threshold

    class_choice = str(get_param(params, "ink_class_choice", "Automatic minority"))
    low_count = int(np.count_nonzero(low_mask))
    high_count = int(np.count_nonzero(high_mask))

    if low_count == 0 and high_count == 0:
        return color
    if class_choice == "Dark class":
        candidate_mask = low_mask
        keep_high_extreme = False
    elif class_choice == "Bright class":
        candidate_mask = high_mask
        keep_high_extreme = True
    elif low_count == 0:
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

    keep_percent = float(get_param(params, "ink_keep_percentile", 50.0))
    keep_percent = float(np.clip(keep_percent, 1.0, 100.0))

    if keep_high_extreme:
        threshold2 = float(np.percentile(candidate_values, 100.0 - keep_percent))
        drawing_mask = candidate_mask & (gray >= threshold2)
    else:
        threshold2 = float(np.percentile(candidate_values, keep_percent))
        drawing_mask = candidate_mask & (gray <= threshold2)

    thickness = int(get_param(params, "ink_thickness", 0))
    if thickness > 0:
        drawing_mask = scipy.ndimage.binary_dilation(drawing_mask, iterations=thickness)

    color[drawing_mask] = 0
    return color


def apply_grayscale_mix_to_color(
    color_image: np.ndarray,
    grayscale_image: np.ndarray,
    params: dict | None = None,
) -> np.ndarray:
    """
    Use a grayscale-generated image as a multiplicative luminance coefficient map.
    """
    color = np.asarray(color_image, dtype=np.float64)
    gray_rgb = np.asarray(grayscale_image, dtype=np.float64)

    if gray_rgb.ndim == 3:
        gray = 0.299 * gray_rgb[:, :, 0] + 0.587 * gray_rgb[:, :, 1] + 0.114 * gray_rgb[:, :, 2]
    else:
        gray = gray_rgb

    coeff = normalize_to_unit(gray)

    blur_sigma = float(get_param(params, "luma_coeff_blur_sigma", 0.0))
    if blur_sigma > 0:
        coeff = scipy.ndimage.gaussian_filter(coeff, sigma=blur_sigma)
        coeff = normalize_to_unit(coeff)

    coeff_gamma = float(get_param(params, "luma_gamma", 1.0))
    if coeff_gamma > 1e-6:
        coeff = coeff ** coeff_gamma

    min_coeff = float(get_param(params, "luma_min_coeff", 0.0))
    min_coeff = float(np.clip(min_coeff, 0.0, 1.0))
    coeff = min_coeff + (1.0 - min_coeff) * coeff

    strength = float(get_param(params, "luma_strength", 1.0))
    strength = float(np.clip(strength, 0.0, 1.0))
    effective_coeff = (1.0 - strength) + strength * coeff

    mixed = color * effective_coeff[:, :, None]
    return np.clip(mixed, 0.0, 255.0).round().astype(np.uint8)



def watershed_flood_from_markers(gradient: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    Lightweight marker-controlled watershed implemented with NumPy + heapq.

    The gradient image is interpreted as a topographic surface. Marker labels
    are flooded outward in order of increasing accumulated gradient cost.
    """
    import heapq

    gradient = np.asarray(gradient, dtype=np.float64)
    markers = np.asarray(markers, dtype=np.int32)
    h, w = gradient.shape

    labels = markers.copy()
    heap: list[tuple[float, int, int, int]] = []

    marker_positions = np.argwhere(markers > 0)
    for y, x in marker_positions:
        lab = int(markers[y, x])
        heapq.heappush(heap, (float(gradient[y, x]), int(y), int(x), lab))

    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while heap:
        cost, y, x, lab = heapq.heappop(heap)
        if labels[y, x] != lab:
            continue

        for dy, dx in neighbors:
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= h or xx < 0 or xx >= w:
                continue

            if labels[yy, xx] == 0:
                labels[yy, xx] = lab
                new_cost = max(cost, float(gradient[yy, xx]))
                heapq.heappush(heap, (new_cost, yy, xx, lab))

    return labels


def _apply_region_coloring_and_boundaries(
    image: np.ndarray,
    labels: np.ndarray,
    params: dict | None = None,
) -> np.ndarray:
    """
    Shared post-processing for any pixel-level segmentation label map.

    Given a uint8 (H, W, 3) image and an integer (H, W) label array, assigns
    each region a representative color (random pixel, mean, or median) and
    optionally draws boundaries.

    Always defaults to random-pixel coloring: for noise-like images the mean/
    median of any large region converges toward grey, so selecting a single
    actual pixel from the region preserves the full color diversity of the
    source image.
    """
    image = np.asarray(image, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int32)
    h, w = image.shape[:2]

    color_mode = str(get_param(params, "seg_region_color_mode", "Random pixel"))
    seed = int(get_param(params, "seg_random_seed", 12345))
    rng = np.random.default_rng(seed)
    out = np.zeros_like(image, dtype=np.uint8)

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        region_colors = image[ys, xs].astype(np.float64)
        if color_mode == "Mean color":
            sampled = np.mean(region_colors, axis=0)
        elif color_mode == "Median color":
            sampled = np.median(region_colors, axis=0)
        else:   # default: Random pixel
            idx = int(rng.integers(0, ys.size))
            sampled = image[ys[idx], xs[idx]].astype(np.float64)
        out[ys, xs] = np.clip(sampled, 0.0, 255.0).round().astype(np.uint8)

    # Boundaries
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:, 1:]  |= labels[:, 1:]  != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:]  != labels[:, :-1]
    boundary[1:, :]  |= labels[1:, :]  != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :]  != labels[:-1, :]

    boundary_style = str(get_param(params, "seg_boundary_style", "None"))
    thickness = int(get_param(params, "seg_boundary_thickness", 0))
    if thickness > 0:
        boundary = scipy.ndimage.binary_dilation(boundary, iterations=thickness)

    if boundary_style == "Black":
        out[boundary] = 0
    elif boundary_style == "Local mean":
        window = int(get_param(params, "seg_boundary_mean_window", 5))
        window = max(3, window | 1)   # ensure odd
        local_mean = np.stack(
            [scipy.ndimage.uniform_filter(out[:, :, c].astype(np.float64), size=window, mode="nearest")
             for c in range(3)],
            axis=2,
        )
        out[boundary] = np.clip(local_mean[boundary], 0.0, 255.0).round().astype(np.uint8)

    return out


def make_kmeans_region_image(image_rgb: np.ndarray, params: dict | None = None) -> np.ndarray:
    """
    Segment image_rgb via K-means clustering in RGB space (scipy.cluster.vq).

    Pixels are reshaped to (N, 3), whitened, and clustered into k centroids.
    Each pixel is assigned the label of its nearest centroid. Region color is
    then chosen by _apply_region_coloring_and_boundaries (random pixel by
    default to avoid grey convergence in noise-like images).
    """
    image = np.asarray(image_rgb, dtype=np.uint8)
    h, w = image.shape[:2]

    k = int(get_param(params, "kmeans_k", 120))
    k = max(2, min(k, h * w))

    pixels = image.reshape(-1, 3).astype(np.float32)
    whitened = scipy.cluster.vq.whiten(pixels)

    _, labels = scipy.cluster.vq.kmeans2(
        whitened,
        k=k,
        iter=10,
        minit="points",
        seed=int(get_param(params, "seg_random_seed", 12345)),
    )
    label_map = labels.reshape(h, w).astype(np.int32)
    return _apply_region_coloring_and_boundaries(image, label_map, params)


def make_slic_region_image(image_rgb: np.ndarray, params: dict | None = None) -> np.ndarray:
    """
    Segment image_rgb using SLIC superpixels (skimage.segmentation.slic).

    SLIC iteratively clusters pixels by proximity in a joint (R,G,B,x,y) space,
    producing compact, roughly equal-area superpixels. The compactness parameter
    controls the trade-off between color homogeneity and spatial regularity.
    Falls back to K-means if scikit-image is unavailable.
    """
    if not SKIMAGE_AVAILABLE:
        return make_kmeans_region_image(image_rgb, params)

    image = np.asarray(image_rgb, dtype=np.uint8)
    n_segments = int(get_param(params, "slic_n_segments", 120))
    compactness = float(get_param(params, "slic_compactness", 10.0))
    sigma = float(get_param(params, "slic_sigma", 1.0))

    labels = skimage_seg.slic(
        image,
        n_segments=max(2, n_segments),
        compactness=max(0.01, compactness),
        sigma=max(0.0, sigma),
        start_label=0,
        channel_axis=2,
    )
    return _apply_region_coloring_and_boundaries(image, labels.astype(np.int32), params)


def make_felzenszwalb_region_image(image_rgb: np.ndarray, params: dict | None = None) -> np.ndarray:
    """
    Segment image_rgb using Felzenszwalb's graph-based algorithm
    (skimage.segmentation.felzenszwalb).

    Merges pixels greedily using a minimum spanning tree: two pixels are merged
    when the edge weight between them is small relative to the internal variation
    of their component. The `scale` parameter directly controls region size:
    larger values produce larger, fewer regions. The number of regions is
    determined automatically.
    Falls back to K-means if scikit-image is unavailable.
    """
    if not SKIMAGE_AVAILABLE:
        return make_kmeans_region_image(image_rgb, params)

    image = np.asarray(image_rgb, dtype=np.uint8)
    scale    = float(get_param(params, "felz_scale",    100.0))
    sigma    = float(get_param(params, "felz_sigma",      0.8))
    min_size = int(get_param(params,   "felz_min_size",   20))

    labels = skimage_seg.felzenszwalb(
        image,
        scale=max(1.0, scale),
        sigma=max(0.0, sigma),
        min_size=max(1, min_size),
        channel_axis=2,
    )
    return _apply_region_coloring_and_boundaries(image, labels.astype(np.int32), params)


def make_meanshift_region_image(image_rgb: np.ndarray, params: dict | None = None) -> np.ndarray:
    """
    Segment image_rgb using Mean-shift density estimation (sklearn.cluster.MeanShift).

    Mean-shift is mode-seeking: it iteratively moves each sample toward the
    local density maximum in feature space. The bandwidth controls the size of
    the search window; larger bandwidth → fewer, larger regions.

    Because mean-shift cost is O(N²), the image is downsampled to at most
    64×64 for clustering, then labels are upsampled to the original size via
    nearest-neighbor assignment (each original pixel is assigned to the cluster
    of its spatially nearest downsampled pixel).
    Falls back to K-means if scikit-learn is unavailable.
    """
    if not SKLEARN_AVAILABLE:
        return make_kmeans_region_image(image_rgb, params)

    image = np.asarray(image_rgb, dtype=np.uint8)
    h, w = image.shape[:2]

    max_side = int(get_param(params, "meanshift_max_side", 64))
    max_side = max(16, min(max_side, 128))
    scale = max(1, max(h, w) // max_side)
    small = image[::scale, ::scale]
    sh, sw = small.shape[:2]

    pixels_small = small.reshape(-1, 3).astype(np.float32)
    bw = float(get_param(params, "meanshift_bandwidth", 0.0))
    if bw <= 0:
        bw = float(estimate_bandwidth(pixels_small, quantile=0.2, n_samples=min(500, len(pixels_small))))
    bw = max(1.0, bw)

    ms = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=1)
    ms.fit(pixels_small)
    small_labels = ms.labels_.reshape(sh, sw).astype(np.int32)

    # Upsample labels to original size (nearest-neighbor via repeat)
    label_map = np.repeat(np.repeat(small_labels, scale, axis=0), scale, axis=1)
    label_map = label_map[:h, :w]

    return _apply_region_coloring_and_boundaries(image, label_map, params)


def make_segmentation_image(image_rgb: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Dispatch to the segmentation method selected in params."""
    method = str(get_param(params, "segmentation_method", "Watershed"))
    if method == "K-means":
        return make_kmeans_region_image(image_rgb, params)
    if method == "SLIC":
        return make_slic_region_image(image_rgb, params)
    if method == "Felzenszwalb":
        return make_felzenszwalb_region_image(image_rgb, params)
    if method == "Mean-shift":
        return make_meanshift_region_image(image_rgb, params)
    return make_watershed_region_image(image_rgb, params)


def make_watershed_region_image(image_rgb: np.ndarray, params: dict | None = None) -> np.ndarray:
    """
    Convert an RGB image into a watershed-segmented region image.
    """
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        image = np.stack([image, image, image], axis=2).astype(np.uint8)

    h, w = image.shape[:2]
    n = max(h, w)

    gray = (
        0.299 * image[:, :, 0].astype(np.float64)
        + 0.587 * image[:, :, 1].astype(np.float64)
        + 0.114 * image[:, :, 2].astype(np.float64)
    )
    gray = normalize_to_unit(gray)

    smooth_sigma = float(get_param(params, "watershed_gradient_smoothing", max(0.8, n / 384.0)))
    gray_smooth = scipy.ndimage.gaussian_filter(gray, sigma=max(0.0, smooth_sigma))

    grad_x = scipy.ndimage.sobel(gray_smooth, axis=1)
    grad_y = scipy.ndimage.sobel(gray_smooth, axis=0)
    gradient = normalize_to_unit(np.hypot(grad_x, grad_y))

    cell_size = int(get_param(params, "watershed_marker_spacing", max(12, int(round(n / 14.0)))))
    cell_size = max(4, cell_size)
    markers = np.zeros((h, w), dtype=np.int32)
    label = 1

    for y0 in range(0, h, cell_size):
        y1 = min(h, y0 + cell_size)
        for x0 in range(0, w, cell_size):
            x1 = min(w, x0 + cell_size)
            sub = gradient[y0:y1, x0:x1]
            if sub.size == 0:
                continue
            yy, xx = np.unravel_index(int(np.argmin(sub)), sub.shape)
            markers[y0 + yy, x0 + xx] = label
            label += 1

    labels = watershed_flood_from_markers(gradient, markers)

    # Use unified coloring + boundary helper.
    # Mirror watershed-specific param keys to the shared seg_ keys so the
    # helper can read them, then delegate.
    merged_params = dict(params or {})
    merged_params.setdefault("seg_region_color_mode", str(get_param(params, "watershed_region_color_mode", "Random pixel")))
    merged_params.setdefault("seg_random_seed",       int(get_param(params, "watershed_random_seed",       12345)))
    merged_params.setdefault("seg_boundary_style",    str(get_param(params, "watershed_boundary_style",    "None")))
    merged_params.setdefault("seg_boundary_thickness",int(get_param(params, "watershed_boundary_thickness",0)))
    merged_params.setdefault("seg_boundary_mean_window", int(get_param(params, "watershed_boundary_mean_window", 5)))
    return _apply_region_coloring_and_boundaries(image, labels, merged_params)


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
    params: dict | None = None,
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

    if output_mode in {"Black mix", "Luma mix", "Watershed"}:
        total_steps = 2 if section_layout == "None" else 2 * n_sections

        def color_progress(label: str, done: int, total: int) -> None:
            if progress_callback is not None:
                progress_callback(f"[Color] {label}", done, total_steps)

        def grayscale_progress(label: str, done: int, total: int) -> None:
            if progress_callback is not None:
                progress_callback(f"[Grayscale] {label}", (total_steps // 2) + done, total_steps)

        color_image = generate_sectioned_image(
            waveform=waveform, sr=sr, target_size=target_size,
            output_mode="Colors", wavelet_type=wavelet_type,
            n_sections=n_sections, section_layout=section_layout,
            progress_callback=color_progress, params=params,
        )
        grayscale_image = generate_sectioned_image(
            waveform=waveform, sr=sr, target_size=target_size,
            output_mode="Grayscale", wavelet_type=wavelet_type,
            n_sections=n_sections, section_layout=section_layout,
            progress_callback=grayscale_progress, params=params,
        )

        if output_mode == "Black mix":
            return apply_black_drawing_from_grayscale(color_image, grayscale_image, params=params)

        mixed_image = apply_grayscale_mix_to_color(color_image, grayscale_image, params=params)

        if output_mode == "Luma mix":
            return mixed_image

        if progress_callback is not None:
            seg_method = str(get_param(params, "segmentation_method", "Watershed"))
            progress_callback(f"Segmentation ({seg_method})", total_steps, total_steps)
        return make_segmentation_image(mixed_image, params=params)

    section_layout = section_layout if section_layout in SECTION_LAYOUT_OPTIONS else "None"

    if section_layout == "None":
        def single_step_cb(label: str) -> None:
            if progress_callback is not None:
                progress_callback(label, 0, 1)

        patch = generate_section_patch(
            section=waveform, sr=sr, patch_size=target_size,
            output_mode=output_mode, wavelet_type=wavelet_type,
            params=params, step_callback=single_step_cb,
        )
        if progress_callback is not None:
            progress_callback("Finalizing image", 1, 1)
        return finalize_sectioned_image(patch, output_mode, params=params)

    sections = split_waveform_into_sections(waveform, n_sections)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.float64)

    if section_layout == "Chronological treemap":
        rectangles = recursive_chronological_layout(0, 0, target_size, target_size, 0, n_sections)

        for idx, rect in enumerate(rectangles):
            def make_step_cb(i=idx):
                def cb(label: str) -> None:
                    if progress_callback is not None:
                        progress_callback(f"Section {i+1}/{n_sections} · {label}", i, n_sections)
                return cb

            section = sections[rect["section"]]
            local_size = max(8, int(max(rect["w"], rect["h"])))
            patch = generate_section_patch(
                section=section, sr=sr, patch_size=local_size,
                output_mode=output_mode, wavelet_type=wavelet_type,
                params=params, step_callback=make_step_cb(),
            )
            patch_rect = fit_square_patch_to_rect_float(patch, rect["w"], rect["h"])

            y0, x0 = rect["y"], rect["x"]
            y1, x1 = y0 + rect["h"], x0 + rect["w"]
            canvas[y0:y1, x0:x1] = patch_rect[:rect["h"], :rect["w"]]

            if progress_callback is not None:
                progress_callback(f"Section {idx+1}/{n_sections} · done", idx + 1, n_sections)

        return finalize_sectioned_image(canvas, output_mode, params=params)

    index_map = build_layout_index_map(target_size, n_sections, section_layout)
    if index_map is None:
        raise ValueError(f"Unsupported section layout: {section_layout}")

    if section_layout == "Clockwise circular slices":
        patch_size = max(8, target_size // 2)
    else:
        patch_size = target_size

    for idx, section in enumerate(sections):
        def make_step_cb(i=idx):
            def cb(label: str) -> None:
                if progress_callback is not None:
                    progress_callback(f"Section {i+1}/{n_sections} · {label}", i, n_sections)
            return cb

        patch = generate_section_patch(
            section=section, sr=sr, patch_size=patch_size,
            output_mode=output_mode, wavelet_type=wavelet_type,
            params=params, step_callback=make_step_cb(),
        )
        patch_full = resize_float_image_to_square(patch, target_size)
        mask = index_map == idx
        canvas[mask] = patch_full[mask]

        if progress_callback is not None:
            progress_callback(f"Section {idx+1}/{n_sections} · done", idx + 1, n_sections)

    return finalize_sectioned_image(canvas, output_mode, params=params)

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
        /* ---- Global layout ---- */
        .block-container {
            max-width: 100%;
            padding-top: 2.75rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-bottom: 2.5rem;
        }

        /* ---- App header ---- */
        .app-header {
            display: flex;
            align-items: baseline;
            gap: 0.75rem;
            margin-bottom: 0.15rem;
        }
        .app-title {
            font-weight: 800;
            font-size: 1.65rem;
            letter-spacing: -0.02em;
            line-height: 1;
            color: inherit;
        }
        .app-title-sep {
            font-size: 1.1rem;
            color: #FF4B4B;
            opacity: 0.85;
        }
        .app-subtitle {
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #9ca3af;
            margin-bottom: 1.1rem;
            margin-top: 0.1rem;
        }

        /* ---- Section headings ---- */
        h2 {
            text-align: center;
            border: 1px solid rgba(255, 75, 75, 0.18);
            border-radius: 0.35rem;
            padding: 0.55rem 0.75rem;
            margin-top: 0.25rem;
            margin-bottom: 1.00rem;
            background: rgba(255, 75, 75, 0.04);
        }

        /* ---- Tabs ---- */
        div[data-testid="stTabs"] [role="tablist"] {
            margin-top: 0;
            gap: 0.3rem;
            border-bottom: 1px solid rgba(255, 75, 75, 0.15);
            padding-bottom: 0;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            padding: 0.45rem 1.0rem;
            border-radius: 0.35rem 0.35rem 0 0;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #FF4B4B !important;
            border-bottom: 2px solid #FF4B4B !important;
        }

        /* ---- Buttons ---- */
        div[data-testid="stButton"] > button {
            border-radius: 0.35rem;
            min-height: 2.5rem;
            white-space: normal;
            text-align: center;
            transition: all 0.15s ease;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: none !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:not([disabled]):hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.35);
        }

        /* ---- Containers / boxes ---- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 0.45rem !important;
        }

        /* ---- Expander ---- */
        div[data-testid="stExpander"] summary {
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.04em;
        }

        /* ---- Captions / muted text ---- */
        .small-muted {
            color: #9ca3af;
            font-size: 0.80rem;
            line-height: 1.5;
        }
        .result-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem 1.2rem;
            margin-top: 0.6rem;
        }
        .result-meta-item {
            font-size: 0.78rem;
            color: #9ca3af;
        }
        .result-meta-item span {
            color: #FF4B4B;
            font-weight: 600;
        }

        /* ---- Section label pill ---- */
        .section-pill {
            display: inline-block;
            font-size: 0.70rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            background: rgba(255, 75, 75, 0.10);
            border: 1px solid rgba(255, 75, 75, 0.22);
            border-radius: 0.25rem;
            padding: 0.1rem 0.45rem;
            color: #FF4B4B;
            margin-bottom: 0.4rem;
        }

        /* ---- Advanced params container titles ---- */
        .param-group-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #9ca3af;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 75, 75, 0.12);
            padding-bottom: 0.3rem;
        }

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
            border: 1px solid rgba(250, 250, 250, 0.18) !important;
            border-radius: 0.35rem !important;
            color: inherit !important;
            text-decoration: none !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            line-height: 1 !important;
            background: rgba(255, 255, 255, 0.025) !important;
            white-space: nowrap !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
            transition: all 0.15s ease !important;
        }
        .portfolio-link:hover {
            border-color: #FF4B4B !important;
            color: #FF4B4B !important;
            background: rgba(255, 75, 75, 0.08) !important;
            text-decoration: none !important;
            transform: translateY(-1px) !important;
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
            width: 1.10rem !important; height: 1.10rem !important;
            min-width: 1.10rem !important; max-width: 1.10rem !important;
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
        "audio_bytes":    None,
        "using_default":  False,
        "audio_source":   "Default sample",
        "audio_filename": None,
        "last_audio_context": None,
        "results":        None,
        "run_in_progress": False,
        "run_requested":  False,
        "last_run_status": None,
        "doc_fr_title":   DOC_FR_TITLES[0],
        "doc_en_title":   DOC_EN_TITLES[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_audio() -> None:
    """Invalidate stored audio and all downstream results."""
    st.session_state.audio_bytes = None
    st.session_state.using_default = False
    st.session_state.audio_filename = None
    st.session_state.last_audio_context = None
    st.session_state.results = None
    st.session_state.run_in_progress = False
    st.session_state.run_requested = False
    st.session_state.last_run_status = None


def clear_results() -> None:
    st.session_state.results = None
    st.session_state.last_run_status = None


def request_run() -> None:
    """Trigger a run and lock the Run button until the computation finishes."""
    st.session_state.run_requested = True
    st.session_state.run_in_progress = True
    st.session_state.last_run_status = None


def set_doc_section(state_key: str, title: str) -> None:
    st.session_state[state_key] = title


def build_download_filename(audio_filename: str | None) -> str:
    """
    Build the PNG download filename as "audivisio-<stem>.png".

    The stem is taken from the stored audio filename (extension stripped),
    lowercased, non-alphanumeric characters replaced by hyphens, and
    truncated to 32 characters so the full filename stays readable.
    Falls back to "audivisio" when no filename is available.
    """
    if not audio_filename:
        return "audivisio.png"
    stem = Path(audio_filename).stem
    # Sanitize: keep alphanumerics and hyphens, collapse runs of non-alnum
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
    sanitized = sanitized[:32].rstrip("-") or "audio"
    return f"audivisio-{sanitized}.png"


# ============================================================
# Shared render helpers
# ============================================================

def render_image_output(label: str, image: np.ndarray | None, caption: str = "") -> None:
    st.markdown(f'<div class="section-pill">{label}</div>', unsafe_allow_html=True)
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
        # Animated loading bar — plays once each time this column re-renders
        # (on tab switch or section click). Pure CSS, no blocking sleep needed.
        st.markdown(
            """
            <style>
            @keyframes _doc_load {
                from { width: 0%; opacity: 1; }
                to   { width: 100%; opacity: 0; }
            }
            ._doc_progress_wrap {
                height: 3px;
                background: rgba(255,75,75,0.12);
                border-radius: 2px;
                overflow: hidden;
                margin-bottom: 1.1rem;
            }
            ._doc_progress_bar {
                height: 100%;
                background: #FF4B4B;
                border-radius: 2px;
                animation: _doc_load 0.55s cubic-bezier(0.4, 0, 0.2, 1) forwards;
            }
            </style>
            <div class="_doc_progress_wrap"><div class="_doc_progress_bar"></div></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(sections[st.session_state[state_key]])



def render_parameter_tabs(
    waveform_preview: np.ndarray | None,
    sr_preview: int | None,
) -> tuple[int, str, str, int, str, dict]:
    """
    Render all synthesis parameters as a five-tab panel.

    Returns
    -------
    target_size     : output image side length in pixels
    output_mode     : one of OUTPUT_MODE_OPTIONS
    section_layout  : one of SECTION_LAYOUT_OPTIONS
    n_sections      : effective section count (1 when layout is None)
    wavelet_type    : one of WAVELET_OPTIONS
    params          : advanced parameter dict consumed by the synthesis pipeline
    """
    params: dict = {}

    (
        tab_signal,
        tab_rendering,
        tab_color,
        tab_effects,
        tab_features,
    ) = st.tabs([
        "Signal",
        "Rendering",
        "Color",
        "Effects",
        "Features",
    ])

    # ── Tab 1 · Signal ────────────────────────────────────────────────────────
    with tab_signal:
        sig_left, sig_right = st.columns(2, gap="large")

        with sig_left:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Output image</div>', unsafe_allow_html=True)
                target_size = st.slider(
                    "Size (px)",
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
                    index=4,
                    key="output_mode",
                    on_change=clear_results,
                )

            with st.container(border=True):
                st.markdown('<div class="param-group-label">CWT wavelet</div>', unsafe_allow_html=True)
                wavelet_type = st.selectbox(
                    "Wavelet",
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

        with sig_right:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Sectioning</div>', unsafe_allow_html=True)
                section_layout = st.selectbox(
                    "Layout",
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
                        "Load an audio file to enable the section slider.",
                        icon="ℹ️",
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

                    slider_context = "_".join(
                        str(item).replace(" ", "_") for item in audio_context
                    )
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
                                "Sectioning is disabled when Layout is None. "
                                "The whole signal produces a single image."
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
                                "The signal is split into chronological sections. "
                                "Each section generates one block of the final image."
                            ),
                        )

                    st.markdown(
                        f'<p class="small-muted">'
                        f'Samples: {n_samples_preview} &nbsp;·&nbsp; '
                        f'SR: {sr_context} Hz &nbsp;·&nbsp; '
                        f'Max sections: {k_max}'
                        f'</p>',
                        unsafe_allow_html=True,
                    )

    # ── Tab 2 · Rendering ─────────────────────────────────────────────────────
    with tab_rendering:
        ren_left, ren_right = st.columns(2, gap="large")

        with ren_left:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Normalization</div>', unsafe_allow_html=True)
                params["robust_lower_percentile"] = st.slider(
                    "Lower percentile", 0.0, 10.0, 1.0, 0.5, on_change=clear_results,
                )
                params["robust_upper_percentile"] = st.slider(
                    "Upper percentile", 90.0, 100.0, 99.0, 0.5, on_change=clear_results,
                )

        with ren_right:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Tone</div>', unsafe_allow_html=True)
                params["gamma_correction"]  = st.slider("Gamma",      0.20, 2.50, 0.85, 0.05, on_change=clear_results)
                params["contrast_strength"] = st.slider("Contrast",   0.20, 3.00, 1.00, 0.05, on_change=clear_results)
                params["brightness_factor"] = st.slider("Brightness", 0.20, 2.50, 1.00, 0.05, on_change=clear_results)
                params["saturation_factor"] = st.slider("Saturation", 0.00, 3.00, 1.00, 0.05, on_change=clear_results)

    # ── Tab 3 · Color ─────────────────────────────────────────────────────────
    with tab_color:
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Frequency band splits</div>', unsafe_allow_html=True)
                params["rgb_low_end"]   = st.slider("Low → mid boundary",  0.10, 0.45, 1.0 / 3.0, 0.01, on_change=clear_results)
                params["rgb_high_start"] = st.slider("Mid → high boundary", 0.55, 0.90, 2.0 / 3.0, 0.01, on_change=clear_results)
                params["rgb_normalization_mode"] = st.selectbox(
                    "RGB normalization", ["Per-channel", "Shared"], index=0, on_change=clear_results,
                )

        with col_right:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Channel balance</div>', unsafe_allow_html=True)
                params["rgb_balance_r"] = st.slider("Red",   0.00, 3.00, 1.00, 0.05, on_change=clear_results)
                params["rgb_balance_g"] = st.slider("Green", 0.00, 3.00, 1.00, 0.05, on_change=clear_results)
                params["rgb_balance_b"] = st.slider("Blue",  0.00, 3.00, 1.00, 0.05, on_change=clear_results)

    # ── Tab 4 · Effects ───────────────────────────────────────────────────────
    with tab_effects:
        eff_left, eff_right = st.columns(2, gap="large")

        with eff_left:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Black mix</div>', unsafe_allow_html=True)
                params["ink_class_choice"] = st.selectbox(
                    "Otsu class",
                    ["Automatic minority", "Dark class", "Bright class"],
                    index=0, on_change=clear_results,
                )
                params["ink_keep_percentile"] = st.slider("Pixel density (%)", 1.0, 100.0, 50.0, 1.0, on_change=clear_results)
                params["ink_smoothing_sigma"] = st.slider("Pre-smoothing σ",   0.0,  10.0,  0.0, 0.25, on_change=clear_results)
                params["ink_thickness"]       = st.slider("Line thickness",      0,     8,    0,    1,  on_change=clear_results)

            with st.container(border=True):
                st.markdown('<div class="param-group-label">Luma mix</div>', unsafe_allow_html=True)
                params["luma_strength"]         = st.slider("Strength",            0.0,  1.0, 1.00, 0.05, on_change=clear_results)
                params["luma_min_coeff"]        = st.slider("Minimum coefficient", 0.0,  1.0, 0.00, 0.05, on_change=clear_results)
                params["luma_gamma"]            = st.slider("Coefficient gamma",  0.20, 3.00, 1.00, 0.05, on_change=clear_results)
                params["luma_coeff_blur_sigma"] = st.slider("Coefficient blur σ",  0.0, 12.0,  0.0, 0.25, on_change=clear_results)

        with eff_right:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Segmentation (Watershed mode)</div>', unsafe_allow_html=True)

                seg_method = st.selectbox(
                    "Method",
                    SEGMENTATION_METHODS,
                    index=3,
                    key="segmentation_method",
                    on_change=clear_results,
                    help=(
                        "Watershed: gradient-based flood from markers.\n"
                        "K-means: color clustering in RGB space (scipy).\n"
                        "SLIC: compact superpixels via color+spatial proximity (scikit-image).\n"
                        "Felzenszwalb: graph-based merge by internal variation (scikit-image).\n"
                        "Mean-shift: mode-seeking density estimation on downsampled image (scikit-learn)."
                    ),
                )
                params["segmentation_method"] = seg_method

                # ── Shared boundary controls (all methods) ──────────────────
                st.markdown('<div class="param-group-label" style="margin-top:0.6rem">Region color</div>', unsafe_allow_html=True)
                params["seg_region_color_mode"] = st.selectbox(
                    "Color mode",
                    ["Random pixel", "Mean color", "Median color"],
                    index=0,
                    on_change=clear_results,
                    help="Random pixel avoids grey convergence on noise-like images.",
                )
                params["seg_random_seed"] = st.number_input(
                    "Random seed", min_value=0, max_value=999999, value=12345, step=1, on_change=clear_results,
                )

                st.markdown('<div class="param-group-label" style="margin-top:0.6rem">Boundaries</div>', unsafe_allow_html=True)
                params["seg_boundary_style"]       = st.selectbox("Style",     ["None", "Black", "Local mean"], index=0, on_change=clear_results)
                params["seg_boundary_thickness"]   = st.slider("Thickness (px)",  0,  8, 0, 1, on_change=clear_results)
                params["seg_boundary_mean_window"] = st.slider("Mean window (px)", 3, 21, 5, 2, on_change=clear_results)

                # ── Method-specific parameters ───────────────────────────────
                if seg_method == "Watershed":
                    st.markdown('<div class="param-group-label" style="margin-top:0.6rem">Watershed</div>', unsafe_allow_html=True)
                    params["watershed_marker_spacing"]     = st.slider("Marker spacing (px)", 4, 160, 36,  1,   on_change=clear_results)
                    params["watershed_gradient_smoothing"] = st.slider("Gradient σ",         0.0, 8.0, 1.3, 0.1, on_change=clear_results)
                    # Mirror to shared keys so the helper picks them up
                    params["watershed_region_color_mode"]  = params["seg_region_color_mode"]
                    params["watershed_random_seed"]        = params["seg_random_seed"]
                    params["watershed_boundary_style"]     = params["seg_boundary_style"]
                    params["watershed_boundary_thickness"] = params["seg_boundary_thickness"]
                    params["watershed_boundary_mean_window"] = params["seg_boundary_mean_window"]

                elif seg_method == "K-means":
                    st.markdown('<div class="param-group-label" style="margin-top:0.6rem">K-means</div>', unsafe_allow_html=True)
                    params["kmeans_k"] = st.slider("Number of clusters k", 10, 400, 120, 5, on_change=clear_results)

                elif seg_method == "SLIC":
                    if not SKIMAGE_AVAILABLE:
                        st.warning("scikit-image not installed — will fall back to K-means.", icon="⚠️")
                    st.markdown('<div class="param-group-label" style="margin-top:0.6rem">SLIC</div>', unsafe_allow_html=True)
                    params["slic_n_segments"]   = st.slider("Target segments",  10, 400, 120,  5,   on_change=clear_results)
                    params["slic_compactness"]  = st.slider("Compactness",      0.1, 50.0, 10.0, 0.5, on_change=clear_results,
                                                            help="Higher = more square superpixels; lower = more color-following.")
                    params["slic_sigma"]        = st.slider("Pre-smoothing σ",  0.0,  5.0,  1.0, 0.1, on_change=clear_results)

                elif seg_method == "Felzenszwalb":
                    if not SKIMAGE_AVAILABLE:
                        st.warning("scikit-image not installed — will fall back to K-means.", icon="⚠️")
                    st.markdown('<div class="param-group-label" style="margin-top:0.6rem">Felzenszwalb</div>', unsafe_allow_html=True)
                    params["felz_scale"]    = st.slider("Scale (region size)",    1.0, 500.0, 100.0, 5.0, on_change=clear_results,
                                                        help="Larger = fewer, larger regions. Number of regions is automatic.")
                    params["felz_sigma"]    = st.slider("Pre-smoothing σ",        0.0,   3.0,   0.8, 0.1, on_change=clear_results)
                    params["felz_min_size"] = st.slider("Minimum region size (px)", 1,   200,    20,   1,  on_change=clear_results)

                elif seg_method == "Mean-shift":
                    if not SKLEARN_AVAILABLE:
                        st.warning("scikit-learn not installed — will fall back to K-means.", icon="⚠️")
                    st.markdown('<div class="param-group-label" style="margin-top:0.6rem">Mean-shift</div>', unsafe_allow_html=True)
                    params["meanshift_bandwidth"] = st.slider(
                        "Bandwidth (0 = auto)",  0.0, 100.0, 0.0, 1.0, on_change=clear_results,
                        help="Search window radius in RGB space. 0 = estimated automatically.",
                    )
                    params["meanshift_max_side"] = st.slider(
                        "Downsample max side (px)", 16, 128, 64, 8, on_change=clear_results,
                        help="Image is downsampled before clustering to keep computation tractable.",
                    )

    # ── Tab 5 · Features ──────────────────────────────────────────────────────
    with tab_features:
        feat_left, feat_right = st.columns(2, gap="large")

        with feat_left:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Magnitude weights (auto-normalized)</div>', unsafe_allow_html=True)
                w_col1, w_col2 = st.columns(2)
                with w_col1:
                    params["mag_weight_stft"]   = st.slider("STFT",   0.0, 1.0, W_STFT_TOTAL, 0.01, on_change=clear_results)
                    params["mag_weight_cwt"]    = st.slider("CWT",    0.0, 1.0, W_CWT_MAG,    0.01, on_change=clear_results)
                    params["mag_weight_mel"]    = st.slider("Mel",    0.0, 1.0, W_MEL,        0.01, on_change=clear_results)
                with w_col2:
                    params["mag_weight_chroma"] = st.slider("Chroma", 0.0, 1.0, W_CHROMA,     0.01, on_change=clear_results)
                    params["mag_weight_mfcc"]   = st.slider("MFCC",   0.0, 1.0, W_MFCC,       0.01, on_change=clear_results)
                    params["mag_weight_rms"]    = st.slider("RMS",    0.0, 1.0, W_RMS,        0.01, on_change=clear_results)

            with st.container(border=True):
                st.markdown('<div class="param-group-label">Phase weights (auto-normalized)</div>', unsafe_allow_html=True)
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    params["phase_weight_stft_mid"]  = st.slider("STFT 1024", 0.0, 1.0, W_PHASE_STFT_1024, 0.01, on_change=clear_results)
                    params["phase_weight_stft_fine"] = st.slider("STFT 512",  0.0, 1.0, W_PHASE_STFT_512,  0.01, on_change=clear_results)
                    params["phase_weight_cwt"]       = st.slider("CWT phase", 0.0, 1.0, W_PHASE_CWT,       0.01, on_change=clear_results)
                with p_col2:
                    params["phase_weight_onset"]     = st.slider("Onset",     0.0, 1.0, W_PHASE_ONSET,     0.01, on_change=clear_results)
                    params["phase_weight_centroid"]  = st.slider("Centroid",  0.0, 1.0, W_PHASE_CENTROID,  0.01, on_change=clear_results)
                    params["phase_weight_zcr"]       = st.slider("ZCR",       0.0, 1.0, W_PHASE_ZCR,       0.01, on_change=clear_results)

        with feat_right:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">STFT windows</div>', unsafe_allow_html=True)
                fft_options = [256, 512, 1024, 2048, 4096, 8192]
                params["stft_n_fft_min"] = st.selectbox("Minimum N_FFT", fft_options, index=0, on_change=clear_results)
                params["stft_n_fft_max"] = st.selectbox("Maximum N_FFT", fft_options, index=5, on_change=clear_results)

            with st.container(border=True):
                st.markdown('<div class="param-group-label">CWT analysis</div>', unsafe_allow_html=True)
                params["cwt_n_scales"]    = st.slider("Number of scales",  16, 128,    CWT_N_SCALES,    4,    on_change=clear_results)
                params["cwt_max_samples"] = st.slider("Maximum samples", 4096, 220500, CWT_MAX_SAMPLES, 4096, on_change=clear_results)

            with st.container(border=True):
                st.markdown('<div class="param-group-label">Mel / MFCC</div>', unsafe_allow_html=True)
                params["n_mels"] = st.slider("Mel bands",       32, 256, N_MELS, 8, on_change=clear_results)
                params["n_mfcc"] = st.slider("MFCC coefficients", 8,  64, N_MFCC, 1, on_change=clear_results)

    return target_size, output_mode, section_layout, n_sections, wavelet_type, params


# ============================================================
# App tab
# ============================================================

def render_app_tab() -> None:
    if not LIBROSA_AVAILABLE:
        st.error("librosa is not installed. Please add `librosa` to requirements.txt.")
        return

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    st.markdown(
        '<div class="app-header">'
        '<span class="app-title">Audio Visualization</span>'        
        '</div>'
        '<div class="app-subtitle">Spectral feature extraction · 2D Fourier synthesis · Inverse transform</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Initialise default audio bytes only when the default source is selected
    # --------------------------------------------------------
    if (
        st.session_state.audio_source == "Default sample"
        and st.session_state.audio_bytes is None
    ):
        def_waveform, def_sr, def_bytes = load_default_audio()
        if def_bytes is not None:
            st.session_state.audio_bytes = def_bytes
            st.session_state.using_default = True

    waveform_preview = None
    sr_preview = None
    if st.session_state.audio_bytes is not None:
        try:
            waveform_preview, sr_preview = load_audio(st.session_state.audio_bytes)
        except Exception:
            waveform_preview, sr_preview = None, None

    # ========================================================
    # Row 1: input box + run button  |  output image
    # ========================================================
    input_col, output_col = st.columns([1.0, 1.35], gap="large")

    with input_col:
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
                        st.session_state.audio_filename = DEFAULT_AUDIO_TITLE
                        st.session_state.last_audio_context = None
                        st.session_state.results = None
                        st.rerun()
                    st.caption(f"**{DEFAULT_AUDIO_TITLE}** — {DEFAULT_AUDIO_DESCRIPTION}")
                    st.audio(def_bytes)
                    if waveform_preview is not None:
                        st.markdown('<div class="section-pill">Waveform</div>', unsafe_allow_html=True)
                        st.image(waveform_to_display_image(waveform_preview), width="stretch")
                        st.markdown('<div class="section-pill">Spectrogram</div>', unsafe_allow_html=True)
                        st.image(spectrogram_to_display_image(waveform_preview), width="stretch")
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
                        st.session_state.audio_filename = uploaded_file.name
                        st.session_state.last_audio_context = None
                        st.session_state.results = None
                        st.rerun()
                    st.audio(uploaded_bytes)
                    if waveform_preview is not None:
                        st.markdown('<div class="section-pill">Waveform</div>', unsafe_allow_html=True)
                        st.image(waveform_to_display_image(waveform_preview), width="stretch")
                        st.markdown('<div class="section-pill">Spectrogram</div>', unsafe_allow_html=True)
                        st.image(spectrogram_to_display_image(waveform_preview), width="stretch")
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
                            st.session_state.audio_filename = "recording.wav"
                            st.session_state.last_audio_context = None
                            st.session_state.results = None
                            st.rerun()
                        st.audio(recorded_bytes)
                        if waveform_preview is not None:
                            st.markdown('<div class="section-pill">Waveform</div>', unsafe_allow_html=True)
                            st.image(waveform_to_display_image(waveform_preview), width="stretch")
                            st.markdown('<div class="section-pill">Spectrogram</div>', unsafe_allow_html=True)
                            st.image(spectrogram_to_display_image(waveform_preview), width="stretch")
                    else:
                        st.info("Record an audio signal with your microphone.")
                        st.session_state.audio_bytes = None
                        st.session_state.using_default = False
                except AttributeError:
                    st.warning("Audio recording requires a newer Streamlit version.")
                    st.session_state.audio_bytes = None
                    st.session_state.using_default = False

        st.button(
            "▶  GENERATE IMAGE",
            type="primary",
            width="stretch",
            disabled=(st.session_state.audio_bytes is None or st.session_state.run_in_progress),
            on_click=request_run,
        )

    with output_col:
        progress_status_placeholder = st.empty()
        progress_bar_placeholder = st.empty()

        if st.session_state.last_run_status == "Done":
            progress_status_placeholder.success("Done — 100%")

        results = st.session_state.results

        if results is None:
            with st.container(border=True):
                st.markdown("#### Output image")
                st.info("Generated image will appear here after you click **▶ Generate image**.")
        else:
            # ---- Main output image ----
            with st.container(border=True):
                st.markdown("#### Output image")
                render_image_output("Generated image", results["generated_image"])

                # Metadata row
                dur = results["duration"]
                meta_html = (
                    '<div class="result-meta">'
                    f'<div class="result-meta-item">Duration <span>{dur:.2f} s</span></div>'
                    f'<div class="result-meta-item">SR <span>{results["sr"]} Hz</span></div>'
                    f'<div class="result-meta-item">Samples <span>{results["n_samples"]}</span></div>'
                    f'<div class="result-meta-item">Sections <span>{results["n_sections"]}</span></div>'
                    f'<div class="result-meta-item">Layout <span>{results["section_layout"]}</span></div>'
                    f'<div class="result-meta-item">Mode <span>{results["output_mode"]}</span></div>'
                    '</div>'
                )
                st.markdown(meta_html, unsafe_allow_html=True)
                st.markdown("")  # spacer

                st.download_button(
                    "⬇  Download PNG",
                    data=results["png_bytes"],
                    file_name=build_download_filename(st.session_state.get("audio_filename")),
                    mime="image/png",
                    width="stretch",
                )

            # ---- 2D Fourier representation of output image ----
            with st.container(border=True):
                st.markdown("#### 2D Fourier representation")
                st.caption(
                    "Centered log-magnitude (viridis) and phase (twilight) of the "
                    "2D DFT of the output image's luminance channel."
                )
                fourier_col1, fourier_col2 = st.columns(2, gap="small")
                mag_img, phase_img = output_image_fourier_to_display_images(results["generated_image"])
                with fourier_col1:
                    render_image_output("Magnitude  log|F(u,v)|", mag_img)
                with fourier_col2:
                    render_image_output("Phase  ∠F(u,v)", phase_img)

    # ========================================================
    # Row 2: parameters box
    # ========================================================
    with st.expander("⚙  Parameters", expanded=False):
        (
            target_size,
            output_mode,
            section_layout,
            n_sections,
            wavelet_type,
            synthesis_params,
        ) = render_parameter_tabs(waveform_preview, sr_preview)

    if "synthesis_params" not in locals():
        synthesis_params = {}
        target_size = IMAGE_SIZE_DEFAULT
        output_mode = OUTPUT_MODE_OPTIONS[4]
        section_layout = SECTION_LAYOUT_OPTIONS[0]
        n_sections = 1
        wavelet_type = WAVELET_OPTIONS[0]

    # ========================================================
    # Run computation
    # ========================================================
    if st.session_state.run_requested and st.session_state.audio_bytes is not None:
        progress_status_placeholder.info("Preparing computation — 0%")
        progress_bar = progress_bar_placeholder.progress(0)

        with st.spinner("Loading audio…"):
            waveform, sr = load_audio(st.session_state.audio_bytes)

        if waveform is None or sr is None:
            st.session_state.run_in_progress = False
            st.session_state.run_requested = False
            st.session_state.last_run_status = None
            progress_bar_placeholder.empty()
            progress_status_placeholder.error("Could not decode the audio file.")
            st.error("Could not decode the audio file.")
        else:
            k_max_runtime = compute_max_sections(len(waveform), target_size)
            n_sections_runtime = 1 if section_layout == "None" else min(max(1, int(n_sections)), k_max_runtime)

            def update_progress(label: str, done: int, total: int) -> None:
                percent = int(round(100.0 * done / max(1, total)))
                progress_status_placeholder.info(f"{label} — {percent}%")
                progress_bar.progress(percent)

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
                    params=synthesis_params,
                )

            progress_bar.progress(100)
            progress_status_placeholder.success("Done — 100%")

            st.session_state.results = {
                "generated_image": image_rgb,
                "png_bytes": image_to_png_bytes(image_rgb),
                "duration": len(waveform) / sr,
                "sr": sr,
                "n_samples": len(waveform),
                "n_sections": n_sections_runtime,
                "section_layout": section_layout,
                "output_mode": output_mode,
            }

            st.session_state.run_in_progress = False
            st.session_state.run_requested = False
            st.session_state.last_run_status = "Done"

            # The Run button was rendered earlier in this script pass while
            # run_in_progress was still True. Force one final rerun so the
            # button is immediately rendered as interactive again after the
            # computation has finished.
            st.rerun()



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
