from __future__ import annotations

# ============================================================
# Audio analysis constants
# ============================================================

# MAX_RECORD_SECONDS = 60: caps in-browser recording. Beyond this, feature
# extraction exceeds user-interaction latency budgets on a single CPU core.
MAX_RECORD_SECONDS: int = 60

# STFT_N_FFT_MIN = 256: smallest DFT window; frequency resolution ≈ 86 Hz/bin
# at 22 050 Hz. Smaller windows make individual semitones unresolvable.
STFT_N_FFT_MIN: int = 256

# STFT_N_FFT_MAX = 8 192: largest DFT window; frequency resolution ≈ 2.7 Hz/bin.
# Larger windows require ≥ 8 192 samples (~0.37 s) for a single frame.
STFT_N_FFT_MAX: int = 8_192

# CWT_N_SCALES = 64: 64 log-spaced scales spanning 9 octaves (~0.14 oct/step),
# matching the frequency resolution of the mel filterbank.
CWT_N_SCALES: int = 64

# CWT_MORLET_W = 6.0: Morlet central frequency parameter. For w ≥ 5, the DC
# leak is exp(−w²/2) ≤ 3.7×10⁻⁶, making the wavelet analytically valid for
# instantaneous phase computation. w = 6 is the conventional choice.
CWT_MORLET_W: float = 6.0

# CWT_MAX_SAMPLES = 44 100: waveform is downsampled to this count before CWT.
# At 22 050 Hz this is 2 s, sufficient to capture the broad spectro-temporal
# envelope while bounding cost (O(N·S·max_scale)).
CWT_MAX_SAMPLES: int = 44_100

# N_MELS = 128: matches the number of auditory critical bands from 0 Hz to
# sr/2 Hz. The mel scale is logarithmic above ~1 000 Hz, reflecting reduced
# cochlear pitch discrimination at high frequencies.
N_MELS: int = 128

# N_MFCC = 20: coefficients 1–13 capture timbre; 14–20 add fine texture.
# Beyond 20, marginal information content decays rapidly.
N_MFCC: int = 20

# ============================================================
# Image geometry
# ============================================================

IMAGE_SIZE_MIN:     int = 64
IMAGE_SIZE_MAX:     int = 1024
IMAGE_SIZE_DEFAULT: int = 512
# IMAGE_SIZE_STEP = 16: keeps all sizes divisible by 16, aligning with
# GPU/SIMD tile widths and ensuring symmetric Hermitian indices.
IMAGE_SIZE_STEP: int = 16

# ============================================================
# Magnitude grid blend weights — must sum to 1.0
# ============================================================
# STFT   (0.45): primary representation; multiple resolutions.
# CWT    (0.15): multi-scale temporal structure orthogonal to STFT.
# Mel    (0.18): perceptually weighted frequency axis.
# Chroma (0.09): pitch-class content, octave-invariant.
# MFCC   (0.09): coarse spectral envelope.
# RMS    (0.04): loudness envelope as spatial amplitude modulation.

W_STFT_TOTAL: float = 0.45
W_CWT_MAG:    float = 0.15
W_MEL:        float = 0.18
W_CHROMA:     float = 0.09
W_MFCC:       float = 0.09
W_RMS:        float = 0.04
# Sum = 0.45 + 0.15 + 0.18 + 0.09 + 0.09 + 0.04 = 1.00

# ============================================================
# Phase grid blend weights — must sum to 1.0 (Morlet)
# ============================================================
# STFT 1024 (0.30): best time–frequency balance; primary source.
# STFT  512 (0.20): higher time resolution; captures fast transients.
# CWT Morlet(0.20): instantaneous phase at multiple scales.
# Onset     (0.15): phase jumps at rhythmic events.
# Centroid  (0.10): pitch brightness variation encodes melodic contour.
# ZCR       (0.05): noisiness encodes timbre texture.
# For Ricker, the CWT weight 0.20 is redistributed to the STFT sources.

W_PHASE_STFT_1024: float = 0.30
W_PHASE_STFT_512:  float = 0.20
W_PHASE_CWT:       float = 0.20
W_PHASE_ONSET:     float = 0.15
W_PHASE_CENTROID:  float = 0.10
W_PHASE_ZCR:       float = 0.05
# Morlet total: 0.30+0.20+0.20+0.15+0.10+0.05 = 1.00
# Ricker total (CWT weight redistributed to STFT): 0.40+0.30+0+0.15+0.10+0.05 = 1.00

# ============================================================
# UI option lists
# ============================================================

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

# ============================================================
# Sectioned image synthesis
# ============================================================

MIN_SECTION_SAMPLES: int = 16_384
MIN_BLOCK_SIDE:      int = 32
DEFAULT_SECTIONS:    int = 32
MAX_SECTIONS_UI:     int = 64

AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]

# ============================================================
# Default audio sample (CC0, Wikimedia Commons)
# The file is 14 s long and released as CC0, long enough for the default
# section count to reach 32 with the current MIN_SECTION_SAMPLES rule.
# ============================================================

DEFAULT_AUDIO_TITLE       = "Phonk sample.ogg"
DEFAULT_AUDIO_DESCRIPTION = "8-bar drift phonk instrumental at 140 BPM, CC0, Wikimedia Commons"
DEFAULT_AUDIO_URL         = "https://upload.wikimedia.org/wikipedia/commons/2/2c/Phonk_sample.ogg"

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
        "url":      "https://e.pcloud.link/publink/show?code=XZX81iZss7g3iD9fGJXmPRRGSi7LBTvLcgX",
        "icon_url": "https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg",
    },
    {
        "platform": "CV EN",
        "label":    "CV EN",
        "url":      "https://e.pcloud.link/publink/show?code=XZ581iZBQvbu1mFKjziunF9lblghze8OXkk",
        "icon_url": "https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg",
    },
]
