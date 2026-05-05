from __future__ import annotations

import io
import urllib.request

import numpy as np
import streamlit as st

from config import DEFAULT_AUDIO_URL, MAX_RECORD_SECONDS

try:
    import librosa
    import librosa.feature
    import librosa.onset
    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    LIBROSA_AVAILABLE = False


@st.cache_data(show_spinner=False)
def load_default_audio() -> tuple[np.ndarray | None, int | None, bytes | None]:
    """
    Load the default open-source music sample from Wikimedia Commons.

    The default file is downloaded and decoded at its original sample rate.
    It is intentionally longer than a short test clip so that the default
    section count can reach 32 when the image size is large enough.

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
    waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    max_samples = int(MAX_RECORD_SECONDS * sr)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    return waveform, sr
