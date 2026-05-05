from __future__ import annotations

import html
import re
import warnings
from pathlib import Path

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

from config import (
    AUDIO_TYPES,
    CWT_MAX_SAMPLES,
    CWT_N_SCALES,
    DEFAULT_AUDIO_DESCRIPTION,
    DEFAULT_AUDIO_TITLE,
    DEFAULT_SECTIONS,
    IMAGE_SIZE_DEFAULT,
    IMAGE_SIZE_MAX,
    IMAGE_SIZE_MIN,
    IMAGE_SIZE_STEP,
    N_MELS,
    N_MFCC,
    OUTPUT_MODE_OPTIONS,
    PORTFOLIO_LINKS,
    SECTION_LAYOUT_OPTIONS,
    SEGMENTATION_METHODS,
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
    WAVELET_OPTIONS,
)
from audio_io import LIBROSA_AVAILABLE, load_audio, load_default_audio
from display import (
    image_to_png_bytes,
    output_image_fourier_to_display_images,
    spectrogram_to_display_image,
    waveform_to_display_image,
)
from segmentation import SKIMAGE_AVAILABLE, SKLEARN_AVAILABLE
from synthesis import compute_max_sections, generate_sectioned_image


# ============================================================
# Documentation loading
# ============================================================

def _read_markdown_file(path: str) -> str:
    """Read a local Markdown file; return a placeholder if missing."""
    fp = Path(path)
    if not fp.exists():
        return f"## Documentation unavailable\n\nThe file `{path}` was not found in the app directory."
    return fp.read_text(encoding="utf-8")


def _split_markdown_by_h2(markdown_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for part in re.split(r"(?m)^##\s+", markdown_text.strip()):
        part = part.strip()
        if not part:
            continue
        title = part.splitlines()[0].strip()
        if title.lower() in {"table des matières", "table of contents"}:
            continue
        sections[title] = "## " + part
    if not sections:
        sections["Documentation"] = markdown_text
    return sections


DOC_FR_SECTIONS = _split_markdown_by_h2(_read_markdown_file("documentation_fr.md"))
DOC_EN_SECTIONS = _split_markdown_by_h2(_read_markdown_file("documentation_en.md"))
DOC_FR_TITLES   = list(DOC_FR_SECTIONS.keys())
DOC_EN_TITLES   = list(DOC_EN_SECTIONS.keys())


# ============================================================
# Portfolio links
# ============================================================

def render_portfolio_links() -> None:
    parts = []
    for item in PORTFOLIO_LINKS:
        show_label = item["platform"] in {"CV FR", "CV EN"}
        link_class = "portfolio-link with-label" if show_label else "portfolio-link icon-only"
        title      = (
            f"Open {item['platform']}: {item['label']}" if not show_label
            else f"Open {item['platform']}"
        )
        label_html = (
            f'<span class="portfolio-label">{html.escape(item["label"])}</span>'
            if show_label else ""
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
            padding-top: 2.75rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-bottom: 2.5rem;
        }
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
        h2 {
            text-align: center;
            border: 1px solid rgba(255, 75, 75, 0.18);
            border-radius: 0.35rem;
            padding: 0.55rem 0.75rem;
            margin-top: 0.25rem;
            margin-bottom: 1.00rem;
            background: rgba(255, 75, 75, 0.04);
        }
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
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 0.45rem !important;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.04em;
        }
        .small-muted { color: #9ca3af; font-size: 0.80rem; line-height: 1.5; }
        .result-meta { display: flex; flex-wrap: wrap; gap: 0.5rem 1.2rem; margin-top: 0.6rem; }
        .result-meta-item { font-size: 0.78rem; color: #9ca3af; }
        .result-meta-item span { color: #FF4B4B; font-weight: 600; }
        .section-pill {
            display: inline-block;
            font-size: 0.70rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
            background: rgba(255, 75, 75, 0.10); border: 1px solid rgba(255, 75, 75, 0.22);
            border-radius: 0.25rem; padding: 0.1rem 0.45rem; color: #FF4B4B; margin-bottom: 0.4rem;
        }
        .param-group-label {
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
            color: #9ca3af; margin-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 75, 75, 0.12); padding-bottom: 0.3rem;
        }
        .portfolio-link-row {
            display: flex; justify-content: flex-end; align-items: center;
            gap: 0.42rem; min-height: 2.35rem; margin: 0 0 -2.65rem 0;
            padding-right: 0.15rem; position: relative; z-index: 20;
        }
        .portfolio-link, .portfolio-link:visited {
            display: inline-flex !important; align-items: center !important; justify-content: center !important;
            height: 2rem !important; border: 1px solid rgba(250, 250, 250, 0.18) !important;
            border-radius: 0.35rem !important; color: inherit !important; text-decoration: none !important;
            font-size: 0.78rem !important; font-weight: 600 !important; line-height: 1 !important;
            background: rgba(255, 255, 255, 0.025) !important; white-space: nowrap !important;
            box-sizing: border-box !important; overflow: hidden !important; transition: all 0.15s ease !important;
        }
        .portfolio-link:hover {
            border-color: #FF4B4B !important; color: #FF4B4B !important;
            background: rgba(255, 75, 75, 0.08) !important; text-decoration: none !important;
            transform: translateY(-1px) !important;
        }
        .portfolio-link.icon-only, .portfolio-link.icon-only:visited {
            width: 2rem !important; min-width: 2rem !important; max-width: 2rem !important;
            padding: 0 !important; gap: 0 !important;
        }
        .portfolio-link.with-label, .portfolio-link.with-label:visited {
            width: auto !important; padding: 0 0.58rem !important; gap: 0.38rem !important;
        }
        .portfolio-icon {
            display: block !important; width: 1.10rem !important; height: 1.10rem !important;
            min-width: 1.10rem !important; max-width: 1.10rem !important;
            object-fit: contain !important; flex: 0 0 auto !important;
            margin: 0 !important; padding: 0 !important; border: 0 !important;
        }
        .portfolio-label { display: inline-block !important; }
        .portfolio-link.icon-only .portfolio-label {
            display: none !important; width: 0 !important; min-width: 0 !important;
            max-width: 0 !important; margin: 0 !important; padding: 0 !important; overflow: hidden !important;
        }
        @media (max-width: 1180px) {
            .portfolio-link-row { justify-content: flex-start; flex-wrap: wrap; margin-bottom: 0.65rem; padding-right: 0; }
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
        "audio_bytes":        None,
        "using_default":      False,
        "audio_source":       "Default sample",
        "audio_filename":     None,
        "last_audio_context": None,
        "results":            None,
        "run_in_progress":    False,
        "run_requested":      False,
        "last_run_status":    None,
        "doc_fr_title":       DOC_FR_TITLES[0],
        "doc_en_title":       DOC_EN_TITLES[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_audio() -> None:
    """Invalidate stored audio and all downstream results."""
    st.session_state.audio_bytes        = None
    st.session_state.using_default      = False
    st.session_state.audio_filename     = None
    st.session_state.last_audio_context = None
    st.session_state.results            = None
    st.session_state.run_in_progress    = False
    st.session_state.run_requested      = False
    st.session_state.last_run_status    = None


def clear_results() -> None:
    st.session_state.results         = None
    st.session_state.last_run_status = None


def request_run() -> None:
    """Trigger a run and lock the Run button until the computation finishes."""
    st.session_state.run_requested   = True
    st.session_state.run_in_progress = True
    st.session_state.last_run_status = None


# ============================================================
# Shared render helpers
# ============================================================

def set_doc_section(state_key: str, title: str) -> None:
    st.session_state[state_key] = title


def build_download_filename(audio_filename: str | None) -> str:
    """
    Build the PNG download filename as "audivisio-<stem>.png".

    The stem is taken from the stored audio filename, lowercased, with
    non-alphanumeric characters replaced by hyphens, truncated to 32 characters.
    """
    if not audio_filename:
        return "audivisio.png"
    stem      = Path(audio_filename).stem
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
    sanitized = sanitized[:32].rstrip("-") or "audio"
    return f"audivisio-{sanitized}.png"


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
            st.button(
                title,
                key=f"{state_key}_{title}",
                type="primary" if st.session_state[state_key] == title else "secondary",
                width="stretch",
                on_click=set_doc_section,
                args=(state_key, title),
            )
    with right_col:
        st.markdown(
            """
            <style>
            @keyframes _doc_load { from { width: 0%; opacity: 1; } to { width: 100%; opacity: 0; } }
            ._doc_progress_wrap { height: 3px; background: rgba(255,75,75,0.12); border-radius: 2px; overflow: hidden; margin-bottom: 1.1rem; }
            ._doc_progress_bar  { height: 100%; background: #FF4B4B; border-radius: 2px; animation: _doc_load 0.55s cubic-bezier(0.4,0,0.2,1) forwards; }
            </style>
            <div class="_doc_progress_wrap"><div class="_doc_progress_bar"></div></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(sections[st.session_state[state_key]])


# ============================================================
# Parameter tabs
# ============================================================

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

    tab_signal, tab_rendering, tab_color, tab_effects, tab_features = st.tabs([
        "Signal", "Rendering", "Color", "Effects", "Features",
    ])

    # ── Tab 1 · Signal ────────────────────────────────────────────────────────
    with tab_signal:
        sig_left, sig_right = st.columns(2, gap="large")

        with sig_left:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Output image</div>', unsafe_allow_html=True)
                target_size = st.slider(
                    "Size (px)", IMAGE_SIZE_MIN, IMAGE_SIZE_MAX, IMAGE_SIZE_DEFAULT, IMAGE_SIZE_STEP,
                    key="target_size", on_change=clear_results,
                )
                output_mode = st.radio(
                    "Output mode", options=OUTPUT_MODE_OPTIONS, index=4,
                    key="output_mode", on_change=clear_results,
                )

            with st.container(border=True):
                st.markdown('<div class="param-group-label">CWT wavelet</div>', unsafe_allow_html=True)
                wavelet_type = st.selectbox(
                    "Wavelet", options=WAVELET_OPTIONS, index=0,
                    key="wavelet_type", on_change=clear_results,
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
                    "Layout", options=SECTION_LAYOUT_OPTIONS, index=0,
                    key="section_layout", on_change=clear_results,
                    help="Choose how chronological audio sections are combined into the final square image.",
                )

                if waveform_preview is None or sr_preview is None:
                    n_sections = 1
                    st.info("Load an audio file to enable the section slider.", icon="ℹ️")
                else:
                    n_samples_preview = len(waveform_preview)
                    sr_context        = int(sr_preview)
                    k_max             = compute_max_sections(n_samples_preview, target_size)
                    audio_context     = (
                        st.session_state.audio_source,
                        len(st.session_state.audio_bytes or b""),
                        int(n_samples_preview), sr_context, int(target_size), int(k_max),
                    )
                    slider_context      = "_".join(str(i).replace(" ", "_") for i in audio_context)
                    n_sections_key      = f"n_sections_{slider_context}"
                    n_sections_default  = min(DEFAULT_SECTIONS, k_max)
                    st.session_state.last_audio_context = audio_context

                    if section_layout == "None":
                        n_sections = 1
                        st.slider(
                            "Number of sections", 1, max(1, k_max), 1, 1,
                            key=f"n_sections_disabled_{slider_context}", disabled=True,
                            help="Sectioning is disabled when Layout is None.",
                        )
                    else:
                        n_sections = st.slider(
                            "Number of sections", 1, k_max, n_sections_default, 1,
                            key=n_sections_key, on_change=clear_results,
                            help="The signal is split into chronological sections. Each section generates one block.",
                        )

                    st.markdown(
                        f'<p class="small-muted">Samples: {n_samples_preview} &nbsp;·&nbsp; '
                        f'SR: {sr_context} Hz &nbsp;·&nbsp; Max sections: {k_max}</p>',
                        unsafe_allow_html=True,
                    )

    # ── Tab 2 · Rendering ─────────────────────────────────────────────────────
    with tab_rendering:
        ren_left, ren_right = st.columns(2, gap="large")
        with ren_left:
            with st.container(border=True):
                st.markdown('<div class="param-group-label">Normalization</div>', unsafe_allow_html=True)
                params["robust_lower_percentile"] = st.slider("Lower percentile", 0.0, 10.0,  1.0, 0.5, on_change=clear_results)
                params["robust_upper_percentile"] = st.slider("Upper percentile", 90.0, 100.0, 99.0, 0.5, on_change=clear_results)
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
                params["rgb_low_end"]            = st.slider("Low → mid boundary",  0.10, 0.45, 1.0 / 3.0, 0.01, on_change=clear_results)
                params["rgb_high_start"]         = st.slider("Mid → high boundary", 0.55, 0.90, 2.0 / 3.0, 0.01, on_change=clear_results)
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
                params["ink_class_choice"]    = st.selectbox("Otsu class", ["Automatic minority", "Dark class", "Bright class"], index=0, on_change=clear_results)
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
                    "Method", SEGMENTATION_METHODS, index=3,
                    key="segmentation_method", on_change=clear_results,
                    help=(
                        "Watershed: gradient-based flood from markers.\n"
                        "K-means: color clustering in RGB space (scipy).\n"
                        "SLIC: compact superpixels via color+spatial proximity (scikit-image).\n"
                        "Felzenszwalb: graph-based merge by internal variation (scikit-image).\n"
                        "Mean-shift: mode-seeking density estimation on downsampled image (scikit-learn)."
                    ),
                )
                params["segmentation_method"] = seg_method

                st.markdown('<div class="param-group-label" style="margin-top:0.6rem">Region color</div>', unsafe_allow_html=True)
                params["seg_region_color_mode"] = st.selectbox(
                    "Color mode", ["Random pixel", "Mean color", "Median color"], index=0,
                    on_change=clear_results, help="Random pixel avoids grey convergence on noise-like images.",
                )
                params["seg_random_seed"] = st.number_input("Random seed", min_value=0, max_value=999999, value=12345, step=1, on_change=clear_results)

                st.markdown('<div class="param-group-label" style="margin-top:0.6rem">Boundaries</div>', unsafe_allow_html=True)
                params["seg_boundary_style"]       = st.selectbox("Style",     ["None", "Black", "Local mean"], index=0, on_change=clear_results)
                params["seg_boundary_thickness"]   = st.slider("Thickness (px)",  0,  8, 0, 1, on_change=clear_results)
                params["seg_boundary_mean_window"] = st.slider("Mean window (px)", 3, 21, 5, 2, on_change=clear_results)

                if seg_method == "Watershed":
                    st.markdown('<div class="param-group-label" style="margin-top:0.6rem">Watershed</div>', unsafe_allow_html=True)
                    params["watershed_marker_spacing"]       = st.slider("Marker spacing (px)", 4, 160, 36,  1,   on_change=clear_results)
                    params["watershed_gradient_smoothing"]   = st.slider("Gradient σ",         0.0, 8.0, 1.3, 0.1, on_change=clear_results)
                    params["watershed_region_color_mode"]    = params["seg_region_color_mode"]
                    params["watershed_random_seed"]          = params["seg_random_seed"]
                    params["watershed_boundary_style"]       = params["seg_boundary_style"]
                    params["watershed_boundary_thickness"]   = params["seg_boundary_thickness"]
                    params["watershed_boundary_mean_window"] = params["seg_boundary_mean_window"]

                elif seg_method == "K-means":
                    st.markdown('<div class="param-group-label" style="margin-top:0.6rem">K-means</div>', unsafe_allow_html=True)
                    params["kmeans_k"] = st.slider("Number of clusters k", 10, 400, 120, 5, on_change=clear_results)

                elif seg_method == "SLIC":
                    if not SKIMAGE_AVAILABLE:
                        st.warning("scikit-image not installed — will fall back to K-means.", icon="⚠️")
                    st.markdown('<div class="param-group-label" style="margin-top:0.6rem">SLIC</div>', unsafe_allow_html=True)
                    params["slic_n_segments"]  = st.slider("Target segments",  10, 400, 120,  5,   on_change=clear_results)
                    params["slic_compactness"] = st.slider("Compactness",      0.1, 50.0, 10.0, 0.5, on_change=clear_results, help="Higher = more square superpixels.")
                    params["slic_sigma"]       = st.slider("Pre-smoothing σ",  0.0,  5.0,  1.0, 0.1, on_change=clear_results)

                elif seg_method == "Felzenszwalb":
                    if not SKIMAGE_AVAILABLE:
                        st.warning("scikit-image not installed — will fall back to K-means.", icon="⚠️")
                    st.markdown('<div class="param-group-label" style="margin-top:0.6rem">Felzenszwalb</div>', unsafe_allow_html=True)
                    params["felz_scale"]    = st.slider("Scale (region size)",    1.0, 500.0, 100.0, 5.0, on_change=clear_results, help="Larger = fewer, larger regions.")
                    params["felz_sigma"]    = st.slider("Pre-smoothing σ",        0.0,   3.0,   0.8, 0.1, on_change=clear_results)
                    params["felz_min_size"] = st.slider("Minimum region size (px)", 1,   200,    20,   1,  on_change=clear_results)

                elif seg_method == "Mean-shift":
                    if not SKLEARN_AVAILABLE:
                        st.warning("scikit-learn not installed — will fall back to K-means.", icon="⚠️")
                    st.markdown('<div class="param-group-label" style="margin-top:0.6rem">Mean-shift</div>', unsafe_allow_html=True)
                    params["meanshift_bandwidth"] = st.slider("Bandwidth (0 = auto)",  0.0, 100.0, 0.0, 1.0, on_change=clear_results, help="Search window radius in RGB space.")
                    params["meanshift_max_side"]  = st.slider("Downsample max side (px)", 16, 128, 64, 8, on_change=clear_results, help="Image is downsampled before clustering.")

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
                    params["phase_weight_onset"]    = st.slider("Onset",    0.0, 1.0, W_PHASE_ONSET,    0.01, on_change=clear_results)
                    params["phase_weight_centroid"] = st.slider("Centroid", 0.0, 1.0, W_PHASE_CENTROID, 0.01, on_change=clear_results)
                    params["phase_weight_zcr"]      = st.slider("ZCR",      0.0, 1.0, W_PHASE_ZCR,      0.01, on_change=clear_results)

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
                params["n_mels"] = st.slider("Mel bands",          32, 256, N_MELS, 8, on_change=clear_results)
                params["n_mfcc"] = st.slider("MFCC coefficients",   8,  64, N_MFCC, 1, on_change=clear_results)

    return target_size, output_mode, section_layout, n_sections, wavelet_type, params


# ============================================================
# App tab
# ============================================================

def render_app_tab() -> None:
    if not LIBROSA_AVAILABLE:
        st.error("librosa is not installed. Please add `librosa` to requirements.txt.")
        return

    st.markdown(
        '<div class="app-header"><span class="app-title">Audio Visualization</span></div>'
        '<div class="app-subtitle">Spectral feature extraction · 2D Fourier synthesis · Inverse transform</div>',
        unsafe_allow_html=True,
    )

    # Initialise default audio bytes when the default source is selected
    if st.session_state.audio_source == "Default sample" and st.session_state.audio_bytes is None:
        _, _, def_bytes = load_default_audio()
        if def_bytes is not None:
            st.session_state.audio_bytes = def_bytes
            st.session_state.using_default = True

    waveform_preview, sr_preview = None, None
    if st.session_state.audio_bytes is not None:
        try:
            waveform_preview, sr_preview = load_audio(st.session_state.audio_bytes)
        except Exception:
            pass

    # ── Row 1: input | output ────────────────────────────────────────────────
    input_col, output_col = st.columns([1.0, 1.35], gap="large")

    with input_col:
        with st.container(border=True):
            st.markdown("#### Input audio signal")

            source_choice = st.radio(
                "Audio source", options=["Default sample", "Upload file", "Record audio"],
                horizontal=True, key="audio_source", on_change=clear_audio,
            )

            if source_choice == "Default sample":
                _, _, def_bytes = load_default_audio()
                if def_bytes is not None:
                    if st.session_state.audio_bytes != def_bytes:
                        st.session_state.audio_bytes    = def_bytes
                        st.session_state.using_default  = True
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
                    st.session_state.audio_bytes   = None
                    st.session_state.using_default = False

            elif source_choice == "Upload file":
                uploaded_file = st.file_uploader("Upload an audio file", type=AUDIO_TYPES, key="audio_upload")
                if uploaded_file is not None:
                    uploaded_bytes = uploaded_file.getvalue()
                    if st.session_state.audio_bytes != uploaded_bytes:
                        st.session_state.audio_bytes    = uploaded_bytes
                        st.session_state.using_default  = False
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
                    st.session_state.audio_bytes   = None
                    st.session_state.using_default = False

            else:  # Record audio
                try:
                    recorded = st.audio_input("Record audio", key="audio_record")
                    if recorded is not None:
                        recorded_bytes = recorded.getvalue()
                        if st.session_state.audio_bytes != recorded_bytes:
                            st.session_state.audio_bytes    = recorded_bytes
                            st.session_state.using_default  = False
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
                        st.session_state.audio_bytes   = None
                        st.session_state.using_default = False
                except AttributeError:
                    st.warning("Audio recording requires a newer Streamlit version.")
                    st.session_state.audio_bytes   = None
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
        progress_bar_placeholder    = st.empty()

        if st.session_state.last_run_status == "Done":
            progress_status_placeholder.success("Done — 100%")

        results = st.session_state.results

        if results is None:
            with st.container(border=True):
                st.markdown("#### Output image")
                st.info("Generated image will appear here after you click **▶ Generate image**.")
        else:
            with st.container(border=True):
                st.markdown("#### Output image")
                render_image_output("Generated image", results["generated_image"])
                dur      = results["duration"]
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
                st.markdown("")
                st.download_button(
                    "⬇  Download PNG",
                    data=results["png_bytes"],
                    file_name=build_download_filename(st.session_state.get("audio_filename")),
                    mime="image/png",
                    width="stretch",
                )

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

    # ── Row 2: parameters ────────────────────────────────────────────────────
    with st.expander("⚙  Parameters", expanded=False):
        (
            target_size, output_mode, section_layout, n_sections,
            wavelet_type, synthesis_params,
        ) = render_parameter_tabs(waveform_preview, sr_preview)

    # Guard: expander may not have rendered if not expanded on first load
    if "synthesis_params" not in locals():
        synthesis_params = {}
        target_size    = IMAGE_SIZE_DEFAULT
        output_mode    = OUTPUT_MODE_OPTIONS[4]
        section_layout = SECTION_LAYOUT_OPTIONS[0]
        n_sections     = 1
        wavelet_type   = WAVELET_OPTIONS[0]

    # ── Run computation ───────────────────────────────────────────────────────
    if st.session_state.run_requested and st.session_state.audio_bytes is not None:
        progress_status_placeholder.info("Preparing computation — 0%")
        progress_bar = progress_bar_placeholder.progress(0)

        with st.spinner("Loading audio…"):
            waveform, sr = load_audio(st.session_state.audio_bytes)

        if waveform is None or sr is None:
            st.session_state.run_in_progress = False
            st.session_state.run_requested   = False
            st.session_state.last_run_status = None
            progress_bar_placeholder.empty()
            progress_status_placeholder.error("Could not decode the audio file.")
        else:
            k_max_runtime      = compute_max_sections(len(waveform), target_size)
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
                "png_bytes":       image_to_png_bytes(image_rgb),
                "duration":        len(waveform) / sr,
                "sr":              sr,
                "n_samples":       len(waveform),
                "n_sections":      n_sections_runtime,
                "section_layout":  section_layout,
                "output_mode":     output_mode,
            }

            st.session_state.run_in_progress = False
            st.session_state.run_requested   = False
            st.session_state.last_run_status = "Done"

            # Force rerun so the Run button re-activates and the result appears immediately.
            st.rerun()


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    configure_page()
    init_session_state()
    render_portfolio_links()

    app_tab, doc_fr_tab, doc_en_tab = st.tabs(["App", "Documentation FR", "Documentation EN"])

    with app_tab:
        render_app_tab()

    with doc_fr_tab:
        render_documentation_tab(DOC_FR_TITLES, DOC_FR_SECTIONS, "doc_fr_title")

    with doc_en_tab:
        render_documentation_tab(DOC_EN_TITLES, DOC_EN_SECTIONS, "doc_en_title")
