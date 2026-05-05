from __future__ import annotations

import numpy as np
from PIL import Image

from config import (
    MAX_SECTIONS_UI,
    MIN_BLOCK_SIDE,
    MIN_SECTION_SAMPLES,
    SECTION_LAYOUT_OPTIONS,
)
from utils import get_param, normalize_to_unit_robust, resize_float_image_to_size, resize_float_image_to_square
from features import apply_global_image_adjustments, apply_rgb_balance, extract_features
from grids import audio_to_image_float
from segmentation import (
    apply_black_drawing_from_grayscale,
    apply_grayscale_mix_to_color,
    make_segmentation_image,
)


# ============================================================
# Section count limit
# ============================================================

def compute_max_sections(n_samples: int, target_size: int) -> int:
    """
    Compute a dynamic upper bound for the number of temporal sections.

    The limit is expressed in samples rather than seconds. Each section should
    contain enough samples for the largest STFT window and the CWT/MFCC/onset
    descriptors to remain meaningful, while each visual block should remain
    readable in the final square image.
    """
    k_time  = max(1, int(n_samples) // MIN_SECTION_SAMPLES)
    k_space = max(1, (int(target_size) // MIN_BLOCK_SIDE) ** 2)
    return max(1, min(k_time, k_space, MAX_SECTIONS_UI))


# ============================================================
# Waveform splitting
# ============================================================

def split_waveform_into_sections(waveform: np.ndarray, n_sections: int) -> list[np.ndarray]:
    """Split a waveform into n chronological sections with nearly equal sample counts."""
    n_sections = max(1, int(n_sections))
    boundaries = np.linspace(0, len(waveform), n_sections + 1, dtype=int)
    sections: list[np.ndarray] = []
    for i in range(n_sections):
        start = int(boundaries[i])
        end   = max(start + 1, int(boundaries[i + 1]))
        end   = min(end, len(waveform))
        sections.append(waveform[start:end].copy())
    return sections


# ============================================================
# Treemap layout
# ============================================================

def recursive_chronological_layout(
    x: int, y: int, w: int, h: int, section_start: int, n_sections: int,
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

    n_first  = (n_sections + 1) // 2
    n_second = n_sections - n_first

    if w >= h:
        w_first = min(max(1, int(round(w * n_first / n_sections))), w - 1)
        return (
            recursive_chronological_layout(x, y, w_first, h, section_start, n_first)
            + recursive_chronological_layout(x + w_first, y, w - w_first, h, section_start + n_first, n_second)
        )
    else:
        h_first = min(max(1, int(round(h * n_first / n_sections))), h - 1)
        return (
            recursive_chronological_layout(x, y, w, h_first, section_start, n_first)
            + recursive_chronological_layout(x, y + h_first, w, h - h_first, section_start + n_first, n_second)
        )


# ============================================================
# Patch fitting
# ============================================================

def fit_square_patch_to_rect_float(patch: np.ndarray, rect_w: int, rect_h: int) -> np.ndarray:
    """Fit a floating-point square patch to a rectangular block by centered crop/resize."""
    rect_w = max(1, int(rect_w))
    rect_h = max(1, int(rect_h))
    patch  = np.asarray(patch, dtype=np.float64)
    h, w   = patch.shape[:2]

    x0      = max(0, (w - min(rect_w, w)) // 2)
    y0      = max(0, (h - min(rect_h, h)) // 2)
    cropped = patch[y0:y0 + min(rect_h, h), x0:x0 + min(rect_w, w)]

    if cropped.shape[1] == rect_w and cropped.shape[0] == rect_h:
        return cropped
    return resize_float_image_to_size(cropped, rect_w, rect_h)


# ============================================================
# Section-level mask/layout index map
# ============================================================

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
        theta  = np.arctan2(x - center, center - y)
        theta  = np.where(theta < 0.0, theta + 2.0 * np.pi, theta)
        return np.clip(np.floor(theta / (2.0 * np.pi) * k).astype(int), 0, k - 1)

    if section_layout == "Concentric circles":
        center     = (n - 1.0) / 2.0
        radius     = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        radius_max = max(1.0, float(radius.max()))
        return np.clip(np.floor(radius / radius_max * k).astype(int), 0, k - 1)

    if section_layout == "Concentric squares":
        center     = (n - 1.0) / 2.0
        radius     = np.maximum(np.abs(x - center), np.abs(y - center))
        radius_max = max(1.0, float(radius.max()))
        return np.clip(np.floor(radius / radius_max * k).astype(int), 0, k - 1)

    if section_layout == "Vertical strips":
        return np.clip(np.floor(x / max(1.0, float(n)) * k).astype(int), 0, k - 1)

    if section_layout == "Horizontal strips":
        return np.clip(np.floor(y / max(1.0, float(n)) * k).astype(int), 0, k - 1)

    return None


# ============================================================
# Per-section patch generation
# ============================================================

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
    return audio_to_image_float(features=features, target_size=patch_size, output_mode=output_mode, params=params)


# ============================================================
# Global finalization
# ============================================================

def finalize_sectioned_image(
    canvas_float: np.ndarray, output_mode: str, params: dict | None = None
) -> np.ndarray:
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
        gray  = normalize_to_unit_robust(canvas_float[:, :, 0], lower, upper)
        image = np.stack([gray, gray, gray], axis=2)
        image = apply_global_image_adjustments(image, params, is_grayscale=True)
    else:
        normalization_mode = str(get_param(params, "rgb_normalization_mode", "Per-channel"))
        if normalization_mode == "Shared":
            image = normalize_to_unit_robust(canvas_float, lower, upper)
        else:
            image = np.stack(
                [normalize_to_unit_robust(canvas_float[:, :, c], lower, upper) for c in range(3)],
                axis=2,
            )
        image = apply_rgb_balance(image, params)
        image = apply_global_image_adjustments(image, params, is_grayscale=False)

    return (image * 255.0).round().astype(np.uint8)


# ============================================================
# Top-level sectioned image generator
# ============================================================

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
        None: the whole signal generates one image, with no temporal sectioning.
        Chronological treemap: the recursive equal-area block layout.
          Each local patch is computed near its target block size.
        Clockwise circular slices: chronological angular sectors around the
          center. Each section patch is computed at half the final side length
          (one quarter of the final pixel count), then resized and cropped by
          its angular mask.
        Concentric circles: chronological circular rings with the first
          section at the center and later sections moving outward.
        Concentric squares: chronological square rings with the first
          section at the center and later sections moving outward.
        Vertical strips: chronological left-to-right rectangular strips.
        Horizontal strips: chronological top-to-bottom rectangular strips.

    For Black mix, Luma mix and Watershed modes, the function first generates
    a Colors image and a Grayscale image independently, then combines them.
    """
    target_size = int(target_size)
    n_sections  = max(1, int(n_sections))

    # --- Composite modes: generate both Color and Grayscale, then blend ---
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

    # --- Base modes: Grayscale or Colors ---
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
    canvas   = np.zeros((target_size, target_size, 3), dtype=np.float64)

    if section_layout == "Chronological treemap":
        rectangles = recursive_chronological_layout(0, 0, target_size, target_size, 0, n_sections)

        for idx, rect in enumerate(rectangles):
            def make_step_cb(i=idx):
                def cb(label: str) -> None:
                    if progress_callback is not None:
                        progress_callback(f"Section {i+1}/{n_sections} · {label}", i, n_sections)
                return cb

            section    = sections[rect["section"]]
            local_size = max(8, int(max(rect["w"], rect["h"])))
            patch      = generate_section_patch(
                section=section, sr=sr, patch_size=local_size,
                output_mode=output_mode, wavelet_type=wavelet_type,
                params=params, step_callback=make_step_cb(),
            )
            patch_rect = fit_square_patch_to_rect_float(patch, rect["w"], rect["h"])
            y0, x0     = rect["y"], rect["x"]
            y1, x1     = y0 + rect["h"], x0 + rect["w"]
            canvas[y0:y1, x0:x1] = patch_rect[:rect["h"], :rect["w"]]

            if progress_callback is not None:
                progress_callback(f"Section {idx+1}/{n_sections} · done", idx + 1, n_sections)

        return finalize_sectioned_image(canvas, output_mode, params=params)

    # --- Mask-based layouts ---
    index_map = build_layout_index_map(target_size, n_sections, section_layout)
    if index_map is None:
        raise ValueError(f"Unsupported section layout: {section_layout!r}")

    patch_size = max(8, target_size // 2) if section_layout == "Clockwise circular slices" else target_size

    for idx, section in enumerate(sections):
        def make_step_cb(i=idx):
            def cb(label: str) -> None:
                if progress_callback is not None:
                    progress_callback(f"Section {i+1}/{n_sections} · {label}", i, n_sections)
            return cb

        patch      = generate_section_patch(
            section=section, sr=sr, patch_size=patch_size,
            output_mode=output_mode, wavelet_type=wavelet_type,
            params=params, step_callback=make_step_cb(),
        )
        patch_full = resize_float_image_to_square(patch, target_size)
        canvas[index_map == idx] = patch_full[index_map == idx]

        if progress_callback is not None:
            progress_callback(f"Section {idx+1}/{n_sections} · done", idx + 1, n_sections)

    return finalize_sectioned_image(canvas, output_mode, params=params)
