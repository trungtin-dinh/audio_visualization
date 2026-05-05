from __future__ import annotations

import numpy as np
import scipy.ndimage
import scipy.cluster.vq

from utils import get_param, normalize_to_unit

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
# Otsu threshold (self-contained, avoids scikit-image dependency)
# ============================================================

def otsu_threshold_unit(values: np.ndarray) -> float:
    """
    Compute Otsu's threshold on values normalized to [0, 1].

    Otsu's method maximizes the between-class variance:

        σ²_B(t) = ω₀(t)·ω₁(t)·[μ₀(t) − μ₁(t)]²

    where ω₀, ω₁ are the class probabilities and μ₀, μ₁ are the class means
    at threshold t. The optimal threshold minimizes intra-class variance,
    equivalent to maximizing σ²_B.
    """
    arr = np.clip(np.asarray(values, dtype=np.float64)[np.isfinite(np.asarray(values))], 0.0, 1.0)
    if arr.size == 0:
        return 0.5
    if arr.max() <= arr.min():
        return float(arr.min())

    hist, bin_edges = np.histogram(arr, bins=256, range=(0.0, 1.0))
    hist  = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5

    prob    = hist / total
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    omega   = np.cumsum(prob)
    mu      = np.cumsum(prob * centers)
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    bcv   = np.zeros_like(centers)
    valid = denom > 1e-12
    bcv[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]

    return float(centers[int(np.argmax(bcv))])


# ============================================================
# Shared region coloring and boundary helper
# ============================================================

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

    Defaults to random-pixel coloring: for noise-like images the mean/median
    of any large region converges toward grey, so selecting a single actual
    pixel from the region preserves the full color diversity of the source image.
    """
    image  = np.asarray(image, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int32)
    h, w   = image.shape[:2]

    color_mode = str(get_param(params, "seg_region_color_mode", "Random pixel"))
    seed       = int(get_param(params, "seg_random_seed", 12345))
    rng        = np.random.default_rng(seed)
    out        = np.zeros_like(image, dtype=np.uint8)

    for lab in np.unique(labels):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        region_colors = image[ys, xs].astype(np.float64)
        if color_mode == "Mean color":
            sampled = np.mean(region_colors, axis=0)
        elif color_mode == "Median color":
            sampled = np.median(region_colors, axis=0)
        else:   # default: Random pixel
            idx     = int(rng.integers(0, ys.size))
            sampled = image[ys[idx], xs[idx]].astype(np.float64)
        out[ys, xs] = np.clip(sampled, 0.0, 255.0).round().astype(np.uint8)

    # Build boundary mask
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:, 1:]  |= labels[:, 1:]  != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:]  != labels[:, :-1]
    boundary[1:, :]  |= labels[1:, :]  != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :]  != labels[:-1, :]

    boundary_style    = str(get_param(params, "seg_boundary_style",     "None"))
    thickness         = int(get_param(params, "seg_boundary_thickness", 0))
    if thickness > 0:
        boundary = scipy.ndimage.binary_dilation(boundary, iterations=thickness)

    if boundary_style == "Black":
        out[boundary] = 0
    elif boundary_style == "Local mean":
        window     = max(3, int(get_param(params, "seg_boundary_mean_window", 5)) | 1)
        local_mean = np.stack(
            [scipy.ndimage.uniform_filter(out[:, :, c].astype(np.float64), size=window, mode="nearest")
             for c in range(3)],
            axis=2,
        )
        out[boundary] = np.clip(local_mean[boundary], 0.0, 255.0).round().astype(np.uint8)

    return out


# ============================================================
# Segmentation methods
# ============================================================

def make_kmeans_region_image(image_rgb: np.ndarray, params: dict | None = None) -> np.ndarray:
    """
    Segment image_rgb via K-means clustering in RGB space (scipy.cluster.vq).

    Pixels are reshaped to (N, 3), whitened, and clustered into k centroids.
    Each pixel is assigned the label of its nearest centroid. Region color is
    then chosen by _apply_region_coloring_and_boundaries (random pixel by
    default to avoid grey convergence in noise-like images).
    """
    image  = np.asarray(image_rgb, dtype=np.uint8)
    h, w   = image.shape[:2]
    k      = max(2, min(int(get_param(params, "kmeans_k", 120)), h * w))
    pixels = image.reshape(-1, 3).astype(np.float32)

    whitened = scipy.cluster.vq.whiten(pixels)
    _, labels = scipy.cluster.vq.kmeans2(
        whitened, k=k, iter=10, minit="points",
        seed=int(get_param(params, "seg_random_seed", 12345)),
    )
    return _apply_region_coloring_and_boundaries(image, labels.reshape(h, w).astype(np.int32), params)


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

    image       = np.asarray(image_rgb, dtype=np.uint8)
    n_segments  = max(2,    int(get_param(params,   "slic_n_segments", 120)))
    compactness = max(0.01, float(get_param(params, "slic_compactness", 10.0)))
    sigma       = max(0.0,  float(get_param(params, "slic_sigma", 1.0)))

    labels = skimage_seg.slic(
        image, n_segments=n_segments, compactness=compactness,
        sigma=sigma, start_label=0, channel_axis=2,
    )
    return _apply_region_coloring_and_boundaries(image, labels.astype(np.int32), params)


def make_felzenszwalb_region_image(image_rgb: np.ndarray, params: dict | None = None) -> np.ndarray:
    """
    Segment image_rgb using Felzenszwalb's graph-based algorithm
    (skimage.segmentation.felzenszwalb).

    Merges pixels greedily using a minimum spanning tree: two pixels are merged
    when the edge weight between them is small relative to the internal variation
    of their component. The `scale` parameter directly controls region size.
    Falls back to K-means if scikit-image is unavailable.
    """
    if not SKIMAGE_AVAILABLE:
        return make_kmeans_region_image(image_rgb, params)

    image    = np.asarray(image_rgb, dtype=np.uint8)
    scale    = max(1.0, float(get_param(params, "felz_scale",    100.0)))
    sigma    = max(0.0, float(get_param(params, "felz_sigma",      0.8)))
    min_size = max(1,   int(get_param(params,   "felz_min_size",   20)))

    labels = skimage_seg.felzenszwalb(
        image, scale=scale, sigma=sigma, min_size=min_size, channel_axis=2,
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
    nearest-neighbor (each original pixel is assigned to the cluster of its
    spatially nearest downsampled pixel).
    Falls back to K-means if scikit-learn is unavailable.
    """
    if not SKLEARN_AVAILABLE:
        return make_kmeans_region_image(image_rgb, params)

    image    = np.asarray(image_rgb, dtype=np.uint8)
    h, w     = image.shape[:2]
    max_side = max(16, min(int(get_param(params, "meanshift_max_side", 64)), 128))
    scale    = max(1, max(h, w) // max_side)
    small    = image[::scale, ::scale]
    sh, sw   = small.shape[:2]

    pixels_small = small.reshape(-1, 3).astype(np.float32)
    bw = float(get_param(params, "meanshift_bandwidth", 0.0))
    if bw <= 0:
        bw = float(estimate_bandwidth(pixels_small, quantile=0.2, n_samples=min(500, len(pixels_small))))
    bw = max(1.0, bw)

    ms = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=1)
    ms.fit(pixels_small)
    small_labels = ms.labels_.reshape(sh, sw).astype(np.int32)

    label_map = np.repeat(np.repeat(small_labels, scale, axis=0), scale, axis=1)[:h, :w]
    return _apply_region_coloring_and_boundaries(image, label_map, params)


def make_watershed_region_image(image_rgb: np.ndarray, params: dict | None = None) -> np.ndarray:
    """
    Convert an RGB image into a watershed-segmented region image.

    A Sobel gradient magnitude is computed on a smoothed luminance channel.
    Markers are placed at the local minimum of the gradient in each grid cell.
    The watershed is flooded from these markers using a priority-queue (min-heap)
    implementation that assigns each unlabelled pixel to its lowest-cost
    reachable marker.
    """
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        image = np.stack([image, image, image], axis=2).astype(np.uint8)

    h, w = image.shape[:2]
    n    = max(h, w)

    gray = normalize_to_unit(
        0.299 * image[:, :, 0].astype(np.float64)
        + 0.587 * image[:, :, 1].astype(np.float64)
        + 0.114 * image[:, :, 2].astype(np.float64)
    )

    smooth_sigma = float(get_param(params, "watershed_gradient_smoothing", max(0.8, n / 384.0)))
    gray_smooth  = scipy.ndimage.gaussian_filter(gray, sigma=max(0.0, smooth_sigma))

    grad_x   = scipy.ndimage.sobel(gray_smooth, axis=1)
    grad_y   = scipy.ndimage.sobel(gray_smooth, axis=0)
    gradient = normalize_to_unit(np.hypot(grad_x, grad_y))

    cell_size = max(4, int(get_param(params, "watershed_marker_spacing", max(12, int(round(n / 14.0))))))
    markers   = np.zeros((h, w), dtype=np.int32)
    label     = 1

    for y0 in range(0, h, cell_size):
        y1 = min(h, y0 + cell_size)
        for x0 in range(0, w, cell_size):
            x1  = min(w, x0 + cell_size)
            sub = gradient[y0:y1, x0:x1]
            if sub.size == 0:
                continue
            yy, xx = np.unravel_index(int(np.argmin(sub)), sub.shape)
            markers[y0 + yy, x0 + xx] = label
            label += 1

    labels = _watershed_flood_from_markers(gradient, markers)

    # Mirror watershed-specific param keys to the shared seg_ keys
    merged = dict(params or {})
    merged.setdefault("seg_region_color_mode",      str(get_param(params, "watershed_region_color_mode",    "Random pixel")))
    merged.setdefault("seg_random_seed",             int(get_param(params, "watershed_random_seed",          12345)))
    merged.setdefault("seg_boundary_style",          str(get_param(params, "watershed_boundary_style",       "None")))
    merged.setdefault("seg_boundary_thickness",      int(get_param(params, "watershed_boundary_thickness",   0)))
    merged.setdefault("seg_boundary_mean_window",    int(get_param(params, "watershed_boundary_mean_window", 5)))
    return _apply_region_coloring_and_boundaries(image, labels, merged)


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


# ============================================================
# Lightweight watershed implementation (NumPy + heapq)
# ============================================================

def _watershed_flood_from_markers(gradient: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    Marker-controlled watershed flood using a min-heap priority queue.

    The gradient image is interpreted as a topographic surface. Marker labels
    are flooded outward in order of increasing accumulated gradient cost.
    """
    import heapq

    gradient = np.asarray(gradient, dtype=np.float64)
    markers  = np.asarray(markers,  dtype=np.int32)
    h, w     = gradient.shape

    labels = markers.copy()
    heap: list[tuple[float, int, int, int]] = []

    for y, x in np.argwhere(markers > 0):
        lab = int(markers[y, x])
        heapq.heappush(heap, (float(gradient[y, x]), int(y), int(x), lab))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while heap:
        cost, y, x, lab = heapq.heappop(heap)
        if labels[y, x] != lab:
            continue
        for dy, dx in neighbors:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and labels[yy, xx] == 0:
                labels[yy, xx] = lab
                heapq.heappush(heap, (max(cost, float(gradient[yy, xx])), yy, xx, lab))

    return labels


# ============================================================
# Black-mix and Luma-mix post-processing
# ============================================================

def apply_black_drawing_from_grayscale(
    color_image: np.ndarray,
    grayscale_image: np.ndarray,
    params: dict | None = None,
) -> np.ndarray:
    """
    Use a grayscale-generated image as a sparse black drawing mask over a color image.

    An Otsu threshold partitions pixels into two classes. The minority class
    (or the class selected via `ink_class_choice`) is further filtered to the
    most extreme `ink_keep_percentile` percent and painted black on the color
    image.
    """
    color    = np.asarray(color_image, dtype=np.uint8).copy()
    gray_rgb = np.asarray(grayscale_image, dtype=np.float64)

    gray = (0.299 * gray_rgb[:, :, 0] + 0.587 * gray_rgb[:, :, 1] + 0.114 * gray_rgb[:, :, 2]
            if gray_rgb.ndim == 3 else gray_rgb)
    gray = normalize_to_unit(gray)

    smooth_sigma = float(get_param(params, "ink_smoothing_sigma", 0.0))
    if smooth_sigma > 0:
        gray = scipy.ndimage.gaussian_filter(gray, sigma=smooth_sigma)

    threshold = otsu_threshold_unit(gray)
    low_mask  = gray <= threshold
    high_mask = gray >  threshold

    low_count  = int(np.count_nonzero(low_mask))
    high_count = int(np.count_nonzero(high_mask))

    if low_count == 0 and high_count == 0:
        return color

    class_choice = str(get_param(params, "ink_class_choice", "Automatic minority"))
    if class_choice == "Dark class":
        candidate_mask    = low_mask
        keep_high_extreme = False
    elif class_choice == "Bright class":
        candidate_mask    = high_mask
        keep_high_extreme = True
    elif low_count == 0:
        candidate_mask    = high_mask
        keep_high_extreme = True
    elif high_count == 0:
        candidate_mask    = low_mask
        keep_high_extreme = False
    elif low_count <= high_count:
        candidate_mask    = low_mask
        keep_high_extreme = False
    else:
        candidate_mask    = high_mask
        keep_high_extreme = True

    candidate_values = gray[candidate_mask]
    if candidate_values.size == 0:
        return color

    keep_percent = float(np.clip(get_param(params, "ink_keep_percentile", 50.0), 1.0, 100.0))
    if keep_high_extreme:
        threshold2    = float(np.percentile(candidate_values, 100.0 - keep_percent))
        drawing_mask  = candidate_mask & (gray >= threshold2)
    else:
        threshold2    = float(np.percentile(candidate_values, keep_percent))
        drawing_mask  = candidate_mask & (gray <= threshold2)

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

    The grayscale image is normalized to [0, 1], optionally blurred and
    gamma-corrected, then used to modulate the color image channel-wise:

        output = color · ((1 − strength) + strength · coeff^gamma)

    where coeff ∈ [min_coeff, 1].
    """
    color    = np.asarray(color_image, dtype=np.float64)
    gray_rgb = np.asarray(grayscale_image, dtype=np.float64)

    gray = (0.299 * gray_rgb[:, :, 0] + 0.587 * gray_rgb[:, :, 1] + 0.114 * gray_rgb[:, :, 2]
            if gray_rgb.ndim == 3 else gray_rgb)

    coeff = normalize_to_unit(gray)

    blur_sigma = float(get_param(params, "luma_coeff_blur_sigma", 0.0))
    if blur_sigma > 0:
        coeff = normalize_to_unit(scipy.ndimage.gaussian_filter(coeff, sigma=blur_sigma))

    coeff_gamma = float(get_param(params, "luma_gamma", 1.0))
    if coeff_gamma > 1e-6:
        coeff = coeff ** coeff_gamma

    min_coeff = float(np.clip(get_param(params, "luma_min_coeff", 0.0), 0.0, 1.0))
    coeff     = min_coeff + (1.0 - min_coeff) * coeff

    strength           = float(np.clip(get_param(params, "luma_strength", 1.0), 0.0, 1.0))
    effective_coeff    = (1.0 - strength) + strength * coeff
    mixed              = color * effective_coeff[:, :, None]
    return np.clip(mixed, 0.0, 255.0).round().astype(np.uint8)
