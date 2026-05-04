## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline at a Glance](#2-pipeline-at-a-glance)
3. [Notation](#3-notation)
4. [Audio Feature Extraction](#4-audio-feature-extraction)
5. [Building the 2D Fourier Spectrum](#5-building-the-2d-fourier-spectrum)
6. [Hermitian Symmetry and Image Reconstruction](#6-hermitian-symmetry-and-image-reconstruction)
7. [Output Modes](#7-output-modes)
8. [Segmentation Methods](#8-segmentation-methods)
9. [Sectioned Synthesis](#9-sectioned-synthesis)
10. [Section Layout Algorithms](#10-section-layout-algorithms)
11. [Post-Processing Pipeline](#11-post-processing-pipeline)
12. [Parameter Reference](#12-parameter-reference)
13. [Limitations](#13-limitations)

---

## 1. Overview

This application converts a mono audio signal into a square image using only classical signal-processing operations. No trained model is involved at any stage. The same audio with the same parameters always produces the same image.

The core principle is **inverse spectral synthesis in two dimensions**. Any real-valued $N \times N$ image has a unique 2D discrete Fourier transform (DFT), a complex matrix whose entries encode the amplitude and phase of spatial sinusoids. The inverse direction — constructing a complex spectrum and inverting it — is equally valid and produces a legitimate spatial image. The application takes this route: it extracts a rich set of audio features, maps them onto the magnitude and phase of a synthetic $N \times N$ complex matrix, and recovers the image via the 2D inverse DFT.

The magnitude of a spectral coefficient at spatial frequency $(u,v)$ determines how much energy that sinusoidal pattern contributes to the image. The phase determines where in space that pattern is positioned. These two degrees of freedom are populated from distinct groups of audio features: energy-based representations (STFT magnitude, CWT magnitude, mel spectrogram, chroma, MFCC, RMS) feed the magnitude grid; phase-based and temporal representations (STFT phase, CWT instantaneous phase, onset strength, spectral centroid, ZCR) feed the phase grid. The resulting image encodes the spectro-temporal structure of the audio in a form that is visually interpretable and fully traceable back to its source.

$$
x[n]
\;\xrightarrow{\;\text{feature extraction}\;}
\bigl\{\mathbf{F}_i\bigr\}
\;\xrightarrow{\;\text{grid construction}\;}
\bigl(\widetilde{M}[u,v],\;\widetilde{\Phi}[u,v]\bigr)
\;\xrightarrow{\;\text{IFFT2}\;}
f[x,y]
\;\xrightarrow{\;\text{post-processing}\;}
\text{RGB image}
$$

---

## 2. Pipeline at a Glance

The table below summarizes each stage of the pipeline, what it produces, and where that output is consumed.

| Stage | Output | Consumed by |
|---|---|---|
| Multi-resolution STFT | log-magnitude matrices; unwrapped phase matrices | Magnitude grid (magnitude); Phase grid (phase) |
| Continuous Wavelet Transform | CWT magnitude; CWT instantaneous phase (Morlet only) | Magnitude grid; Phase grid |
| Mel spectrogram | log-mel energy matrix | Magnitude grid |
| Chroma | pitch-class energy matrix | Magnitude grid |
| MFCC | cepstral coefficient matrix | Magnitude grid |
| RMS energy | scalar time series | Magnitude grid |
| Spectral centroid | scalar time series | Phase grid |
| Onset strength | scalar time series | Phase grid |
| Zero-crossing rate | scalar time series | Phase grid |
| Magnitude grid $\widetilde{M}[u,v]$ | $N \times N$ float in $[0,1]$ | Complex spectrum assembly |
| Phase grid $\widetilde{\Phi}[u,v]$ | $N \times N$ float in $(-\pi, \pi]$ | Complex spectrum assembly |
| Hermitian symmetrization | $N \times N$ complex $Z_\text{sym}$ | 2D IFFT |
| IFFT2 + normalization | floating-point spatial image | Output modes, sectioned assembly |
| Output mode | RGB image patch | Sectioned canvas or final output |
| Post-processing | final $N \times N$ uint8 RGB image | Export |

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $x[n]$ | mono audio waveform, $n = 0, \ldots, L-1$ |
| $L$ | total number of samples |
| $f_s$ | sample rate in Hz |
| $N$ | output image side length in pixels |
| $N_k$ | STFT window length (power of two) |
| $H_k = N_k/4$ | STFT hop length (75 % overlap) |
| $X_{N_k}[m,t]$ | complex STFT coefficient, frequency bin $m$, frame $t$ |
| $W[s,n]$ | complex CWT coefficient, scale $s$, time $n$ |
| $S$ | number of CWT scales |
| $B$ | number of mel filterbank bands |
| $C$ | number of MFCC coefficients |
| $\widetilde{M}[u,v]$ | constructed magnitude grid, $N \times N$, values in $[0,1]$ |
| $\widetilde{\Phi}[u,v]$ | constructed phase grid, $N \times N$, values in $(-\pi,\pi]$ |
| $Z_\text{sym}[u,v]$ | Hermitian-symmetrized spectrum |
| $f[x,y]$ | reconstructed spatial image, $\operatorname{Re}(\text{IFFT2}(Z_\text{sym}))$ |
| $k$ | number of temporal sections |
| $\overline{\mathbf{A}}$ | array $\mathbf{A}$ linearly rescaled to $[0,1]$ |

Frequency bin $m$ of an STFT at rate $f_s$ with window $N_k$ corresponds to $f_m = m f_s / N_k$ Hz. The DC coefficient of the 2D DFT sits at index $(0,0)$ following the NumPy convention. Spatial indices: $x$ is the column (horizontal), $y$ is the row (vertical).

---

## 4. Audio Feature Extraction

The waveform is loaded at its original sample rate without resampling, downmixed to mono by channel averaging if necessary, and truncated to 60 seconds. librosa normalizes the signal to $[-1, 1]$. All features are extracted directly at $f_s$, so the physical frequency interpretation of every bin remains exact.

### 4.1 Multi-Resolution STFT

The STFT decomposes the signal into simultaneous time and frequency information by applying a DFT to overlapping windowed segments. With a Hann window $w[n] = \frac{1}{2}(1 - \cos(2\pi n / N_k))$ and hop length $H_k = N_k / 4$:

$$
X_{N_k}[m,t] = \sum_{n=0}^{N_k-1} x[n + t H_k]\, w[n]\, e^{-j2\pi mn/N_k}, \qquad m = 0, \ldots, \lfloor N_k/2 \rfloor
$$

The Hann window suppresses spectral leakage: without tapering, the abrupt truncation at frame boundaries introduces artificial high-frequency content. Its sidelobes decay as $1/f^3$, which is the standard choice for audio analysis.

A single window length forces a hard trade-off between time and frequency resolution, formalized by the Gabor–Heisenberg uncertainty principle $\sigma_t \cdot \sigma_f \geq 1/(4\pi)$. To mitigate this, the pipeline computes STFTs at all powers of two $N_k \in [N_{\min}, N_{\max}]$ such that $N_k \leq L/2$, defaulting to the set $\{256, 512, 1024, 2048, 4096, 8192\}$. Fine-window STFTs ($N_k = 256, 512$) resolve fast transients and onsets; coarse-window STFTs ($N_k = 4096, 8192$) resolve closely spaced harmonic partials. All resolutions contribute equally to the final magnitude grid, with individual weights $w_{N_k} = w_{\text{STFT}} / |\mathcal{R}|$.

Before blending, the raw magnitude is compressed as $\hat{M} = \log(1 + |X_{N_k}[m,t]|)$. This approximates the decibel scale and prevents the few loudest transients from dominating the range.

The STFT is also the primary source for the **phase grid**. Two resolutions contribute directly: $N_k = 1024$ (weight 0.30) as the primary phase source with balanced time-frequency resolution, and $N_k = 512$ (weight 0.20) for finer temporal phase tracking of fast transients. Before interpolation, the wrapped phase $\angle X_{N_k} \in (-\pi, \pi]$ is unwrapped along both the frequency and time axes to remove artificial $\pm 2\pi$ jumps, producing a smoothly varying phase field that interpolates to $N \times N$ without discontinuity artifacts.

### 4.2 Continuous Wavelet Transform

The STFT analyses all frequencies with the same time resolution. The CWT instead scales the analysis window proportionally to the oscillation period, giving finer time resolution at high frequencies and finer frequency resolution at low frequencies — the constant-$Q$ property that matches the cochlea's behavior. The transform at scale $s$ and time $\tau$ is:

$$
W[s, \tau] = \frac{1}{\sqrt{s}} \sum_{n=0}^{L-1} x[n]\, \overline{\psi\!\left(\frac{n - \tau}{s}\right)}
$$

The factor $1/\sqrt{s}$ normalizes energy across scales, and the conjugated time-reversal turns the operation into a matched filter: $W[s,\tau]$ is large when the signal locally resembles the wavelet at scale $s$ near $\tau$.

**Morlet wavelet.** The default choice is $\psi(t) = \pi^{-1/4} e^{j\omega_0 t} e^{-t^2/2}$ with $\omega_0 = 6$. The Gaussian envelope localizes the wavelet in time; the complex exponential makes it oscillate at $\omega_0 / s$ rad/sample. Crucially, the Morlet wavelet is **analytic**: its Fourier transform is negligible at negative frequencies (the DC admissibility error is $e^{-\omega_0^2/2} = e^{-18} \approx 10^{-8}$). Analyticity means the complex coefficients $W[s,\tau]$ have a well-defined **instantaneous phase** $\angle W[s,\tau]$: for a pure sinusoid at frequency $\omega$, this phase advances at exactly rate $\omega$ in $\tau$, encoding the true local phase of the signal at scale $s$. This instantaneous phase feeds the phase grid at weight 0.20.

**Ricker (Mexican hat) wavelet.** The alternative is $\psi(t) \propto (1 - t^2/\sigma^2) e^{-t^2/(2\sigma^2)}$, the second derivative of a Gaussian. It is real-valued, so no analytic signal exists and no instantaneous phase is defined. In Ricker mode, only $|W[s,\tau]|$ is used; the CWT phase weight (0.20) is redistributed to the STFT phase sources.

Scales are log-spaced from $s = 1$ to $s_{\max} = \min(512, L'/2)$ across $S = 64$ steps. Log-spacing gives equal numbers of scales per octave. Because CWT cost grows as $O(L \cdot S \cdot s_{\max})$, the signal is decimated to at most $L' = 44\,100$ samples before CWT, preserving broad spectral shape while keeping compute tractable.

### 4.3 Perceptual Representations

**Mel spectrogram.** The mel scale is a perceptual pitch scale that models cochlear frequency resolution: approximately linear below 1 000 Hz and logarithmic above. A bank of $B = 128$ triangular filters uniformly spaced on the mel axis maps the STFT power spectrum to mel-band energies:

$$
M[b, t] = \sum_{m=0}^{N_k/2} H_b[m] \cdot |X_{N_k}[m,t]|^2
$$

where $H_b[m]$ is the triangular passband of filter $b$. The log-compressed result $\log(1 + M[b,t])$ feeds the magnitude grid at weight 0.18. Compared to the raw STFT, the mel spectrogram emphasizes the frequency range most relevant to human perception and suppresses the upper bins where many STFT bins map to a single mel band.

**Chroma.** Chroma folds the STFT spectrum onto the 12 pitch classes of the equal-tempered scale (C, C#, …, B) by summing energy across all octaves:

$$
C[p, t] = \sum_{\{m\,:\,\lfloor 12 \log_2(f_m / f_{\text{ref}})\rfloor \bmod 12\, =\, p\}} |X_{N_k}[m,t]|
$$

with $f_{\text{ref}} = 261.63$ Hz (middle C). Chroma is octave-invariant: a C major chord in any register produces the same 12-dimensional vector. It captures tonal and harmonic content independently of absolute pitch, contributing to the magnitude grid at weight 0.09.

**MFCC.** The Mel-Frequency Cepstral Coefficients are the DCT-II of the log-mel spectrum:

$$
\text{cc}[c, t] = \sum_{b=0}^{B-1} \log M[b, t]\;\cos\!\left(\frac{\pi c}{B}\!\left(b + \tfrac{1}{2}\right)\right), \qquad c = 0, \ldots, C-1
$$

with $C = 20$ coefficients. The DCT-II approximately diagonalizes the covariance matrix of mel spectra, so the coefficients are nearly decorrelated: each carries nearly independent information about the spectral envelope. Coefficient 0 tracks log-energy; coefficients 1–13 encode broad timbral shape; higher coefficients add finer texture. Absolute values $|\text{cc}[c,t]|$ contribute to the magnitude grid at weight 0.09.

### 4.4 Temporal Scalar Features

These features produce a single time series rather than a time-frequency matrix, and their primary role is in the **phase grid** where they introduce temporal structure as spatially varying phase offsets.

**RMS energy** $\text{rms}[t] = \sqrt{\frac{1}{N_{\min}}\sum_n x[n+tH]^2}$ tracks instantaneous loudness. It contributes a small weight (0.04) to the magnitude grid as a spatial amplitude modulation: loud passages produce more energy in the spectrum.

**Zero-crossing rate** $\text{zcr}[t]$ counts sign changes per sample within each frame. High ZCR indicates noisy or fricative content; low ZCR indicates tonal content. It contributes to the phase grid at weight 0.05, scaled to a phase range of $\pm\pi/4$.

**Spectral centroid** $\mu_f[t] = \sum_m f_m |X|^2 / \sum_m |X|^2$ is the power-weighted mean frequency, strongly correlated with perceived brightness. It contributes to the phase grid at weight 0.10, scaled to $\pm\pi$.

**Onset strength** detects note attacks and rhythmic events as the mean of the positive half-wave rectified first-order difference of the log-mel spectrogram across bands: $\text{onset}[t] = \frac{1}{B}\sum_b \max(0,\, \log M[b,t] - \log M[b,t-1])$. It is large when spectral energy suddenly increases, and contributes to the phase grid at weight 0.15, scaled to $\pm\pi/2$.

All scalar features follow the same path into the 2D grids: normalize to $[0,1]$, resample to $N$ points, then replicate across all $N$ rows to produce an $N \times N$ array whose columns represent time. This encodes time as the horizontal axis of the spatial image.

---

## 5. Building the 2D Fourier Spectrum

### 5.1 Interpolation to $N \times N$

Every feature matrix has a shape determined by its own parameters (window size, hop, number of scales, etc.) and the signal length, all generally different from $N \times N$. Resampling to $N \times N$ uses a 2D bicubic spline of degree 3, evaluated on a normalized $[0,1]^2$ destination grid. Bicubic splines produce $C^1$-continuous results and avoid the ringing of sinc interpolation. If a source axis has fewer than 4 points, the spline degree is reduced to $\min(3, \text{size}-1)$.

### 5.2 Band Slicing for Color Mode

In Colors mode (and modes derived from it), the normalized frequency axis of each feature is partitioned into three bands before interpolation: Low $[0, \alpha)$, Mid $[\alpha, \beta)$, High $[\beta, 1)$ with default splits $\alpha = 1/3$, $\beta = 2/3$. Each band is interpolated and processed independently, generating three distinct spatial channels stacked as R (Low), G (Mid), B (High). The result is a spectral coloring of the image: bass energy drives the red channel, midrange drives green, and high-frequency energy drives blue.

### 5.3 Magnitude Grid

The magnitude grid $\widetilde{M} \in [0,1]^{N \times N}$ is the weighted sum of the six contributions listed below. Each feature matrix is independently normalized to $[0,1]$ and interpolated to $N \times N$ before blending. All weights are auto-normalized to sum to 1.

$$
\widetilde{M} = w_{\text{STFT}}\,\overline{M}_{\text{STFT}} + w_{\text{CWT}}\,\overline{M}_{\text{CWT}} + w_{\text{mel}}\,\overline{M}_{\text{mel}} + w_{\text{chr}}\,\overline{C} + w_{\text{mfcc}}\,\overline{|\text{cc}|} + w_{\text{RMS}}\,\overline{E}
$$

| Contribution | Default weight | What it encodes |
|---|---|---|
| STFT (all resolutions averaged) | 0.45 | Multi-scale spectral energy |
| Mel spectrogram | 0.18 | Perceptually weighted spectral energy |
| CWT magnitude | 0.15 | Constant-$Q$ spectral energy |
| Chroma | 0.09 | Tonal/harmonic pitch-class content |
| MFCC (absolute values) | 0.09 | Spectral envelope shape |
| RMS energy | 0.04 | Loudness dynamics |

### 5.4 Phase Grid

Phase is the dominant factor in visual image structure: two images with identical magnitude spectra but shuffled phases are perceptually unrelated. A classical demonstration shows that when the phase spectra of two images are swapped, the output looks like the image whose phase was used, not the one whose magnitude was used. This is why the phase grid receives more design attention than the magnitude grid.

The phase grid $\widetilde{\Phi} \in (-\pi, \pi]^{N \times N}$ blends six sources:

$$
\widetilde{\Phi} = w_{\text{mid}}\,\Phi_{1024} + w_{\text{fine}}\,\Phi_{512} + w_{\text{cwt}}\,\Phi_{\text{CWT}} + w_{\text{onset}}\,\Phi_{\text{onset}} + w_{\text{cen}}\,\Phi_{\text{centroid}} + w_{\text{zcr}}\,\Phi_{\text{ZCR}}
$$

The STFT phases at $N_k = 1024$ and $N_k = 512$ are unwrapped along both axes before interpolation. Phase unwrapping removes the artificial $\pm 2\pi$ jumps introduced by $\operatorname{atan2}$: consecutive frames of a stationary sinusoid should show smoothly advancing phase, not random wraps. The cumulative correction at each step is $2\pi \cdot \operatorname{round}\!\bigl((\phi[m,t] - \phi[m,t-1])/(2\pi)\bigr)$. The unwrapped field is smooth and interpolates to $N \times N$ without discontinuity artifacts.

The temporal scalar features (onset, centroid, ZCR) are broadcast to $N \times N$ (Section 4.4) and scaled to fixed sub-ranges: onset to $\pm\pi/2$, centroid to $\pm\pi$, ZCR to $\pm\pi/4$. These ranges are calibrated so that each feature modulates the phase noticeably without oversaturating it. After summation the result is wrapped back to $(-\pi, \pi]$.

Default phase weights (Morlet): $w_{\text{mid}} = 0.30$, $w_{\text{fine}} = 0.20$, $w_{\text{cwt}} = 0.20$, $w_{\text{onset}} = 0.15$, $w_{\text{cen}} = 0.10$, $w_{\text{zcr}} = 0.05$. In Ricker mode the CWT weight is redistributed proportionally to $w_{\text{mid}}$ and $w_{\text{fine}}$.

---

## 6. Hermitian Symmetry and Image Reconstruction

The 2D IFFT of an arbitrary complex matrix produces a complex spatial image. For the output to be real-valued, the input spectrum must satisfy the **Hermitian symmetry** condition:

$$
F[u,v] = \overline{F[(-u)\bmod N,\;(-v)\bmod N]} \qquad \forall\, u, v
$$

The assembled spectrum $Z = \widetilde{M} \cdot e^{j\widetilde{\Phi}}$ does not satisfy this in general, since $\widetilde{M}$ and $\widetilde{\Phi}$ are built from audio features with no structural link between $(u,v)$ and its Hermitian conjugate index. The unique nearest Hermitian-symmetric matrix to $Z$ in the Frobenius norm is:

$$
Z_\text{sym}[u,v] = \frac{Z[u,v] + \overline{Z[(-u)\bmod N,\;(-v)\bmod N]}}{2}
$$

This projection is implemented as $Z_\text{sym} = (Z + \overline{Z_r})/2$ where $Z_r = \operatorname{roll}_{+1,+1}(Z[::-1,::-1])$, the flip-then-roll achieving the modular conjugate index mapping efficiently. At the DC index $(0,0)$, $Z_r[0,0] = Z[0,0]$, so $Z_\text{sym}[0,0] = \operatorname{Re}(Z[0,0]) \in \mathbb{R}$, ensuring the spatial mean of the output is a real number. After symmetrization, $\bigl|\operatorname{Im}(\text{IFFT2}(Z_\text{sym}))\bigr| \sim 10^{-14}$ (floating-point precision only) and is discarded.

The reconstructed image is:

$$
f[x, y] = \operatorname{Re}\!\left(\frac{1}{N^2} \sum_{u=0}^{N-1}\sum_{v=0}^{N-1} Z_\text{sym}[u,v]\; e^{j2\pi(ux+vy)/N}\right)
$$

The real part is a floating-point $N \times N$ image encoding the interference pattern of all the spatial frequency components specified by $Z_\text{sym}$.

---

## 7. Output Modes

Five modes are available, organized as a hierarchy. Grayscale and Colors each perform one independent reconstruction. Black mix, Luma mix, and Watershed each run the pipeline twice — once in Colors mode and once in Grayscale mode — and then combine the results.

**Grayscale.** A single $(\widetilde{M}, \widetilde{\Phi})$ pair is built at full bandwidth. The IFFT2 produces one real channel, replicated across R, G, B.

**Colors.** The frequency axis is split into three bands (Section 5.2). Three independent IFFT2 reconstructions produce three spatial channels stacked as R, G, B, creating a spectral coloring where frequency content is directly visible as color.

**Black mix.** The Colors and Grayscale images are both computed. The grayscale channel is optionally Gaussian-smoothed and then binarized with Otsu's threshold, which maximizes the between-class variance:

$$
\sigma_B^2(\theta) = \omega_0(\theta)\,\omega_1(\theta)\,[\mu_0(\theta) - \mu_1(\theta)]^2
$$

where $\omega_i$ and $\mu_i$ are the class probability mass and mean at threshold $\theta$. The user selects which Otsu class (dark, bright, or the automatic minority class) forms the drawing mask; within that class, the top $d\%$ most extreme pixels are kept and optionally morphologically dilated by $t$ pixels. These pixels are set to black in the Colors image.

**Luma mix.** The Colors image is multiplied channel-wise by a luminance coefficient map derived from the Grayscale image. The grayscale channel $g$ is normalized, optionally blurred, raised to a user-defined gamma $\gamma_\alpha$, and mixed with a minimum floor $\alpha_{\min}$ at strength $\lambda$:

$$
I_\text{out}[x,y,c] = I_\text{color}[x,y,c]\cdot\bigl[(1-\lambda) + \lambda(\alpha_{\min} + (1-\alpha_{\min})\cdot g[x,y]^{\gamma_\alpha})\bigr]
$$

Regions where the grayscale image is dark are darkened multiplicatively; bright regions are left at full color.

**Watershed.** Applies a segmentation algorithm to the Luma mix image to produce a mosaic of filled regions. The segmentation method is chosen independently of this output mode (see Section 8).

---

## 8. Segmentation Methods

Segmentation partitions the Luma mix image into connected regions, each filled with a representative color. All five methods below share the same coloring strategy and boundary rendering logic: the representative color for each region is by default a **randomly selected pixel** from within the region. This is a deliberate choice — for noise-like images (which is the typical output of the spectral synthesis pipeline), the mean or median of any large region converges toward gray by the law of large numbers, washing out color. A random pixel preserves the full color diversity of the source. Mean and median are available but not recommended for this application.

Boundary rendering is optional for all methods: the region boundaries can be drawn as black, as the local color mean, or left invisible.

### 8.1 Watershed

The watershed algorithm treats the image gradient magnitude as a topographic surface and simulates a flood from seed markers placed in gradient valleys. The Sobel gradient $g = \sqrt{g_x^2 + g_y^2}$ is computed on the smoothed grayscale image. Markers are seeded at the local gradient minimum within each $d \times d$ cell of a regular grid. A min-heap flood expands each labeled region by the **minimax path cost**:

$$
\text{cost}(p) = \max\bigl(\text{cost}(\text{parent}(p)),\; g[p]\bigr)
$$

This criterion stops the flood at gradient ridges (edges), producing region boundaries that align with the actual edges of the image. The number of regions is controlled by the marker spacing $d$: smaller spacing produces more, smaller regions.

### 8.2 K-means

Pixels are clustered in RGB color space using the standard K-means algorithm (`scipy.cluster.vq.kmeans2`). The $N \times N = n_p$ pixels are reshaped to an $(n_p, 3)$ float array, whitened (each channel divided by its standard deviation), and clustered into $k$ centroids by iteratively assigning each pixel to its nearest centroid and recomputing centroids. The default $k = 120$ targets the 100–150 region range. Since the centroids are mathematical points in color space rather than actual image pixels, they are not used for coloring — region color comes from the random pixel strategy.

K-means uses only scipy (already a dependency) and is always available as a fallback.

### 8.3 SLIC

SLIC (Simple Linear Iterative Clustering) produces **superpixels** by clustering pixels in a joint $(R, G, B, x, y)$ feature space. The spatial coordinates are included with a compactness weight that trades off between color homogeneity and spatial regularity: high compactness → square, grid-like regions; low compactness → irregular, color-following regions. SLIC initializes cluster centers on a regular grid and iterates a local K-means step within a $2s \times 2s$ search window around each center, where $s = \sqrt{n_p / k}$ is the grid spacing. This locality makes it $O(n_p)$ regardless of $k$, unlike global K-means. Requires scikit-image; falls back to K-means if unavailable.

### 8.4 Felzenszwalb

Felzenszwalb's algorithm constructs a minimum spanning tree of the pixel adjacency graph and merges adjacent pixels greedily. Two components $A$ and $B$ are merged when the edge weight $w(A,B)$ between them (the minimum color difference at their boundary) is small relative to the internal variation of each component:

$$
w(A, B) \leq \min\!\left(\max_{e \in A} w(e) + \frac{\tau}{|A|},\;\; \max_{e \in B} w(e) + \frac{\tau}{|B|}\right)
$$

where the $\tau/|$region$|$ terms enforce a minimum internal evidence before merging. The scale parameter $\tau$ directly controls region granularity: larger $\tau$ produces fewer, larger regions. The number of regions is **determined automatically** — there is no need to specify $k$ in advance. Requires scikit-image; falls back to K-means if unavailable.

### 8.5 Mean-shift

Mean-shift is a mode-seeking algorithm: each sample iteratively moves toward the weighted centroid of its neighborhood until convergence. Applied to pixel colors, it finds the modes of the color density and clusters pixels by which mode they converge to. The neighborhood size is controlled by the bandwidth $h$ (the kernel radius in color space); larger $h$ produces fewer, larger regions. The number of clusters is also automatic.

Because mean-shift cost is $O(n_p^2)$ per iteration, the image is downsampled to a user-specified maximum side length (default 64 px) before clustering. Labels for original pixels are recovered by assigning each to the cluster of its spatially nearest downsampled pixel. Bandwidth can be set manually or estimated automatically from the data using a quantile of the pairwise distance distribution. Requires scikit-learn; falls back to K-means if unavailable.

---

## 9. Sectioned Synthesis

By default, the full waveform generates a single image. When $k > 1$ sections are selected, the waveform is split into $k$ chronological, non-overlapping segments with near-equal lengths:

$$
x_i[n] = x\!\left[\left\lfloor \tfrac{iL}{k}\right\rfloor + n\right], \quad n = 0,\ldots, L_i - 1, \quad L_i = \left\lfloor \tfrac{(i+1)L}{k}\right\rfloor - \left\lfloor \tfrac{iL}{k}\right\rfloor
$$

Each section is processed through the complete feature extraction and IFFT2 pipeline independently, producing a raw floating-point patch. No per-patch normalization is applied: patches retain their raw values so that the global normalization applied after assembly treats the entire canvas uniformly. This prevents the intensity discontinuities at patch boundaries that independent per-patch normalization would create.

After all patches are placed, a single global percentile normalization is applied to the assembled canvas:

$$
\hat{f}[x,y,c] = \operatorname{clip}\!\left(\frac{f[x,y,c] - q_{p_1}}{q_{p_2} - q_{p_1}},\; 0,\; 1\right)
$$

with defaults $p_1 = 1\%$, $p_2 = 99\%$. Percentiles rather than global min/max provide robustness against isolated extreme pixels.

The maximum allowed section count is $k_{\max} = \min(\lfloor L / L_{\min}\rfloor,\; \lfloor N / 32\rfloor^2,\; 64)$ where $L_{\min} = 16\,384$ samples is the minimum length for all features to be meaningful (it must accommodate the largest STFT window, the CWT scale range, and several MFCC frames).

---

## 10. Section Layout Algorithms

Each layout assigns every pixel of the $N \times N$ canvas to one of the $k$ sections. Section 0 always corresponds to the beginning of the audio, section $k-1$ to the end. Six spatial arrangements are available.

**Chronological treemap.** The canvas is recursively split in two along its longest side, with the split position proportional to the section counts assigned to each half: $W_1 = \lfloor W \lceil k/2 \rceil / k \rfloor$. This slice-and-dice strategy produces one rectangle per section with nearly equal areas. For the treemap, each patch is generated at size $\max(W_\text{rect}, H_\text{rect})$ and then bicubic-resized and center-cropped to fit its rectangle.

**Clockwise circular slices.** Each pixel's clockwise polar angle from the vertical $\theta = \operatorname{atan2}(x-c,\, c-y) \bmod 2\pi$ determines its section: $i = \lfloor k\theta / (2\pi)\rfloor$. Section patches are generated at half the canvas side (reducing compute) and upsampled before masking.

**Concentric circles.** Section index is $i = \lfloor k \cdot r / r_{\max}\rfloor$ where $r$ is the Euclidean distance from center. The implicit radii $r_i = r_{\max}\sqrt{i/k}$ give each annulus equal area $\pi r_{\max}^2 / k$.

**Concentric squares.** Same principle using the Chebyshev distance $r_\infty = \max(|x-c|, |y-c|)$. Section 0 occupies the innermost square; section $k-1$ the outermost frame.

**Vertical / horizontal strips.** $i = \lfloor k \cdot x / N\rfloor$ (vertical, time left-to-right) or $i = \lfloor k \cdot y / N\rfloor$ (horizontal, time top-to-bottom).

---

## 11. Post-Processing Pipeline

After the floating-point canvas is assembled and globally normalized to $[0,1]$, six operations are applied in order. All are performed in floating-point; the result is clipped to $[0,1]$ and converted to `uint8` only at the very end.

**Per-channel balance.** Independent gains $g_R, g_G, g_B \in [0,3]$ scale the three channels before normalization, shifting the overall color balance.

**Robust normalization.** The percentile normalization described in Section 9 is applied here in the single-section case as well.

**Contrast.** A linear stretch centered at 0.5: $\hat{I} \leftarrow \operatorname{clip}(0.5 + c_s(\hat{I} - 0.5), 0, 1)$. Values of $c_s > 1$ spread intensities away from the midpoint; $c_s < 1$ compresses them.

**Brightness.** A simple multiplicative scale: $\hat{I} \leftarrow \operatorname{clip}(b \cdot \hat{I},\, 0,\, 1)$.

**Gamma correction.** The power-law transform $\hat{I} \leftarrow \hat{I}^\gamma$ brightens ($\gamma < 1$) or darkens ($\gamma > 1$) the image nonlinearly. The default $\gamma = 0.85$ partially compensates for the underexposure tendency of log-compressed spectral features, which cluster near zero for quiet signals.

**Saturation scaling.** The ITU-R BT.601 luminance $Y = 0.299R + 0.587G + 0.114B$ is computed and used to interpolate between grayscale and full color: $I'^{(c)} = \operatorname{clip}(Y + s_\text{sat}(I^{(c)} - Y), 0, 1)$. At $s_\text{sat} = 0$ the output is fully desaturated; at $s_\text{sat} = 1$ saturation is unchanged; $s_\text{sat} > 1$ oversaturates.

---

## 12. Parameter Reference

| Parameter | Default | Range | Pipeline stage |
|---|---|---|---|
| Image size $N$ | 512 px | 64–1024, step 16 | All grids |
| Output mode | Watershed | 5 options | §7 |
| Section layout | None | 7 options | §10 |
| Sections $k$ | 32 | 1–$k_{\max}$ | §9 |
| CWT wavelet | Morlet | Morlet / Ricker | §4.2 |
| STFT min window | 256 | {256, 512, …, 8192} | §4.1 |
| STFT max window | 8192 | {256, 512, …, 8192} | §4.1 |
| CWT scales $S$ | 64 | 16–128 | §4.2 |
| CWT max samples | 44 100 | 4096–220 500 | §4.2 |
| Mel bands $B$ | 128 | 32–256 | §4.3 |
| MFCC coefficients $C$ | 20 | 8–64 | §4.3 |
| Magnitude weights | see §5.3 | 0–1, auto-normalized | §5.3 |
| Phase weights | see §5.4 | 0–1, auto-normalized | §5.4 |
| Frequency splits $\alpha, \beta$ | 1/3, 2/3 | 0.10–0.45, 0.55–0.90 | §5.2 |
| RGB normalization mode | Per-channel | Per-channel / Shared | §11 |
| RGB balance $g_c$ | 1.0 | 0–3 | §11 |
| Norm percentiles $p_1, p_2$ | 1 %, 99 % | 0–10 %, 90–100 % | §9, §11 |
| Gamma $\gamma$ | 0.85 | 0.20–2.50 | §11 |
| Contrast $c_s$ | 1.0 | 0.20–3.0 | §11 |
| Brightness $b$ | 1.0 | 0.20–2.5 | §11 |
| Saturation $s_\text{sat}$ | 1.0 | 0–3 | §11 |
| Black mix — Otsu class | Auto minority | Auto / Dark / Bright | §7 |
| Black mix — pixel density | 50 % | 1–100 % | §7 |
| Black mix — smoothing $\sigma$ | 0 | 0–10 | §7 |
| Black mix — thickness | 0 px | 0–8 | §7 |
| Luma — strength $\lambda$ | 1.0 | 0–1 | §7 |
| Luma — min coefficient | 0 | 0–1 | §7 |
| Luma — coeff. gamma $\gamma_\alpha$ | 1.0 | 0.20–3.0 | §7 |
| Luma — coeff. blur $\sigma$ | 0 | 0–12 | §7 |
| Segmentation method | Watershed | 5 methods | §8 |
| Region color mode | Random pixel | Random pixel / Mean / Median | §8 |
| Boundary style | None | None / Black / Local mean | §8 |
| Boundary thickness | 0 px | 0–8 | §8 |
| Watershed — marker spacing | 36 px | 4–160 | §8.1 |
| Watershed — gradient $\sigma$ | 1.3 | 0–8 | §8.1 |
| K-means — clusters $k$ | 120 | 10–400 | §8.2 |
| SLIC — segments | 120 | 10–400 | §8.3 |
| SLIC — compactness | 10.0 | 0.1–50 | §8.3 |
| Felzenszwalb — scale $\tau$ | 100 | 1–500 | §8.4 |
| Felzenszwalb — min size | 20 px | 1–200 | §8.4 |
| Mean-shift — bandwidth $h$ | 0 (auto) | 0–100 | §8.5 |
| Mean-shift — max side | 64 px | 16–128 | §8.5 |

---

## 13. Limitations

**Phase dominates appearance, more than you might expect.** The magnitude grid determines the coarseness and energy distribution of the image; the phase grid determines its visual structure. In practice, changing magnitude weights produces subtle textural variations, while changing phase weights produces drastic structural changes. If the output looks unsatisfying, the phase sources — and in particular their individual weights — are the most productive parameters to explore.

**The constructed spectrum is not the DFT of any natural image.** Magnitude and phase are built from independent audio features with no constraint linking them to each other or to any image prior. The IFFT2 always produces a valid spatial image, but its statistical properties are entirely determined by the feature combination rather than by any generative model of natural images. This is not a defect; it is the defining character of the approach.

**Spatial axes have no principled physical meaning.** Audio features are asymmetric: the time axis and the frequency axis have different physical units and semantics. The 2D Fourier grid is symmetric in $u$ and $v$. There is no principled mapping of time or frequency to any specific spatial direction. Time is partially encoded in the phase trajectories and, in sectioned mode, in the spatial tiling; but vertical and horizontal spatial directions carry no guaranteed meaning.

**Section boundaries reflect real spectral changes.** Adjacent section patches may have noticeably different brightness or color at their boundaries. This is not an artifact of the pipeline — it is an accurate reflection of genuine changes in the signal's spectral character between sections. In layouts where time flows spatially (strips, treemap), such discontinuities encode meaningful transitions: silence to onset, change of timbre, structural section change.

**Computational cost scales with sections and CWT.** The dominant cost terms are the CWT ($O(L' \cdot S \cdot s_{\max})$, controlled by the max samples parameter) and sectioning ($k$ times the single-pass cost). Composite modes (Black mix, Luma mix, Watershed) run the full pipeline twice, doubling the total time. For interactive exploration, $k \leq 8$ and $N \leq 512$ are recommended on a single CPU core.

---
