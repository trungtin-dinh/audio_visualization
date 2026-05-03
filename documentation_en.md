## Table of Contents

1. [Overview and Core Idea](#1-overview-and-core-idea)
2. [Notation and Conventions](#2-notation-and-conventions)
3. [Signal Loading and Preprocessing](#3-signal-loading-and-preprocessing)
4. [The Spectral Synthesis Paradigm](#4-the-spectral-synthesis-paradigm)
5. [Feature Extraction: Multi-Resolution STFT](#5-feature-extraction-multi-resolution-stft)
6. [Feature Extraction: Continuous Wavelet Transform](#6-feature-extraction-continuous-wavelet-transform)
7. [Feature Extraction: Mel Spectrogram](#7-feature-extraction-mel-spectrogram)
8. [Feature Extraction: Chroma](#8-feature-extraction-chroma)
9. [Feature Extraction: Mel-Frequency Cepstral Coefficients](#9-feature-extraction-mel-frequency-cepstral-coefficients)
10. [Feature Extraction: Temporal Scalar Features](#10-feature-extraction-temporal-scalar-features)
11. [From Features to 2D Grids: Interpolation and Band Slicing](#11-from-features-to-2d-grids-interpolation-and-band-slicing)
12. [Magnitude Grid Construction](#12-magnitude-grid-construction)
13. [Phase Grid Construction](#13-phase-grid-construction)
14. [Hermitian Symmetry Enforcement](#14-hermitian-symmetry-enforcement)
15. [Image Reconstruction via 2D Inverse DFT](#15-image-reconstruction-via-2d-inverse-dft)
16. [Output Modes](#16-output-modes)
17. [Sectioned Image Synthesis](#17-sectioned-image-synthesis)
18. [Section Layout Algorithms](#18-section-layout-algorithms)
19. [Post-Processing Pipeline](#19-post-processing-pipeline)
20. [Parameter Reference](#20-parameter-reference)
21. [Limitations and Interpretation](#21-limitations-and-interpretation)

---

## 1. Overview and Core Idea

This application converts a mono audio signal into a square image without any trained model. Every step is a deterministic classical signal-processing or image-processing operation. The same audio with the same parameters always produces the same image.

**The fundamental operation.** Any $N \times N$ real-valued image $f[x,y]$ has a unique two-dimensional discrete Fourier transform (2D DFT) $F[u,v] \in \mathbb{C}$, a complex $N \times N$ matrix. Inversely, given any complex $N \times N$ matrix satisfying a symmetry condition (detailed in Section 14), the 2D inverse DFT produces a real-valued spatial image. The application exploits this: instead of starting from an image and computing its DFT, it starts from an audio signal, extracts audio features, assembles a synthetic 2D complex spectrum $Z[u,v]$ from those features, and inverts it to obtain a spatial image. The image is therefore a visual encoding of the spectro-temporal structure of the audio.

Decomposing $Z[u,v] = |Z[u,v]| \cdot e^{j\angle Z[u,v]}$ shows the two degrees of freedom available:

- **The magnitude** $|Z[u,v]|$ controls how much energy is carried by spatial frequency $(u,v)$. High magnitude at low spatial frequencies produces slowly varying, smooth textures; high magnitude at high spatial frequencies produces fine detail and edges.
- **The phase** $\angle Z[u,v]$ controls where in space the spatial frequency component is positioned. Phase is the dominant factor in image appearance: two images with identical magnitude spectra but different phases are visually completely different.

Audio features are mapped onto these two grids. The magnitude grid is populated from energy-based representations (STFT log-magnitude, CWT magnitude, mel spectrogram, chroma, MFCC, RMS). The phase grid is populated from phase-based representations (STFT phase, CWT instantaneous phase, onset strength, spectral centroid, ZCR). The full pipeline is:

$$
x[n]
\;\xrightarrow{\;\text{feature extraction}\;}
\bigl\{\,\mathbf{F}_i\,\bigr\}
\;\xrightarrow{\;\text{grid construction}\;}
\bigl(\widetilde{M}[u,v],\;\widetilde{\Phi}[u,v]\bigr)
\;\xrightarrow{\;\text{symmetrize + IFFT2}\;}
f[x,y]
\;\xrightarrow{\;\text{post-processing}\;}
\text{image RGB}
$$

---

## 2. Notation and Conventions

The following symbols are used consistently throughout this document.

| Symbol | Meaning |
|---|---|
| $x[n]$ | Mono audio waveform, $n = 0, \ldots, L-1$ |
| $L$ | Total number of samples in the waveform |
| $f_s$ | Sample rate in Hz |
| $N$ | Output image side length in pixels (user-selectable, default 512) |
| $N_k$ | STFT window length (a power of two, e.g. 256, 512, …, 8192) |
| $H_k = N_k/4$ | STFT hop length (75 % overlap) |
| $T_k$ | Number of STFT frames at resolution $N_k$; $T_k \approx L / H_k$ |
| $X_{N_k}[m,t]$ | Complex STFT coefficient at frequency bin $m$, frame $t$, resolution $N_k$ |
| $w[n]$ | Analysis window (Hann) |
| $s$ | CWT scale parameter (dimensionless) |
| $W[s,n]$ | Complex CWT coefficient at scale $s$, time sample $n$ |
| $S$ | Number of CWT scales (default 64) |
| $B$ | Number of mel filterbank bands (default 128) |
| $P = 12$ | Number of chroma pitch classes |
| $C$ | Number of MFCC coefficients (default 20) |
| $\widetilde{M}[u,v]$ | Constructed 2D Fourier magnitude grid, shape $N \times N$, values in $[0,1]$ |
| $\widetilde{\Phi}[u,v]$ | Constructed 2D Fourier phase grid, shape $N \times N$, values in $(-\pi,\pi]$ |
| $Z[u,v]$ | Complex spectrum assembled as $\widetilde{M}[u,v] \cdot e^{j\widetilde{\Phi}[u,v]}$ |
| $Z_{\mathrm{sym}}[u,v]$ | Hermitian-symmetrized version of $Z$ |
| $f[x,y]$ | Reconstructed spatial image, $\operatorname{Re}(\mathrm{IFFT2}(Z_{\mathrm{sym}}))$ |
| $k$ | Number of temporal sections (user-selectable, default 32) |
| $x_i[n]$ | Waveform of the $i$-th temporal section |
| $\overline{\mathbf{A}}$ | Array $\mathbf{A}$ normalized to $[0,1]$ by linear min–max rescaling |
| $\operatorname{clip}(v, a, b)$ | Clamp value $v$ to interval $[a,b]$ |
| $j$ | Imaginary unit, $j^2 = -1$ |
| $\bar{z}$ | Complex conjugate of $z$ |

**Index conventions.** All 2D arrays use row-major ordering (row = vertical axis, column = horizontal axis). For a spatial image $f[x,y]$, $x$ is the column index (horizontal) and $y$ is the row index (vertical). For a spectrum $Z[u,v]$, $u$ and $v$ are the horizontal and vertical spatial-frequency indices, with the DC component at $(0,0)$ following the NumPy FFT convention (zero-frequency first).

---

## 3. Signal Loading and Preprocessing

The audio file is decoded to a mono waveform at its **original sample rate** $f_s$ using librosa. No fixed resampling rate is imposed; preserving $f_s$ avoids unnecessary interpolation artifacts and keeps the frequency-bin interpretation of all STFT-based features exact ($f_m = m f_s / N_k$ Hz for bin $m$).

**Downmixing.** If the file is stereo or multichannel, the channels are averaged to produce a single mono waveform before any analysis.

**Duration cap.** The waveform is truncated to at most $L_{\max} = 60 \times f_s$ samples (60 seconds at the original sample rate). Beyond this, multi-resolution STFT and CWT extraction would require tens of seconds on a single CPU core, exceeding interactive latency budgets.

**Amplitude range.** librosa normalizes the waveform to floating-point in $[-1, 1]$. No further amplitude normalization is applied before feature extraction, so all energy-based features (RMS, STFT magnitude, mel spectrogram) faithfully reflect the relative dynamics of the signal.

---

## 4. The Spectral Synthesis Paradigm

### 4.1 The 2D Discrete Fourier Transform

For a real-valued $N \times N$ image $f[x,y]$ (indices $0 \leq x,y \leq N-1$), the 2D DFT is:

$$
F[u,v] = \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f[x,y]\, e^{-j2\pi(ux + vy)/N}
$$

and the inverse:

$$
f[x,y] = \frac{1}{N^2} \sum_{u=0}^{N-1} \sum_{v=0}^{N-1} F[u,v]\, e^{j2\pi(ux + vy)/N}
$$

Each coefficient $F[u,v]$ encodes the contribution of a two-dimensional sinusoidal pattern of spatial frequencies $(u/N, v/N)$ cycles per pixel. The DC coefficient $F[0,0] = \sum_{x,y} f[x,y]$ is proportional to the mean brightness. Coefficients near the DC represent slowly varying large-scale patterns; coefficients near $N/2$ represent fine detail at the Nyquist spatial frequency.

### 4.2 Constructing a Spectrum Produces a Valid Image

The IFFT2 accepts any complex $N \times N$ input. However, for the output to be **real-valued** (a genuine image with no complex part), the input must satisfy the **Hermitian symmetry** condition:

$$
F[u,v] = \overline{F[(-u)\bmod N,\; (-v)\bmod N]}
$$

This is the 2D generalization of the 1D result: the DFT of a real sequence satisfies $F[k] = \overline{F[N-k]}$. If this condition is not satisfied, the IFFT2 produces a complex array; discarding the imaginary part is equivalent to averaging the spectrum with its Hermitian conjugate, introducing an implicit modification of both magnitude and phase. To make this modification explicit and controlled, the assembled spectrum $Z$ is projected onto the nearest Hermitian-symmetric matrix before inversion (Section 14).

### 4.3 What the Audio Features Control

The **magnitude grid** $\widetilde{M}[u,v]$ determines the energy distribution across spatial frequencies. A magnitude grid concentrated at low $(u,v)$ produces a smooth image; one with energy spread across all $(u,v)$ produces a complex textured image. The specific magnitudes assigned from audio features shape the visual coarseness and density of the output.

The **phase grid** $\widetilde{\Phi}[u,v]$ determines the spatial positioning of each frequency component. For a single sinusoid at frequency $(u,v)$, changing its phase shifts the pattern in space. The combined effect across all $(u,v)$ controls whether structures appear at specific locations, whether they form coherent edges or diffuse noise, and whether local patterns are aligned or scrambled. In practice, phase is the dominant factor in visual image structure: two images with identical magnitude spectra but shuffled phases are completely unrecognizable as related. This is why the phase grid is sourced from the most structured audio features available — STFT and CWT instantaneous phase — rather than from random values.

---

## 5. Feature Extraction: Multi-Resolution STFT

### 5.1 Definition

The Short-Time Fourier Transform (STFT) decomposes a signal into simultaneous time and frequency information by applying a DFT to overlapping windowed segments. For window length $N_k$, hop length $H_k = N_k / 4$, and Hann window $w$:

$$
X_{N_k}[m,t] = \sum_{n=0}^{N_k-1} x[n + t H_k]\, w[n]\, e^{-j2\pi mn/N_k}, \qquad m = 0, \ldots, \lfloor N_k/2 \rfloor
$$

The result is a complex matrix of shape $(\lfloor N_k/2 \rfloor + 1) \times T_k$ where $T_k = \lceil L / H_k \rceil$ is the number of frames. The frequency at bin $m$ is $f_m = m f_s / N_k$ Hz. Only the non-redundant bins $0, \ldots, N_k/2$ are kept by the symmetry of the real input.

**Hann window.** The Hann (raised cosine) window $w[n] = \frac{1}{2}(1 - \cos(2\pi n / N_k))$ tapers the signal to zero at both ends before the DFT. Without tapering, the abrupt signal truncation at frame boundaries introduces artificial high-frequency content (spectral leakage). The Hann window's sidelobes decay as $1/f^3$, making it a standard choice in audio analysis.

**75 % overlap.** Setting $H_k = N_k/4$ means consecutive frames overlap by three-quarters. This dense temporal sampling gives smooth spectral trajectories and satisfies the constant-overlap-add reconstruction condition together with the Hann window.

### 5.2 The Time–Frequency Uncertainty Principle

The STFT does not escape the Gabor–Heisenberg uncertainty principle:

$$
\sigma_t \cdot \sigma_f \geq \frac{1}{4\pi}
$$

where $\sigma_t$ and $\sigma_f$ are the root-mean-square spreads of the analysis window in time and frequency respectively. A long window ($N_k$ large) gives fine frequency resolution $\Delta f = f_s / N_k$ but poor time resolution $\Delta t = N_k / f_s$. A short window gives the opposite. There is no window that simultaneously achieves both beyond this fundamental limit.

### 5.3 Multi-Resolution Analysis

To partially circumvent the single-window trade-off, the application computes STFTs at all powers of two in the range $[N_{\min}, N_{\max}]$:

$$
\mathcal{R} = \bigl\{N_k = 2^k : N_{\min} \leq 2^k \leq N_{\max},\; 2^k \leq L/2\bigr\}
$$

with defaults $N_{\min} = 256$, $N_{\max} = 8192$. This spans up to $|\mathcal{R}| = 6$ resolutions: 256, 512, 1024, 2048, 4096, 8192. Fine-window STFTs ($N_k = 256, 512$) resolve onsets and fast transients; coarse-window STFTs ($N_k = 4096, 8192$) resolve close harmonic partials. All resolutions are blended with equal sub-weights:

$$
w_{N_k} = \frac{w_{\text{STFT}}}{|\mathcal{R}|}
$$

so their total contribution equals the global STFT weight $w_{\text{STFT}} = 0.45$ by default.

### 5.4 Magnitude and Phase Extraction

From the complex matrix $X_{N_k}$, the per-bin magnitude and phase are:

$$
|X_{N_k}[m,t]| = \sqrt{\operatorname{Re}(X_{N_k}[m,t])^2 + \operatorname{Im}(X_{N_k}[m,t])^2}
$$

$$
\angle X_{N_k}[m,t] = \operatorname{atan2}\!\bigl(\operatorname{Im}(X_{N_k}[m,t]),\; \operatorname{Re}(X_{N_k}[m,t])\bigr) \in (-\pi, \pi]
$$

The magnitude is logarithmically compressed before use (Section 5.5). The phase is used in the phase grid after unwrapping (Section 13.2).

### 5.5 Log-Magnitude Compression

Raw STFT magnitudes span a very large dynamic range (60–80 dB between the loudest and quietest components). Mapping this linearly to $[0,1]$ would dominate the magnitude grid with a few loud transients. The compression:

$$
\hat{M}_{N_k}[m,t] = \log\bigl(1 + |X_{N_k}[m,t]|\bigr)
$$

compresses the dynamic range while preserving relative ordering. It approximates the decibel scale in a numerically stable form (no $\log(0)$ for silent frames). After compression, $\hat{M}_{N_k}$ is linearly normalized to $[0,1]$.

---

## 6. Feature Extraction: Continuous Wavelet Transform

### 6.1 Motivation: Constant-Q Resolution

The STFT uses a fixed window for all frequencies: time and frequency resolution is the same at 100 Hz as at 10 000 Hz. The cochlea and musical pitch perception both operate with **constant-$Q$** resolution — the bandwidth of each frequency channel is a fixed fraction of its center frequency (roughly one-third of an octave). The CWT achieves this by stretching the analysis window proportionally to scale.

### 6.2 Definition

The CWT of a discrete signal $x[n]$ at scale $s > 0$ and position $\tau$ is:

$$
W[s, \tau] = \frac{1}{\sqrt{s}} \sum_{n=0}^{L-1} x[n]\, \overline{\psi\!\left(\frac{n - \tau}{s}\right)}
$$

The factor $1/\sqrt{s}$ normalizes energy across scales. The conjugation and time-reversal turn the operation into a matched filter: $W[s,\tau]$ is large when the local signal resembles the wavelet $\psi$ at scale $s$ near time $\tau$.

In practice, this is implemented as a convolution of $x$ with the time-reversed conjugated scaled wavelet for each scale:

$$
W[s, \cdot] = x \star \overline{\psi_s[-\cdot]}
$$

computed by `scipy.signal.convolve` with `mode="same"` (output length equals input length). The wavelet is truncated to $\min(10s, L)$ samples; beyond this the Gaussian tails contribute negligibly.

### 6.3 Morlet Wavelet

The Morlet mother wavelet is a complex sinusoid modulated by a Gaussian envelope:

$$
\psi(t) = \pi^{-1/4} \cdot e^{j\omega_0 t} \cdot e^{-t^2/2}
$$

with $\omega_0 = 6$ (default central frequency parameter). The Gaussian $e^{-t^2/2}$ localizes the wavelet in time; the complex exponential $e^{j\omega_0 t}$ makes it oscillatory at frequency $\omega_0$ rad/sample. At scale $s$, the wavelet oscillates at $\omega_0 / s$ rad/sample, corresponding to $\omega_0 f_s / (2\pi s)$ Hz.

The Morlet wavelet is **analytic**: its Fourier transform $\hat{\psi}(\omega)$ is negligible for $\omega < 0$. The analyticity condition requires $\omega_0 \geq 5$; at $\omega_0 = 6$ the DC admissibility error is $|\hat{\psi}(0)| \propto e^{-\omega_0^2/2} = e^{-18} \approx 1.5 \times 10^{-8}$. Analyticity implies that $W[s,\tau]$ has a well-defined **instantaneous phase** $\angle W[s,\tau]$: for a pure sinusoid of angular frequency $\omega$, this phase advances linearly at rate $\omega$ in $\tau$, encoding the true local phase of the signal at scale $s$.

The normalized discrete kernel at scale $s$ is:

$$
\psi_s[n] = \pi^{-1/4} \cdot \frac{1}{\sqrt{s}} \cdot \exp\!\left(j\frac{\omega_0 n}{s}\right) \cdot \exp\!\left(-\frac{n^2}{2s^2}\right), \qquad n = -\left\lfloor\frac{M}{2}\right\rfloor, \ldots, \left\lfloor\frac{M}{2}\right\rfloor
$$

### 6.4 Ricker (Mexican Hat) Wavelet

The Ricker wavelet is the second derivative of a Gaussian:

$$
\psi(t) = \frac{2}{\sqrt{3\sigma}\,\pi^{1/4}} \left(1 - \frac{t^2}{\sigma^2}\right) e^{-t^2/(2\sigma^2)}
$$

It is real-valued, zero-mean, and symmetric. Because it is **real**, it is not analytic: its Fourier transform has support on both positive and negative frequencies, so no instantaneous phase is defined. In Ricker mode, only $|W[s,\tau]|$ is extracted; the CWT phase contribution is set to zero and its weight in the phase grid is redistributed proportionally to the STFT phase sources.

The Ricker wavelet is optimal for detecting localized energy concentrations and step-like transitions. Its shape — a central positive peak flanked by negative wings — acts as a second-derivative filter, making it sensitive to changes in signal curvature.

### 6.5 Scale Grid and Computational Constraint

Scales are geometrically spaced (log-uniform) from $s = 1$ to $s_{\max} = \min(512,\, L_{\mathrm{CWT}}/2)$:

$$
s_k = s_{\max}^{k/(S-1)}, \qquad k = 0, \ldots, S-1, \qquad S = 64 \text{ (default)}
$$

Log-spacing assigns equal numbers of scales per octave, matching the constant-$Q$ principle. The smallest scale resolves the shortest-duration, highest-frequency patterns; the largest scale resolves broad slow patterns.

**Computational truncation.** The CWT at $S$ scales costs $O(L \cdot S \cdot s_{\max})$ operations, which is prohibitive for long signals. The waveform is downsampled to at most $L_{\mathrm{CWT}} = 44{,}100$ samples (default) by taking every $\lceil L / L_{\mathrm{CWT}} \rceil$-th sample. Since the CWT captures broad spectro-temporal structure rather than fine temporal detail, this truncation is acceptable.

---

## 7. Feature Extraction: Mel Spectrogram

### 7.1 The Mel Scale

The mel scale is a perceptual pitch scale derived from psychoacoustic experiments on equal pitch-interval perception. It models the non-linear frequency resolution of the basilar membrane: nearly linear below about 1 000 Hz and logarithmic above. A standard approximation is:

$$
m(f) = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right), \qquad f(m) = 700\!\left(10^{m/2595} - 1\right)
$$

Equal increments in mel correspond to equal perceived pitch differences. At musical frequencies, a semitone in the equal-tempered scale corresponds to approximately the same mel interval regardless of octave.

### 7.2 Mel Filterbank

A bank of $B = 128$ triangular filters is placed uniformly on the mel axis between 0 Hz and $f_s/2$ Hz. Filter $b$ has a triangular passband with peak at mel-domain center $m_b$ and zeros at $m_{b-1}$ and $m_{b+1}$. In the linear frequency domain, filter $b$ maps to:

$$
H_b[m] = \begin{cases}
\dfrac{f_m - f(m_{b-1})}{f(m_b) - f(m_{b-1})} & f(m_{b-1}) \leq f_m \leq f(m_b) \\[6pt]
\dfrac{f(m_{b+1}) - f_m}{f(m_{b+1}) - f(m_b)} & f(m_b) \leq f_m \leq f(m_{b+1}) \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

The mel spectrogram is the energy of each band:

$$
M[b, t] = \sum_{m=0}^{N_k/2} H_b[m] \cdot |X_{N_k}[m,t]|^2
$$

compressed as $\log(1 + M[b,t])$, producing a $B \times T_{\mathrm{mel}}$ matrix.

The mel spectrogram contributes to the magnitude grid (weight $w_{\mathrm{mel}} = 0.18$). Its perceptual frequency axis gives relatively more weight to low-frequency content (where human hearing is most discriminating) compared to the linear-frequency STFT.

---

## 8. Feature Extraction: Chroma

### 8.1 Pitch-Class Folding

Chroma maps the STFT spectrum onto the 12 pitch classes of the equal-tempered chromatic scale (C, C#, D, …, B) by summing energy across all octaves. Each STFT bin $f_m$ is assigned to pitch class:

$$
p(m) = \left\lfloor 12 \log_2\!\left(\frac{f_m}{f_{\mathrm{ref}}}\right) \right\rfloor \bmod 12
$$

with $f_{\mathrm{ref}} = 261.63$ Hz (middle C, C4). Every octave of C maps to class 0, every octave of D to class 2, and so on. The chroma vector at frame $t$ is:

$$
C[p, t] = \sum_{\{m\,:\,p(m) = p\}} |X_{N_k}[m,t]|
$$

producing a $12 \times T_{\mathrm{chroma}}$ matrix.

Chroma is **octave-invariant**: a C major chord (C, E, G) in any register produces the same 12-dimensional chroma vector. It encodes tonal and harmonic content — the pitch-class distribution — independently of absolute pitch height. At weight $w_{\mathrm{chroma}} = 0.09$, its contribution introduces harmonic periodicity into the magnitude grid structure.

---

## 9. Feature Extraction: Mel-Frequency Cepstral Coefficients

### 9.1 The Cepstrum Concept

The cepstrum of a signal is the inverse Fourier transform of the log-spectrum. The motivation is the convolution model of sound production: a vocal source $s[n]$ convolved with a vocal tract impulse response $h[n]$ gives $x[n] = s[n] \star h[n]$; in the log-frequency domain this becomes $\log|X| = \log|S| + \log|H|$, an additive decomposition. The inverse transform separates the slow-varying spectral envelope $|H|$ from the fast-varying fine spectral structure $|S|$ by quefrency (the cepstral analog of frequency).

### 9.2 MFCC Definition

MFCCs are computed as the Discrete Cosine Transform type II (DCT-II) applied to the log-mel spectrum:

$$
\mathrm{cc}[c, t] = \sum_{b=0}^{B-1} \log M[b, t]\; \cos\!\left(\frac{\pi c}{B}\left(b + \frac{1}{2}\right)\right), \qquad c = 0, \ldots, C-1
$$

with $C = 20$ retained coefficients (default). The DCT-II is chosen over the DFT because: (1) the log-mel spectrum is a positive real sequence that the DCT treats as implicitly even-extended, making it the natural basis; (2) the DCT-II approximately diagonalizes the covariance matrix of mel spectra, making the MFCC coefficients approximately decorrelated — each coefficient carries nearly independent information.

**Interpretation of coefficients.** $\mathrm{cc}[0,t]$ is proportional to the log-energy of the frame (sum of all log-mel energies). Coefficients 1–13 encode the broad spectral envelope: vowel identity, instrument body resonance, spectral brightness. Coefficients 14–20 encode finer texture. Beyond $C = 20$, the added information per coefficient decays rapidly.

MFCCs contribute to the magnitude grid at weight $w_{\mathrm{MFCC}} = 0.09$ using absolute values $|\mathrm{cc}[c,t]|$, as the sign is not meaningful for an energy-based spatial magnitude.

---

## 10. Feature Extraction: Temporal Scalar Features

All scalar features below are computed with a fixed hop length $H = 256$ samples ($\approx 11.6$ ms at $f_s = 22{,}050$ Hz), producing a 1D time series over $T_{\mathrm{sc}} \approx L / H$ frames.

### 10.1 Root-Mean-Square Energy

$$
\mathrm{rms}[t] = \sqrt{\frac{1}{N_{\min}} \sum_{n = tH}^{tH + N_{\min} - 1} x[n]^2}
$$

Tracks instantaneous loudness. High during loud passages; near zero during silences. Contributes to the magnitude grid (weight $w_{\mathrm{RMS}} = 0.04$) as a spatial amplitude modulation.

### 10.2 Zero-Crossing Rate

$$
\mathrm{zcr}[t] = \frac{1}{2(N_{\min}-1)} \sum_{n=tH}^{tH + N_{\min} - 2} \bigl|\operatorname{sgn}(x[n+1]) - \operatorname{sgn}(x[n])\bigr|
$$

Each term is 1 when consecutive samples have opposite signs. High ZCR indicates noisy, fricative, or high-pitched content; low ZCR indicates tonal or bass-heavy content. Contributes to the phase grid (weight 0.05) as a small phase modulation.

### 10.3 Spectral Centroid

$$
\mu_f[t] = \frac{\sum_{m=0}^{N_k/2} f_m \cdot |X_{N_k}[m,t]|^2}{\sum_{m=0}^{N_k/2} |X_{N_k}[m,t]|^2}
$$

The power-weighted mean frequency. Strongly correlated with perceived **brightness**: cymbals and bright strings have high centroid; bass instruments have low centroid. Contributes to the phase grid (weight 0.10).

### 10.4 Additional Spectral Descriptors

**Spectral bandwidth** — standard deviation of the power spectrum around the centroid:

$$
\sigma_f[t] = \sqrt{\frac{\sum_{m} (f_m - \mu_f[t])^2 \cdot |X_{N_k}[m,t]|^2}{\sum_{m} |X_{N_k}[m,t]|^2}}
$$

Measures spectral spread: broad-band noise has high bandwidth; a pure tone has near-zero bandwidth.

**Spectral rolloff**: the frequency $f_r[t]$ below which 85 % of the total frame energy is concentrated. Summarizes the high-frequency content of the signal.

**Spectral flatness** (Wiener entropy):

$$
\mathrm{flat}[t] = \frac{\left(\prod_{m} |X_{N_k}[m,t]|\right)^{1/(N_k/2+1)}}{\frac{1}{N_k/2+1}\sum_{m} |X_{N_k}[m,t]|}
$$

Ratio of geometric mean to arithmetic mean of spectral magnitudes. Near 1 for white noise (spectrally flat); near 0 for a pure sinusoid.

### 10.5 Onset Strength

$$
\mathrm{onset}[t] = \frac{1}{B} \sum_{b=0}^{B-1} \max\!\bigl(0,\; \log M[b,t] - \log M[b,t-1]\bigr)
$$

The mean over mel bands of the half-wave rectified first-order difference of the log-mel spectrogram. Peaks sharply at note onsets, drum hits, and other sudden increases in spectral energy. Negative differences (energy releases) are discarded by the $\max(0,\cdot)$ rectification. Contributes to the phase grid (weight 0.15).

### 10.6 From 1D to 2D: Row Broadcasting

Scalar temporal features produce a single vector of $T_{\mathrm{sc}}$ time samples. To contribute to the $N \times N$ phase or magnitude grid:

1. Normalize to $[0,1]$: $\bar{r}[t] = (r[t] - \min r) / (\max r - \min r)$.
2. Interpolate to $N$ samples using 1D bicubic resampling.
3. Replicate across all $N$ rows: the result is an $N \times N$ array where every row is identical and the columns represent the temporal dimension.

This encoding maps time to the horizontal spatial axis: the leftmost column represents the beginning of the signal; the rightmost column represents the end.

---

## 11. From Features to 2D Grids: Interpolation and Band Slicing

### 11.1 Bicubic Spline Resampling

Feature matrices have varying shapes determined by $N_k$, $H$, $B$, $S$, and $L$, all different from the target $N \times N$. They are resampled using a 2D bicubic spline of degree 3 (`scipy.interpolate.RectBivariateSpline`). The source and destination grids are both normalized to $[0,1]^2$, making the operation scale-invariant:

$$
\hat{A}[i,j] = \hat{A}\!\left(\frac{i}{N-1},\;\frac{j}{N-1}\right), \qquad i, j = 0, \ldots, N-1
$$

Bicubic splines produce $C^1$-smooth results and avoid the ringing artifacts of sinc interpolation while being computationally lighter than higher-order splines. If the source has fewer than 4 points along an axis, the degree is reduced to $\min(3, \text{source size} - 1)$ to prevent overfitting.

### 11.2 Band Slicing for Color Mode

In **Colors** mode (and all modes derived from it), the normalized frequency axis $[0,1)$ of each feature is split into three bands before interpolation:

$$
\text{Low}: [0,\,\alpha) \qquad \text{Mid}: [\alpha,\,\beta) \qquad \text{High}: [\beta,\, 1)
$$

with defaults $\alpha = 1/3$, $\beta = 2/3$ (user-adjustable). For a matrix of shape $(K, T)$ along the frequency axis of size $K$, band $[\alpha_0, \alpha_1)$ maps to rows $[\lfloor \alpha_0 K \rfloor,\, \max(\lfloor \alpha_0 K \rfloor + 1,\, \lfloor \alpha_1 K \rfloor))$. Each band is independently interpolated to $N \times N$ and used to build a separate magnitude and phase grid. Three independent IFFT2 reconstructions produce three spatial channels stacked as R (Low), G (Mid), B (High).

This is spectral coloring: low-frequency audio energy (bass, fundamentals) drives the red channel; mid-frequency energy (harmonics, formants) drives the green channel; high-frequency energy (overtones, noise, sibilance) drives the blue channel.

---

## 12. Magnitude Grid Construction

The magnitude grid $\widetilde{M} \in [0,1]^{N \times N}$ is a weighted sum of six feature contributions, each independently normalized to $[0,1]$ and resampled to $N \times N$:

$$
\widetilde{M} = w_{\text{STFT}} \cdot \overline{M}_{\text{STFT}} + w_{\text{CWT}} \cdot \overline{M}_{\text{CWT}} + w_{\text{mel}} \cdot \overline{M}_{\text{mel}} + w_{\text{chr}} \cdot \overline{C} + w_{\text{mfcc}} \cdot \overline{|\mathrm{cc}|} + w_{\text{RMS}} \cdot \overline{E}
$$

where $\sum_i w_i = 1$ (weights are auto-normalized). The STFT contribution is the equal average over all active resolutions:

$$
\overline{M}_{\text{STFT}} = \frac{1}{|\mathcal{R}|} \sum_{N_k \in \mathcal{R}} \overline{\hat{M}_{N_k}}
$$

The final sum is again normalized to $[0,1]$.

Default weights and rationale:

| Feature | Default $w$ | Role in the magnitude grid |
|---|---|---|
| STFT (all resolutions combined) | 0.45 | Primary spectral representation; multi-resolution provides broad temporal–spectral coverage |
| Mel spectrogram | 0.18 | Perceptually weighted frequency axis; emphasizes the range most relevant to hearing |
| CWT magnitude | 0.15 | Constant-$Q$ resolution; adds multi-scale structure orthogonal to fixed-window STFT |
| Chroma | 0.09 | Tonal/harmonic content; octave-invariant pitch-class energy |
| MFCC (absolute values) | 0.09 | Spectral envelope; compact decorrelated timbral representation |
| RMS energy | 0.04 | Amplitude envelope; loudness dynamics as a spatially varying modulation |

---

## 13. Phase Grid Construction

### 13.1 Why Phase Dominates Visual Structure

A classical experiment in image processing demonstrates that swapping the phase spectra of two images (while keeping their magnitude spectra) produces outputs that look like the image whose phase was used, not the one whose magnitude was used. This is because phase encodes the spatial arrangement of structures — edges, corners, textures — while magnitude encodes only how frequently those spatial patterns occur. Constructing a structured phase grid from real audio features rather than random values is therefore the most important design decision for the visual quality of the output.

### 13.2 Phase Unwrapping

The raw STFT phase $\angle X_{N_k}[m,t]$ lies in $(-\pi,\pi]$ by the $\operatorname{atan2}$ definition. For a stationary sinusoid, the phase advances smoothly from frame to frame, but the $(-\pi,\pi]$ wrapping creates artificial $\pm 2\pi$ jumps wherever the true phase crosses $\pm\pi$. Phase unwrapping removes these jumps by accumulating the true phase differences. For a 2D phase array, unwrapping is applied first along the frequency axis, then along the time axis:

$$
\tilde{\phi}[m,t] = \phi[m,t] + 2\pi \sum_{\tau=1}^{t} \operatorname{round}\!\left(\frac{\phi[m,\tau] - \phi[m,\tau-1]}{2\pi}\right)
$$

After unwrapping, $\tilde{\phi}$ takes any real value and varies smoothly. Once interpolated to $N \times N$, this smooth variation produces spatially coherent phase patterns rather than the scattered values that would result from wrapped phase.

### 13.3 Phase Grid Assembly

Six contributions are blended:

$$
\widetilde{\Phi} = w_{\text{mid}} \cdot \Phi_{1024} + w_{\text{fine}} \cdot \Phi_{512} + w_{\text{cwt}} \cdot \Phi_{\text{CWT}} + w_{\text{onset}} \cdot \Phi_{\text{onset}} + w_{\text{cen}} \cdot \Phi_{\text{centroid}} + w_{\text{zcr}} \cdot \Phi_{\text{ZCR}}
$$

**STFT phase grids.** Unwrapped phase from $N_k = 1024$ (balanced time-frequency resolution, weight 0.30) and $N_k = 512$ (finer time resolution for fast transients, weight 0.20) are interpolated to $N \times N$.

**CWT phase grid.** For the Morlet wavelet, the unwrapped instantaneous phase $\angle W[s,\tau]$ (shape $S \times L'$, where $L'$ is the CWT waveform length) is interpolated to $N \times N$ at weight 0.20. This provides multi-scale phase information at constant-$Q$ resolution, complementary to the fixed-window STFT phases. For the Ricker wavelet (real, no analytic signal), this weight is redistributed proportionally to $w_{\text{mid}}$ and $w_{\text{fine}}$.

**Temporal feature phase grids.** Scalar temporal features are broadcast to $N \times N$ (Section 10.6) and scaled to sub-intervals of $(-\pi,\pi]$:

| Source | Phase scale | Effect |
|---|---|---|
| Onset strength $\mathrm{onset}[t]$ | $\times\, \pi/2$ | Rhythmic events perturb phase by up to $\pi/2$ |
| Spectral centroid $\mu_f[t]$ | $\times\, \pi$ | Brightness variation sweeps up to one full phase cycle |
| ZCR $\mathrm{zcr}[t]$ | $\times\, \pi/4$ | Noisiness contributes a small phase modulation |

The scaling constants are design choices that give each temporal feature a calibrated influence: onset (the most structured rhythm signal) is allowed the largest range; ZCR the smallest.

### 13.4 Final Wrapping

After summation, $\widetilde{\Phi}$ is wrapped back to $(-\pi,\pi]$:

$$
\widetilde{\Phi} \leftarrow \bigl(\widetilde{\Phi} + \pi\bigr) \bmod 2\pi - \pi
$$

---

## 14. Hermitian Symmetry Enforcement

### 14.1 Necessity

The 2D IDFT of an arbitrary complex $N \times N$ matrix $Z$ is generally complex. A real-valued image requires the input to satisfy the **Hermitian symmetry** condition:

$$
F[u,v] = \overline{F[(-u)\bmod N,\; (-v)\bmod N]} \qquad \forall\, u, v
$$

The assembled spectrum $Z = \widetilde{M} \cdot e^{j\widetilde{\Phi}}$ does not satisfy this in general, because $\widetilde{M}$ and $\widetilde{\Phi}$ are constructed independently from audio features with no constraints linking $(u,v)$ to $(-u,-v)$. Without correction, the imaginary part of $\mathrm{IFFT2}(Z)$ would be non-negligible, and discarding it would introduce an uncontrolled implicit modification of the spectrum.

### 14.2 Orthogonal Projection

The set of Hermitian-symmetric $N \times N$ complex matrices is a subspace. The unique nearest Hermitian-symmetric matrix to $Z$ in the Frobenius norm is:

$$
Z_{\mathrm{sym}}[u,v] = \frac{Z[u,v] + \overline{Z[(-u)\bmod N,\; (-v)\bmod N]}}{2}
$$

This is an orthogonal projection: $(Z_{\mathrm{sym}})_{\mathrm{sym}} = Z_{\mathrm{sym}}$, and the discarded component $Z - Z_{\mathrm{sym}}$ is anti-Hermitian and orthogonal to all Hermitian matrices in the Frobenius inner product.

### 14.3 Implementation

The map $Z[u,v] \mapsto Z[(-u)\bmod N,\, (-v)\bmod N]$ is implemented in three steps:

1. $Z_{\mathrm{flip}} = Z[::-1,\, ::-1]$ — flip both axes, giving $Z_{\mathrm{flip}}[u,v] = Z[N{-}1{-}u,\, N{-}1{-}v]$.
2. $Z_r = \operatorname{roll}(\operatorname{roll}(Z_{\mathrm{flip}},\, 1,\, \text{axis}=0),\, 1,\, \text{axis}=1)$ — circular shift by 1 in both axes, giving $Z_r[u,v] = Z[(-u)\bmod N,\, (-v)\bmod N]$.
3. $Z_{\mathrm{sym}} = (Z + \overline{Z_r}) / 2$.

**DC verification.** At $(u,v)=(0,0)$: $Z_r[0,0] = Z[0,0]$, so $Z_{\mathrm{sym}}[0,0] = (Z[0,0] + \overline{Z[0,0]})/2 = \operatorname{Re}(Z[0,0]) \in \mathbb{R}$. The DC coefficient is always real, ensuring the spatial mean of the reconstructed image is a real number.

After symmetrization, $\bigl|\operatorname{Im}(\mathrm{IFFT2}(Z_{\mathrm{sym}}))\bigr| \sim 10^{-14}$ (floating-point rounding only) and is discarded.

---

## 15. Image Reconstruction via 2D Inverse DFT

### 15.1 Complex Spectrum Assembly

The complex coefficient matrix is assembled element-wise:

$$
Z[u, v] = \widetilde{M}[u, v] \cdot e^{j\,\widetilde{\Phi}[u,v]}
$$

This encodes $\widetilde{M}[u,v]$ as the amplitude and $\widetilde{\Phi}[u,v]$ as the phase angle of spatial frequency component $(u,v)$.

### 15.2 2D Inverse DFT

After Hermitian symmetrization:

$$
f[x, y] = \frac{1}{N^2} \sum_{u=0}^{N-1} \sum_{v=0}^{N-1} Z_{\mathrm{sym}}[u,v]\; e^{j2\pi(ux + vy)/N}
$$

Only $\operatorname{Re}(f[x,y])$ is retained. The output is a floating-point $N \times N$ image encoding the spatial interference pattern of all the sinusoidal components defined by $Z_{\mathrm{sym}}$.

### 15.3 Normalization Timing

In the **non-sectioned** path ($k=1$), $f$ is normalized to $[0,1]$ by global min–max immediately after the IFFT2 before output mode post-processing. In the **sectioned** path ($k>1$), each section patch retains its raw floating-point values; a single global percentile-based normalization is applied to the assembled canvas after all patches are placed (Section 17.4), ensuring a uniform global scale across all sections.

---

## 16. Output Modes

Each mode applies a different transformation to the floating-point image. The five modes form a hierarchy: **Colors** is the primary color image; **Black mix**, **Luma mix**, and **Watershed** each derive from a combination of Colors and Grayscale images.

### 16.1 Grayscale

A single pair $(\widetilde{M}, \widetilde{\Phi})$ is built from all features at full bandwidth. The IFFT2 produces one real channel $f$, replicated into R = G = B. This is the direct, uncolored output of the spectral synthesis pipeline.

### 16.2 Colors

The frequency axis is split into three bands (Section 11.2): Low → R, Mid → G, High → B. For each band $c$:

1. Feature matrices are sliced to the band's frequency rows.
2. $(\widetilde{M}^{(c)}, \widetilde{\Phi}^{(c)})$ are built from the sliced features.
3. IFFT2 produces channel $f^{(c)}$.

Stacking the three channels gives an RGB image where spectral coloring makes frequency content directly visible as color.

### 16.3 Black Mix

Both Colors and Grayscale images are computed in full. The grayscale channel $g \in [0,1]^{N \times N}$ is optionally Gaussian-smoothed ($\sigma_{\mathrm{ink}}$) and then binarized with **Otsu's threshold**.

**Otsu's method.** From the 256-bin normalized histogram $p_i$ of $g$, the threshold $\theta^*$ maximizes the between-class variance:

$$
\sigma_B^2(\theta) = \omega_0(\theta)\,\omega_1(\theta)\,\bigl[\mu_0(\theta) - \mu_1(\theta)\bigr]^2
$$

where $\omega_0 = \sum_{i \leq \theta} p_i$ is the probability mass of the dark class, $\omega_1 = 1 - \omega_0$ of the bright class, and $\mu_0, \mu_1$ are the class means. Maximizing between-class variance minimizes within-class variance, giving the threshold that best separates a bimodal distribution.

The user selects which class (dark, bright, or automatic minority) forms the drawing mask. Within that class, the top $d\%$ most extreme pixels are kept (darkest of the dark class or brightest of the bright class). Optional morphological dilation by $t$ iterations thickens the marks. Selected pixels are set to black in the Colors image; all others are unchanged.

### 16.4 Luma Mix

The Colors image $I_{\mathrm{color}}$ is modulated channel-wise by a luminance coefficient map derived from the Grayscale image:

$$
I_{\mathrm{out}}[x,y,c] = I_{\mathrm{color}}[x,y,c] \cdot \alpha_{\mathrm{eff}}[x,y]
$$

The coefficient map is computed as follows from the grayscale luminance $g = 0.299R + 0.587G + 0.114B$ of the Grayscale image:

1. Normalize: $g \leftarrow \overline{g} \in [0,1]$.
2. Optional Gaussian blur with $\sigma_{\mathrm{luma}}$.
3. Re-normalize after blur.
4. Gamma correction: $\alpha \leftarrow \alpha^{\gamma_\alpha}$.
5. Clamp minimum and mix: $\alpha_{\mathrm{eff}} = (1-\lambda) + \lambda \cdot \bigl[\alpha_{\min} + (1-\alpha_{\min})\cdot\alpha\bigr]$

where $\lambda \in [0,1]$ is the mixing strength and $\alpha_{\min} \in [0,1]$ is the minimum allowed coefficient. At $\lambda=0$, the output is $I_{\mathrm{color}}$ unchanged; at $\lambda=1$, $\alpha_{\mathrm{eff}}$ ranges from $\alpha_{\min}$ (darkest grayscale regions) to 1 (brightest grayscale regions). The effect is a multiplicative spatial modulation of the color image by the structure of the grayscale image.

### 16.5 Watershed

The watershed transform segments the image into irregular regions bounded by gradient ridges. It is applied to the Luma mix image.

**Step 1 — Gradient.** The Luma mix image is converted to grayscale (BT.601 luminance) and Gaussian-smoothed with $\sigma_w$. The Sobel gradient magnitude:

$$
g[x,y] = \sqrt{g_x[x,y]^2 + g_y[x,y]^2}
$$

is normalized to $[0,1]$. Large values mark edges; small values mark flat interior regions.

**Step 2 — Marker seeding.** The image is partitioned into a regular grid of non-overlapping $d \times d$ cells. Within each cell, the pixel with the smallest gradient value is selected as a seed marker and assigned a unique integer label. Placing markers at gradient minima ensures they land in flat regions away from edges.

**Step 3 — Minimax flood.** A priority-queue (min-heap) expansion floods all markers simultaneously. At each step, the lowest-cost unlabeled pixel adjacent to any labeled pixel is assigned the label of its labeled neighbor. Cost is the minimax path cost:

$$
\mathrm{cost}(p) = \max\bigl(\mathrm{cost}(\mathrm{parent}(p)),\; g[p]\bigr)
$$

This criterion makes the flood stop at gradient ridges: two markers separated by a high-gradient ridge accumulate a high cost at the boundary and do not easily cross. The result is a labeling where each region is bounded by locally optimal edge contours.

**Step 4 — Region coloring.** Each region is assigned a representative color from the Luma mix image: a randomly selected pixel, the per-channel mean, or the per-channel median (user choice).

**Step 5 — Boundary rendering.** Optional: boundary pixels (those adjacent to a pixel of a different label, optionally dilated by $t$ iterations) are set to black or replaced by the local mean color in a $w \times w$ window.

---

## 17. Sectioned Image Synthesis

### 17.1 Motivation

A single image from the full waveform integrates spectral analysis over the entire duration: transient events at the beginning and the sustained character at the end are blended into one static image. Splitting the waveform into $k$ chronological sections and generating one patch per section creates a spatial mosaic encoding temporal evolution: each patch reflects the spectral character of its time segment.

### 17.2 Waveform Splitting

The waveform is divided into $k$ contiguous, non-overlapping segments with near-equal lengths:

$$
x_i[n] = x\!\left[\left\lfloor \frac{iL}{k} \right\rfloor + n\right], \qquad n = 0,\ldots, L_i - 1, \qquad L_i = \left\lfloor \frac{(i+1)L}{k} \right\rfloor - \left\lfloor \frac{iL}{k} \right\rfloor
$$

Integer arithmetic ensures $\sum_{i=0}^{k-1} L_i = L$ exactly without rounding residuals.

### 17.3 Independent Patch Generation

Each section $x_i$ is processed through the complete feature extraction and IFFT2 pipeline independently, producing a raw floating-point patch $P_i$ of shape $N \times N \times 3$. No per-patch normalization is applied; patches are kept in their raw floating-point range so the subsequent global normalization operates on the full dynamic range of the assembled canvas.

### 17.4 Global Normalization After Assembly

After all patches are placed on the canvas, a single robust normalization is applied to the entire canvas simultaneously:

$$
\hat{f}[x,y,c] = \operatorname{clip}\!\left(\frac{f[x,y,c] - q_{p_1}}{q_{p_2} - q_{p_1}},\; 0,\; 1\right)
$$

where $q_{p_1}$ and $q_{p_2}$ are the $p_1$-th and $p_2$-th percentiles of all values across the canvas (defaults: $p_1 = 1\%$, $p_2 = 99\%$). Using percentiles rather than global min/max provides robustness against isolated extreme values. This single normalization ensures a consistent brightness scale across all sections and avoids the sharp intensity discontinuities that per-patch normalization would produce.

### 17.5 Minimum Section Length Constraint

Each section must contain at least $L_{\min} = 16{,}384$ samples for all features to be meaningful: the largest default STFT window ($N_k = 8192$) requires $N_k$ samples; the CWT needs enough samples to span one full period of its lowest-frequency scale; MFCC and mel require at least several frames. The maximum section count is:

$$
k_{\max} = \min\!\left(\left\lfloor \frac{L}{L_{\min}} \right\rfloor,\; \left\lfloor \frac{N}{N_{\min,\mathrm{block}}} \right\rfloor^2,\; 64\right)
$$

where $N_{\min,\mathrm{block}} = 32$ px is the minimum readable block side in the assembled image, and 64 is the interface upper bound.

---

## 18. Section Layout Algorithms

Each layout assigns every pixel of the $N \times N$ canvas to one of the $k$ sections. Except for the Chronological treemap, layouts are implemented as dense integer index maps: an $N \times N$ array whose value at each pixel is the section index assigned to that pixel. Section 0 always corresponds to the beginning of the audio; section $k-1$ to the end.

### 18.1 None

$k = 1$. The full waveform generates one $N \times N$ image with no sectioning.

### 18.2 Chronological Treemap

The canvas is recursively subdivided into rectangles using a binary slice-and-dice strategy. At each recursion level, the current rectangle of width $W$ and height $H$ is split along its longest side (horizontally if $W \geq H$, vertically otherwise; a square is split vertically by convention). The split position is proportional to the section count assigned to each half:

$$
W_1 = \left\lfloor W \cdot \frac{\lceil k/2 \rceil}{k} \right\rfloor, \qquad W_2 = W - W_1
$$

The first $\lceil k/2 \rceil$ sections fill the left (or top) sub-rectangle; the remaining $\lfloor k/2 \rfloor$ fill the right (or bottom). The recursion terminates when a rectangle is assigned exactly one section. All leaf rectangles have nearly equal area, so each section contributes a comparable number of pixels regardless of $k$.

For the treemap, each patch is generated at size $\max(W_{\mathrm{rect}}, H_{\mathrm{rect}})$ (the longer side of the target rectangle) and then bicubic-resized and center-cropped to exactly fit its rectangle.

### 18.3 Clockwise Circular Slices

The canvas is divided into $k$ equal angular sectors around the center pixel $c = (N-1)/2$. The clockwise polar angle measured from the upward vertical direction:

$$
\theta(x,y) = \operatorname{atan2}(x - c,\; c - y) \bmod 2\pi
$$

Section index: $i = \lfloor k \cdot \theta / (2\pi) \rfloor$, clamped to $[0, k-1]$. Each patch is generated at size $N/2$ (to reduce computation) and then resized to $N \times N$ before masking onto its sector.

### 18.4 Concentric Circles

The Euclidean distance from the center:

$$
r(x,y) = \sqrt{(x-c)^2 + (y-c)^2}
$$

Section index: $i = \operatorname{clip}(\lfloor k \cdot r / r_{\max} \rfloor, 0, k-1)$ where $r_{\max}$ is the maximum distance in the canvas. The square-root implicit in the index formula ensures equal area per annulus: each section contributes $\pi r_{\max}^2 / k$ pixels.

### 18.5 Concentric Squares

The Chebyshev distance ($\ell^\infty$ norm) from the center:

$$
r_\infty(x,y) = \max(|x-c|,\; |y-c|)
$$

Section index: $i = \operatorname{clip}(\lfloor k \cdot r_\infty / r_{\infty,\max} \rfloor, 0, k-1)$. This partitions the canvas into nested square frames; section 0 occupies the innermost square (audio beginning) and section $k-1$ the outermost frame (audio end).

### 18.6 Vertical and Horizontal Strips

**Vertical strips:** $i = \operatorname{clip}(\lfloor k \cdot x / N \rfloor, 0, k-1)$. Time flows left to right.

**Horizontal strips:** $i = \operatorname{clip}(\lfloor k \cdot y / N \rfloor, 0, k-1)$. Time flows top to bottom.

---

## 19. Post-Processing Pipeline

After the canvas is assembled and globally normalized to $[0,1]$, the following operations are applied in this order. All operations are in floating-point; the result is clipped to $[0,1]$ before conversion to `uint8` $\in [0,255]$.

**1. Per-channel RGB balance** (applied to the unnormalized floating-point canvas before global normalization):

$$
I^{(c)} \leftarrow g_c \cdot I^{(c)}, \qquad c \in \{R,G,B\}, \quad g_c \in [0, 3]
$$

Independent channel gains allow shifting the color balance.

**2. Robust global normalization:**

$$
\hat{I} \leftarrow \operatorname{clip}\!\left(\frac{I - q_{p_1}}{q_{p_2} - q_{p_1}},\; 0,\; 1\right)
$$

Percentiles $p_1 = 1\%$, $p_2 = 99\%$ by default (user-adjustable). Percentiles instead of global min/max provide robustness against isolated extremes.

**3. Linear contrast adjustment** centered at 0.5:

$$
\hat{I} \leftarrow \operatorname{clip}\!\bigl(0.5 + c_s(\hat{I} - 0.5),\; 0,\; 1\bigr), \qquad c_s \in [0.2, 3.0]
$$

$c_s > 1$: values move away from 0.5 (higher contrast). $c_s < 1$: values compress toward 0.5 (lower contrast).

**4. Brightness scaling:**

$$
\hat{I} \leftarrow \operatorname{clip}(b \cdot \hat{I},\; 0,\; 1), \qquad b \in [0.2, 2.5]
$$

**5. Gamma correction** (power-law transfer function):

$$
\hat{I} \leftarrow \hat{I}^{\gamma}, \qquad \gamma \in (0, 2.5]
$$

$\gamma < 1$ brightens (expands shadows, compresses highlights); $\gamma > 1$ darkens. The default $\gamma = 0.85$ partially compensates for the underexposure tendency of log-compressed spectral features, which cluster near zero for quiet signals. The human visual system applies an analogous gamma correction to perceived brightness, making this a natural calibration step.

**6. Saturation scaling** (RGB images only):

$$
Y = 0.299\,R + 0.587\,G + 0.114\,B \quad \text{(BT.601 luminance)}
$$

$$
I'^{(c)} \leftarrow \operatorname{clip}\!\bigl(Y + s_{\mathrm{sat}} \cdot (I^{(c)} - Y),\; 0,\; 1\bigr), \qquad s_{\mathrm{sat}} \in [0, 3]
$$

Linear interpolation (or extrapolation) between the grayscale luminance ($s_{\mathrm{sat}}=0$) and the full-color image ($s_{\mathrm{sat}}=1$), extended to chromatic oversaturation ($s_{\mathrm{sat}}>1$).

---

## 20. Parameter Reference

| Parameter | Default | Range | Section |
|---|---|---|---|
| Output image size $N$ (px) | 512 | 64–1024, step 16 | §15 |
| Output mode | Watershed | Grayscale / Colors / Black mix / Luma mix / Watershed | §16 |
| Section layout | None | None + 6 geometric layouts | §18 |
| Number of sections $k$ | 32 | 1–$k_{\max}$ | §17 |
| CWT wavelet | Morlet | Morlet / Ricker | §6 |
| STFT minimum $N_k$ | 256 | {256, 512, 1024, 2048, 4096, 8192} | §5.3 |
| STFT maximum $N_k$ | 8192 | {256, 512, 1024, 2048, 4096, 8192} | §5.3 |
| CWT scales $S$ | 64 | 16–128 | §6.5 |
| CWT max samples $L_{\mathrm{CWT}}$ | 44100 | 4096–220500 | §6.5 |
| Mel bands $B$ | 128 | 32–256 | §7.2 |
| MFCC coefficients $C$ | 20 | 8–64 | §9.2 |
| Magnitude weights $w_i$ | See §12 | 0–1 each, auto-normalized | §12 |
| Phase weights $w_i$ | See §13 | 0–1 each, auto-normalized | §13 |
| Low→mid split $\alpha$ | 1/3 | 0.10–0.45 | §11.2 |
| Mid→high split $\beta$ | 2/3 | 0.55–0.90 | §11.2 |
| RGB normalization mode | Per-channel | Per-channel / Shared | §19 |
| RGB channel balance $g_c$ | 1.0 each | 0–3 | §19 |
| Normalization lower percentile $p_1$ | 1 % | 0–10 % | §19 |
| Normalization upper percentile $p_2$ | 99 % | 90–100 % | §19 |
| Gamma $\gamma$ | 0.85 | 0.20–2.50 | §19 |
| Contrast $c_s$ | 1.0 | 0.20–3.0 | §19 |
| Brightness $b$ | 1.0 | 0.20–2.5 | §19 |
| Saturation $s_{\mathrm{sat}}$ | 1.0 | 0–3 | §19 |
| Otsu class | Auto minority | Auto minority / Dark / Bright | §16.3 |
| Black pixel density $d$ (%) | 50 | 1–100 | §16.3 |
| Pre-smoothing $\sigma_{\mathrm{ink}}$ | 0 | 0–10 | §16.3 |
| Black line thickness (px) | 0 | 0–8 | §16.3 |
| Luma mixing strength $\lambda$ | 1.0 | 0–1 | §16.4 |
| Luma minimum coefficient $\alpha_{\min}$ | 0 | 0–1 | §16.4 |
| Luma coefficient gamma $\gamma_\alpha$ | 1.0 | 0.20–3.0 | §16.4 |
| Luma coefficient blur $\sigma_{\mathrm{luma}}$ | 0 | 0–12 | §16.4 |
| Watershed marker spacing $d$ (px) | 36 | 4–160 | §16.5 |
| Watershed gradient smoothing $\sigma_w$ | 1.3 | 0–8 | §16.5 |
| Watershed region color mode | Random pixel | Random pixel / Mean / Median | §16.5 |
| Watershed random seed | 12345 | 0–999999 | §16.5 |
| Watershed boundary style | None | None / Black / Local mean | §16.5 |
| Watershed boundary thickness (px) | 0 | 0–8 | §16.5 |
| Watershed boundary mean window (px) | 5 | 3–21 | §16.5 |

---

## 21. Limitations and Interpretation

**Magnitude–phase decoupling.** The magnitude and phase grids are sourced from different audio features and blended with independent weight sets. Their combination into a single complex spectrum does not correspond to the DFT of any naturally occurring image. The IFFT2 always produces a valid spatial image, but its statistical properties (spatial frequency distribution, texture isotropy, edge statistics) are determined by the feature combination rather than by a principled image prior. The output is not a "picture of the sound" in any perceptually grounded sense; it is a deterministic visual encoding of the chosen spectral features.

**Phase dominates appearance.** Because phase encodes spatial structure (Section 4.3), the choice of phase-grid weights and sources has a far larger impact on the visual output than the choice of magnitude-grid weights. Experiments varying only magnitude weights produce subtle textural changes; varying only phase weights produce drastic changes in the spatial arrangement and character of the image. Users seeking to explore the parameter space should start with the phase weights.

**Spatial isotropy vs. audio anisotropy.** Audio features are inherently asymmetric: the time axis differs fundamentally from the frequency axis (they have different physical units and semantics); the STFT has linear frequency spacing; the CWT has logarithmic scale spacing. The assembled 2D Fourier grid is symmetric in $u$ and $v$ — horizontal and vertical spatial frequencies are treated identically. There is therefore no principled directional correspondence between the audio's time or frequency axes and any specific spatial direction in the output image. The temporal evolution of the audio is encoded in the phase patterns and, in sectioned mode, in the spatial layout of patches; but not through a direct Cartesian mapping.

**Section boundary continuity.** Even with global normalization, adjacent section patches may have substantially different spectral statistics, producing visible luminance or chromatic discontinuities at their boundaries. This is not an artifact but an accurate reflection of genuine changes in the audio's spectral character. In temporal layouts (strips, treemap), such discontinuities encode meaningful transitions: silence-to-attack, change of instrument, structural section change. Whether they are desirable is aesthetic.

**CPU computation cost.** Dominant cost terms:
- Multi-resolution STFT: $O(|\mathcal{R}| \cdot L \log N_k)$ — scales linearly with number of resolutions and signal length.
- CWT: $O(L_{\mathrm{CWT}} \cdot S \cdot s_{\max})$ — the $L_{\mathrm{CWT}}$ truncation is the primary mitigation; the default value (44 100 samples) keeps it tractable.
- Sectioning: the full pipeline runs once per section; $k$ sections costs $k$ times the single-pass time. Composite modes (Black mix, Luma mix, Watershed) run the pipeline twice (once for Colors, once for Grayscale), doubling the compute time.
- Watershed: $O(N^2 \log N^2)$ for the heap flood — negligible at $N \leq 512$.

For interactive exploration, $k \leq 8$ sections and $N \leq 512$ are recommended on typical CPU hardware.

---
