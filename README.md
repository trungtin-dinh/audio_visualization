---
title: Audio Visualization
emoji:
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.36.0
app_file: app_sl.py
pinned: false
license: mit
short_description: Turn audio into images with inverse spectral synthesis
---

# Audio Visualization

This repository contains an interactive Streamlit mini app that converts an audio signal into a square image using classical signal-processing operations.

The app is designed as an educational and portfolio demo at the intersection of audio analysis, Fourier-domain synthesis, image generation and segmentation. It does not use a trained model. Instead, it extracts deterministic audio descriptors, maps them to the magnitude and phase of a synthetic 2D Fourier spectrum, applies Hermitian symmetry, and reconstructs an image with a 2D inverse Fourier transform.

The main idea is to treat the audio signal as a source of spectral and temporal structure, then use that structure to build an image:

```text
Input audio
|
+-- Audio feature extraction
|   +-- multi-resolution STFT
|   +-- continuous wavelet transform
|   +-- mel spectrogram
|   +-- chroma
|   +-- MFCC
|   +-- temporal descriptors
|
+-- 2D Fourier spectrum construction
|   +-- magnitude grid
|   +-- phase grid
|   +-- Hermitian symmetrization
|
+-- 2D inverse Fourier transform
|
+-- Output rendering
    +-- grayscale image
    +-- RGB spectral image
    +-- black drawing overlay
    +-- luminance-modulated color image
    +-- watershed-style segmented image
```

When the parameters are fixed, the same audio always produces the same image.

## Main features

- Load the default open-source audio sample.
- Upload a personal audio file.
- Record audio directly from the browser when supported by Streamlit.
- Display a waveform preview of the loaded signal.
- Generate a square image from audio features using inverse 2D spectral synthesis.
- Use multi-resolution STFT, CWT, mel spectrogram, chroma, MFCC, RMS, onset strength, spectral centroid and zero-crossing rate.
- Build separate 2D Fourier magnitude and phase grids from audio descriptors.
- Enforce Hermitian symmetry before applying `ifft2`, so the reconstructed image is real-valued.
- Choose several output modes: `Grayscale`, `Colors`, `Black mix`, `Luma mix` and `Watershed`.
- Split the audio into chronological sections and arrange them with several spatial layouts.
- Adjust advanced synthesis, rendering, RGB, segmentation and post-processing parameters.
- Export the generated image as PNG.
- Read the English and French documentation tabs.

## Method overview

The application is based on the following spectral synthesis model:

```text
x[n]
|
+-- feature extraction
|
+-- grid construction
|   +-- M_tilde[u, v]      magnitude grid
|   +-- Phi_tilde[u, v]    phase grid
|
+-- complex spectrum
|   Z[u, v] = M_tilde[u, v] exp(j Phi_tilde[u, v])
|
+-- Hermitian projection
|   Z_sym[u, v] = (Z[u, v] + conj(Z[-u, -v])) / 2
|
+-- image reconstruction
    f[x, y] = Re(IFFT2(Z_sym))
```

The magnitude grid controls how much each spatial sinusoidal component contributes to the output image. The phase grid controls where those components are positioned. Since phase is crucial for spatial structure, the app gives direct control over the phase feature weights.

## Audio feature extraction

The input waveform is decoded as mono audio and processed at its original sample rate. The feature extraction stage combines several complementary representations.

### Multi-resolution STFT

Several STFT window lengths are used at the same time. Short windows capture transients and fast events, while long windows capture more precise harmonic frequency structure.

The STFT coefficient is:

```text
X_Nk[m,t] = sum_n x[n + tH_k] w[n] exp(-j 2 pi m n / N_k)
```

The magnitude of the STFT contributes to the Fourier magnitude grid. The unwrapped phase of selected STFT resolutions contributes to the Fourier phase grid.

### Continuous Wavelet Transform

The CWT adds a constant-Q representation. It gives finer temporal resolution at high frequencies and finer frequency resolution at low frequencies.

The app supports:

- `Morlet`, which provides both CWT magnitude and instantaneous phase;
- `Ricker (Mexican hat)`, which provides magnitude only.

### Perceptual and temporal descriptors

The app also uses:

- mel spectrogram for perceptually weighted spectral energy;
- chroma for pitch-class and harmonic content;
- MFCC for spectral envelope information;
- RMS for loudness dynamics;
- spectral centroid for brightness;
- onset strength for rhythmic and attack information;
- zero-crossing rate for noisiness and high-frequency temporal changes.

These descriptors are normalized, interpolated and mixed into the final 2D grids.

## Building the 2D Fourier spectrum

All feature matrices have different shapes because they depend on the signal length, window size, number of scales and hop length. Each feature map is interpolated to the output image size `N x N`.

The magnitude grid is a weighted sum of energy-related descriptors:

```text
M_tilde =
    w_STFT   M_STFT
  + w_CWT    M_CWT
  + w_mel    M_mel
  + w_chr    C
  + w_mfcc   |MFCC|
  + w_RMS    E
```

The phase grid is a weighted sum of phase-related and temporal descriptors:

```text
Phi_tilde =
    w_mid       Phi_1024
  + w_fine      Phi_512
  + w_cwt       Phi_CWT
  + w_onset     Phi_onset
  + w_centroid  Phi_centroid
  + w_zcr       Phi_ZCR
```

All weights are user-adjustable and automatically normalized.

## Hermitian symmetry and image reconstruction

A real-valued image has a Fourier spectrum with Hermitian symmetry:

```text
F[u,v] = conj(F[-u mod N, -v mod N])
```

The audio-derived spectrum does not naturally satisfy this constraint, so the app projects it onto the Hermitian-symmetric subspace:

```text
Z_sym[u,v] = (Z[u,v] + conj(Z[-u mod N, -v mod N])) / 2
```

Then the image is reconstructed with:

```text
f[x,y] = Re(IFFT2(Z_sym))
```

The imaginary part after this operation is only numerical floating-point residue and is discarded.

## Output modes

### Grayscale

A single full-band magnitude and phase pair is constructed. The reconstructed channel is repeated over R, G and B.

### Colors

The frequency axis is split into three bands:

```text
R channel -> low-frequency band
G channel -> mid-frequency band
B channel -> high-frequency band
```

Each band is reconstructed independently and then stacked into an RGB image.

### Black mix

The app computes both `Colors` and `Grayscale`. The grayscale image is binarized with Otsu thresholding, and the selected class is used as a sparse black drawing mask over the color image.

### Luma mix

The app computes both `Colors` and `Grayscale`. The grayscale image becomes a luminance coefficient map that multiplicatively modulates the RGB image:

```text
I_out[x,y,c] = I_color[x,y,c] * alpha[x,y]
```

This mode keeps the color texture while using the grayscale reconstruction as a spatial envelope.

### Watershed

The `Luma mix` result is segmented into connected regions. Each region receives a representative color, and the boundary style can be adjusted or disabled.

## Sectioned synthesis

The app can process the whole audio at once, or divide it into chronological sections. If `k` sections are selected, the waveform is split into `k` nearly equal segments. Each segment generates its own image patch, and all patches are assembled into a final square canvas.

The available section layouts are:

- `None`;
- `Chronological treemap`;
- `Clockwise circular slices`;
- `Concentric circles`;
- `Concentric squares`;
- `Vertical strips`;
- `Horizontal strips`.

This makes it possible to visualize temporal evolution in the final image. The first section always corresponds to the beginning of the audio, and the last section corresponds to the end.

## Post-processing and parameters

The interface exposes a large set of parameters for detailed experimentation:

- output image size;
- output mode;
- number of sections;
- section layout;
- CWT wavelet type;
- robust normalization percentiles;
- contrast, brightness, gamma and saturation;
- RGB band split and channel balance;
- Black mix thresholding and mask density;
- Luma mix strength and coefficient shaping;
- Watershed marker spacing, boundary style and region coloring;
- magnitude feature weights;
- phase feature weights;
- STFT, CWT, mel and MFCC analysis parameters.

The default parameters are chosen to produce a stable image without requiring the user to understand all advanced settings. However, exposing these controls makes the app useful as a small experimental lab for audio-driven image synthesis.

## Repository structure

```text
.
├── app_sl.py              # Streamlit app
├── documentation_en.md    # English documentation
├── documentation_fr.md    # French documentation
├── requirements.txt       # Python dependencies
├── packages.txt           # System dependency for audio decoding
├── LICENSE.txt            # License file
└── README.md              # Repository and app description
```

## Installation

Clone the repository:

```bash
git clone https://github.com/trungtin-dinh/audio_visualization.git
cd audio_visualization
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The repository uses:

```text
streamlit
numpy
scipy
librosa
soundfile
matplotlib
scikit-image
```

The `packages.txt` file includes:

```text
libsndfile1
```

This system package is useful for audio file decoding on Streamlit Community Cloud and similar Linux-based deployments.

## Run the Streamlit app locally

```bash
streamlit run app_sl.py
```

The local interface will usually be available at:

```text
http://localhost:8501
```

## Deployment notes

This repository is configured as a Streamlit app through the Hugging Face / Spaces-style metadata at the top of this README:

```yaml
sdk: streamlit
app_file: app_sl.py
```

For Streamlit Community Cloud, select `app_sl.py` as the main file.

## Documentation

The repository includes two Markdown documentation files:

- `documentation_en.md` for the English documentation;
- `documentation_fr.md` for the French documentation.

These files explain the audio feature extraction, 2D Fourier spectrum construction, Hermitian symmetry, output modes, segmentation methods, sectioned synthesis, layout algorithms, post-processing pipeline, parameters and limitations.

## Notes and limitations

This app is an educational and artistic visualization system. It does not recognize speech, instruments or musical semantics. It only converts measurable audio descriptors into an image through deterministic signal-processing operations.

The generated spectrum is not the Fourier transform of a natural image. It is a synthetic spectrum built from audio features. The resulting image is therefore not expected to look like a photograph. Its visual structure comes from the interference of spatial sinusoids whose magnitude and phase were derived from the audio.

The computational cost increases with:

- the output image size;
- the number of sections;
- the CWT maximum sample count;
- the number of CWT scales;
- composite modes such as `Black mix`, `Luma mix` and `Watershed`, which run the reconstruction pipeline twice.

For online use, moderate image sizes and section counts are recommended.

## License

This project is released under the MIT License.

## Author

Developed by Trung-Tin Dinh as part of a portfolio of interactive signal, audio, image and computer vision mini apps.
