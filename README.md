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

Audio Visualization is an interactive Streamlit application that converts an audio signal into a square image using deterministic signal-processing operations.

The project is not based on a trained generative model. Instead, it extracts audio descriptors, maps them to the magnitude and phase of a synthetic 2D Fourier spectrum, enforces Hermitian symmetry, and reconstructs an image with a 2D inverse Fourier transform.

When the same audio and the same parameters are used, the generated image is always the same.

## Live app

Streamlit app: <https://audio-visualization.streamlit.app>

Portfolio: <https://share.streamlit.io/user/trungtin-dinh>

Source code: <https://github.com/trungtin-dinh/audio_visualization>

## What the app does

The app takes an input audio signal and transforms its temporal, spectral and perceptual structure into a visual representation.

It supports:

- default open-source audio sample;
- user audio upload;
- browser audio recording, when supported by Streamlit;
- waveform and spectrogram preview;
- deterministic audio-to-image generation;
- several output modes, from grayscale reconstruction to colored and segmented renderings;
- export of the generated image as a PNG file;
- integrated English and French documentation.

The goal is not to recognize speech, classify music or generate a realistic photograph. The goal is to build an interpretable visual object from measurable audio features.

## Processing pipeline

```text
Input audio signal
        |
        v
Audio feature extraction
        |
        +-- multi-resolution STFT
        +-- continuous wavelet transform
        +-- mel spectrogram
        +-- chroma
        +-- MFCC
        +-- RMS energy
        +-- onset strength
        +-- spectral centroid
        +-- zero-crossing rate
        |
        v
2D Fourier spectrum construction
        |
        +-- magnitude grid
        +-- phase grid
        +-- Hermitian symmetrization
        |
        v
2D inverse Fourier transform
        |
        v
Output rendering
        |
        +-- grayscale image
        +-- RGB spectral image
        +-- black drawing overlay
        +-- luminance-modulated color image
        +-- segmented image
```

## Core idea

A real-valued image can be described by its 2D discrete Fourier transform. Each Fourier coefficient controls one spatial sinusoidal component. Its magnitude controls the strength of that component, while its phase controls where the component is positioned in the image.

This app builds the process in reverse.

The input audio signal is first transformed into several time-frequency and temporal descriptors. These descriptors are resized to the target image size and mixed into two grids:

```text
M_tilde[u, v]      magnitude grid
Phi_tilde[u, v]   phase grid
```

The complex spectrum is then assembled as:

```text
Z[u, v] = M_tilde[u, v] * exp(j * Phi_tilde[u, v])
```

Because the inverse 2D Fourier transform of an arbitrary complex spectrum is generally complex-valued, the spectrum is projected onto the Hermitian-symmetric subspace:

```text
Z_sym[u, v] = (Z[u, v] + conj(Z[-u mod N, -v mod N])) / 2
```

The final spatial image is reconstructed by:

```text
f[x, y] = Re(IFFT2(Z_sym))
```

The imaginary part is only numerical floating-point residue after Hermitian symmetrization and is discarded.

## Audio features used

The image is driven by complementary audio representations.

| Feature group | Role in the synthesis |
|---|---|
| Multi-resolution STFT magnitude | Spectral energy at several time-frequency resolutions |
| STFT phase | Main source of spatial phase structure |
| Continuous wavelet transform | Constant-Q time-frequency representation |
| Mel spectrogram | Perceptually weighted spectral energy |
| Chroma | Harmonic and pitch-class information |
| MFCC | Spectral envelope and timbral shape |
| RMS energy | Loudness dynamics |
| Onset strength | Rhythmic attacks and sudden spectral changes |
| Spectral centroid | Perceptual brightness |
| Zero-crossing rate | Noisiness and high-frequency temporal activity |

The magnitude grid mainly uses energy-related descriptors. The phase grid mainly uses phase and temporal descriptors, because phase has a strong effect on the spatial organization of the reconstructed image.

## Output modes

The app provides several rendering modes.

### Grayscale

A single full-band spectrum is reconstructed. The result is repeated over the red, green and blue channels.

### Colors

The frequency axis is split into low, mid and high bands. Each band is reconstructed independently and assigned to one RGB channel.

```text
Low frequencies   -> red channel
Mid frequencies   -> green channel
High frequencies  -> blue channel
```

### Black mix

The app combines the colored image with a binarized grayscale reconstruction. The grayscale image is converted into a sparse black drawing mask using Otsu thresholding.

### Luma mix

The grayscale reconstruction is used as a luminance coefficient map that modulates the colored image.

```text
I_out[x, y, c] = I_color[x, y, c] * alpha[x, y]
```

### Watershed / segmented rendering

The luminance-modulated image can be segmented into connected regions. Each region receives a representative color, and optional boundary rendering can be applied.

The app also exposes other segmentation approaches, including K-means, SLIC, Felzenszwalb and Mean-shift, depending on the installed dependencies.

## Sectioned synthesis

The app can process the whole audio at once or split it into chronological sections.

When sectioning is enabled, the waveform is divided into nearly equal temporal segments. Each segment generates its own image patch, and the patches are assembled into a final square canvas.

Available layouts include:

- chronological treemap;
- clockwise circular slices;
- concentric circles;
- concentric squares;
- vertical strips;
- horizontal strips.

This makes it possible to visualize the temporal evolution of the audio signal inside the final image.

## Main parameters

The interface exposes several groups of parameters.

| Group | Examples |
|---|---|
| Signal | image size, output mode, section layout, number of sections, CWT wavelet |
| Rendering | robust percentiles, gamma, contrast, brightness, saturation |
| Color | RGB band split, per-channel balance, RGB normalization mode |
| Effects | black mask density, luma strength, segmentation method, boundary style |
| Features | magnitude weights, phase weights, STFT range, CWT scales, mel bands, MFCC coefficients |

The default settings are chosen to generate a stable result without requiring the user to understand every parameter. Advanced users can still modify the full signal-processing chain.

## Repository structure

```text
.
├── app_sl.py              # Streamlit entry point
├── ui.py                  # Streamlit interface, session state and layout
├── audio_io.py            # Audio loading, decoding and default sample handling
├── features.py            # Audio feature extraction
├── grids.py               # Magnitude/phase grid construction and Hermitian projection
├── synthesis.py           # Full audio-to-image synthesis pipeline
├── segmentation.py        # Segmentation and region-color rendering methods
├── display.py             # Waveform, spectrogram, Fourier and image display helpers
├── utils.py               # Shared numerical utilities
├── config.py              # Constants, defaults and UI options
├── documentation_en.md    # English technical documentation
├── documentation_fr.md    # French technical documentation
├── requirements.txt       # Python dependencies
├── packages.txt           # System packages for deployment
├── LICENSE.txt            # MIT license
└── README.md              # Project description
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

The repository also includes a `packages.txt` file for Linux-based online deployments. It contains system-level audio decoding support.

## Run locally

```bash
streamlit run app_sl.py
```

The app should then be available at:

```text
http://localhost:8501
```

## Deployment notes

For Streamlit Community Cloud, use:

```text
app_sl.py
```

as the main file.

The metadata block at the top of this README also makes the repository compatible with Hugging Face Spaces-style Streamlit configuration.

## Documentation

The app includes two documentation tabs and two standalone Markdown files:

- `documentation_en.md`;
- `documentation_fr.md`.

They describe the audio feature extraction, 2D Fourier spectrum construction, Hermitian symmetry, output modes, segmentation methods, sectioned synthesis, layout algorithms, post-processing pipeline, parameters and limitations.

## Limitations

This app is an educational and artistic signal-processing demo. It does not understand the semantic content of the audio signal.

It does not detect words, instruments, genres or musical meaning. It only converts measurable signal descriptors into the magnitude and phase of a synthetic 2D Fourier spectrum.

The generated spectrum is not the Fourier transform of a natural image. It is an artificial spectrum derived from audio features. The output should therefore be interpreted as an audio-driven visual synthesis, not as a photographic reconstruction.

The computational cost increases with:

- output image size;
- number of sections;
- CWT maximum sample count;
- number of CWT scales;
- composite output modes such as `Black mix`, `Luma mix` and segmented rendering.

For online use, moderate image sizes and section counts are recommended.

## License

This project is released under the MIT License.

## Author

Developed by Trung-Tin Dinh as part of a portfolio of interactive signal, audio, image and computer vision mini apps.
