# AAC Encoder / Decoder – Multimedia Systems Project

This project implements an **AAC-like audio encoder and decoder** in Python, developed for the *Multimedia Systems* course.

The system progressively builds three encoding levels:

- **Level 1**: SSC + MDCT Filterbank  
- **Level 2**: + TNS (Temporal Noise Shaping)  
- **Level 3**: + Psychoacoustic Model + Quantizer + Huffman Coding  

The objective is to achieve **maximum compression with controlled perceptual distortion**, following the core principles of the AAC standard.

---

# Full Pipeline (Level 3)

## Encoder
```
WAV (time domain)
    ↓
SSC (Frame Type Selection)
    ↓
FilterBank (MDCT + Windowing)
    ↓
TNS (Frequency-domain prediction)
    ↓
Psychoacoustic Model (SMR computation)
    ↓
Quantizer (Non-uniform quantization)
    ↓
Huffman Encoding
    ↓
.mat compressed structure
```

## Decoder
```
.mat
    ↓
Huffman Decoding
    ↓
Inverse Quantizer
    ↓
Inverse TNS
    ↓
Inverse FilterBank (IMDCT)
    ↓
Overlap-Add
    ↓
WAV (time domain)
```

---

# Implemented Modules

## 1) SSC – Sequence Segmentation Control

Frames are classified into:

- `OLS` – Only Long Sequence  
- `ESH` – Eight Short Sequence  
- `LSS` – Long Start Sequence  
- `LPS` – Long Stop Sequence  

Attack detection is performed using a first-order high-pass IIR filter and energy evaluation over 8 segments of 128 samples each.

This mechanism prevents **pre-echo artifacts** during transient regions.

---

## 2) Filter Bank – MDCT

- Frame size: **2048 samples**
- Overlap: **50% (hop size = 1024)**
- Window type: **KBD (Kaiser-Bessel Derived)**

For `ESH` frames:
- Central 1152 samples are selected
- Split into 8 overlapping subframes of 256 samples

The MDCT is fully invertible and satisfies the Princen–Bradley condition for perfect reconstruction.

---

## 3) TNS – Temporal Noise Shaping

- LPC order: **p = 4**
- Band-wise normalization
- Quantized prediction coefficients
- Stability check
- FIR filtering in the MDCT domain

Purpose: 
- Reduce pre-echo  
- Redistribute quantization noise temporally  
- Improve perceptual quality  

---

## 4) Psychoacoustic Model

The psychoacoustic stage computes:

- FFT per frame/subframe
- Predictability (using two previous frames)
- Band energies
- Spreading function
- Tonality index
- Masking threshold
- **SMR (Signal-to-Mask Ratio)**

The SMR determines how much quantization noise is allowed per critical band.

---

## 5) Non-Uniform Quantizer

AAC-style non-uniform quantization:

S(k) = sign(X(k)) ⌊ |X(k) · 2^(−a/4)|^(3/4) ⌋

Dequantization:

X̂(k) = sign(S(k)) · |S(k)|^(4/3) · 2^(a/4)

### Iterative Scalefactor Adjustment

Scalefactors are increased while:

- Pe(b) < T(b)
- |a(b+1) − a(b)| ≤ 60

The process stops when:
- All bands satisfy masking constraints, or  
- Maximum iterations are reached  

---

## 6) Huffman Coding

- Automatic codebook selection for MDCT symbols
- Codebook 11 forced for scalefactors
- ESC mode support

Per frame metrics recorded:

- `bits_S`
- `bits_sfc`
- `nonzero` coefficients
- `maxAbs`
- selected `codebook`
- quantizer iterations

---

# Experimental Results

## Level 1 & Level 2

| Level | SNR (dB) |
|--------|----------|
| 1      | 254.03629 |
| 2      | 254.03630 |

Practically lossless (only floating-point rounding errors).

---

## Level 3 (example – max_iter = 60)

- **SNR:** 10.8 dB  
- **Bitrate:** 177.02 kbps  
- **Compression Ratio:** 8.68×  

### Effect of max iterations

| Max Iter | SNR (dB) | Bitrate (kbps) | Compression |
|-----------|----------|----------------|-------------|
| 40 | 24.71 | 372.59 | 4.12× |
| 45 | 22.20 | 354.03 | 4.34× |
| 50 | 18.82 | 255.18 | 6.02× |
| 55 | 14.73 | 243.57 | 6.31× |
| 60 | 10.80 | 177.02 | 8.68× |
| 65 | 7.68  | 206.07 | 7.45× |
| 70 | 6.44  | 207.86 | 7.39× |
| 75 | 5.97  | 209.72 | 7.32× |
| 100 | 5.97 | 214.14 | 7.17× |
| 250 | 5.97 | 218.00 | 7.05× |

An interesting behavior appears beyond 60 iterations, where bitrate starts increasing again due to large scalefactor differences increasing Huffman cost.

---

# How to Run

From the project root:

```bash
python run_level_1.py
python run_level_2.py
python run_level_3.py
```

Or with full path:

```bash
python path/to/run_level_3.py
```

---

# Outputs

- `output_levelX.wav` – Decoded audio
- `coded_level3_metrics.csv` – Per-frame encoding metrics

---

# Academic Context

Developed as part of the **Multimedia Systems** course assignment.  
Implements a simplified yet structurally faithful AAC-like codec.

---
