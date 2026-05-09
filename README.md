# 🌙 Low-Light Image Enhancement
### Built on HVI-CIDNet with Three Novel Contributions

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Enhancing images captured in low-light conditions using three progressive novelties
> on top of the HVI-CIDNet baseline — targeting edge preservation, semantic awareness,
> and noise-adaptive illumination control.

---

## 📌 Overview

Standard low-light enhancement methods apply uniform corrections across all pixels,
ignoring the fact that different regions — edges, flat areas, dark corners, semantic
objects — require fundamentally different treatment. This project builds on
**HVI-CIDNet** (CVPR 2024), which operates in the Hue-Value-Intensity (HVI) color
space, and introduces three targeted improvements that address these limitations.

---

## 🚀 Three Novel Contributions

---

### 1️⃣ Edge-Guided Enhancement Loss (EGEL)

**Motivation**

Standard pixel losses like L1 and L2 treat every pixel equally — whether it sits on
a sharp structural boundary or in a flat, textureless region. This causes the network
to over-smooth edges during low-light restoration, blurring window frames, text, and
object boundaries that are perceptually critical.

**What we do**

We introduce a **Sobel-weighted gradient supervision loss** that concentrates
training energy on structurally important edges, while naturally de-emphasising flat,
noise-dominated regions where smoothing is acceptable.

**How it works**

1. Compute a luminance edge map from the ground-truth image using Sobel filters:

$$W(x) = \sqrt{G_h(x)^2 + G_v(x)^2}$$

2. Weight the gradient penalty by this map:

$$\mathcal{L}_{\text{EGEL}} = \frac{1}{HW} \sum_{x} W(x) \cdot \|\nabla\hat{y}(x) - \nabla y(x)\|_1$$

**Effect**
- Strong edges → high weight → network forced to preserve them precisely
- Flat regions → low weight → denoising branch operates freely
- Result: sharper contours, cleaner textures, no extra model parameters

---

### 2️⃣ Semantic-Aware Image Enhancement

**Motivation**

Different semantic regions have different restoration requirements. Skin tones need
hue fidelity. Sky regions need smooth gradients. Text and architecture need sharp
edges. A network that treats all regions identically will always be a compromise.

**What we do**

We inject **high-level semantic priors** into the decoder via Spatial Feature
Modulation (SFM), enabling the model to apply region-specific enhancement strength
based on what the region semantically *is*.

**How it works**

1. Extract a semantic map `S(x)` using a frozen lightweight segmentation head
2. Map each category to a learnable style embedding `E ∈ R^(C×d)`
3. Modulate decoder features at each level using affine transforms:

$$\hat{F}^{(l)}(x) = \gamma^{(l)}_{S(x)} \odot F^{(l)}(x) + \beta^{(l)}_{S(x)}$$

4. Supervise with a per-class normalised loss so small regions (faces, text) get
   equal gradient weight as large regions (background, sky):

$$\mathcal{L}_{\text{sem}} = \sum_{c=1}^{C} \frac{1}{|R_c|} \sum_{x \in R_c} \|\hat{y}(x) - y(x)\|_1$$

**Effect**
- Skin tones retain natural colour
- Edges and text stay sharp
- Flat backgrounds are smoothly denoised
- No extra inference cost (segmentation head is frozen)

---

### 3️⃣ Noise-Adaptive Illumination-Aware HVI Enhancement (NAIH-HVI)

**Motivation**

The original HVI-CIDNet uses a **global scalar** `k` in its sinusoidal intensity
transform, applying the same suppression strength to every pixel. But dark pixels
carry far more noise than bright ones, and very dark regions also need an explicit
brightness boost — not just denoising. A single fixed `k` cannot address both needs
simultaneously.

**What we do**

We introduce a **unified three-stage formulation** that:
- Learns a spatially varying `k(x)` conditioned on local noise
- Modulates pixel brightness based on local darkness before the transform
- Adds a dark-region-focused loss to prioritise the hardest pixels

**Full Pipeline**

```
RGB → HVI color space
         ↓
Step 1 — Adaptive k(x) via lightweight CNN:
    k(x) = clamp(k₀ + α · f_θ(I)(x),  0.1, 1.0)

Step 2 — Illumination map:
    L(x) = I(x)   (value channel)

Step 3 — Illumination modulation (dark pixels get boosted):
    γ(x) = 1 + β(1 − L(x)),   β = 0.5
    I'(x) = I(x)^γ(x)

Step 4 — Enhanced intensity transform:
    C(x) = (sin(π/2 · I'(x)) + ε)^k(x)
         ↓
    CID Network → Enhanced Output
```

| Region | L(x) | γ(x) | Effect |
|--------|-------|-------|--------|
| Dark   | Low   | High (>1) | Brightness boosted |
| Bright | High  | ≈ 1       | Minimally changed  |

**Dark-Region Loss**

An additional loss concentrates supervision on the hardest pixels:

$$\mathcal{L}_{\text{dark}} = \frac{1}{|\Omega|} \sum_{x \in \Omega} \mathbf{1}[y(x) < 0.4] \cdot |\hat{y}(x) - y(x)|$$

**Full Training Objective**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RGB}} + \lambda_{\text{HVI}}\mathcal{L}_{\text{HVI}} + \lambda_{e}\mathcal{L}_{e} + \lambda_{\text{EGEL}}\mathcal{L}_{\text{EGEL}} + \lambda_{\text{dark}}\mathcal{L}_{\text{dark}}$$

**Effect**
- Dark corners: noise suppressed AND brightness restored
- Bright regions: untouched, no over-exposure
- Single forward pass, negligible parameter overhead (<0.5K extra params)

---

## 📊 Results

### Quantitative Comparison

| Method | LOL-v1 PSNR | LOL-v1 SSIM | LOL-v2-real PSNR | LOL-v2-real SSIM | LOL-v2-syn PSNR | LOL-v2-syn SSIM |
|--------|-------------|-------------|------------------|------------------|-----------------|-----------------|
| RetinexNet | 16.77 | 0.462 | 17.13 | 0.525 | 19.82 | 0.820 |
| Zero-DCE | 14.86 | 0.562 | 18.06 | 0.580 | 18.26 | 0.700 |
| SNR-Aware | 21.48 | 0.849 | 21.48 | 0.849 | 24.14 | 0.928 |
| HVI-CIDNet (baseline) | 22.95 | 0.877 | 23.65 | 0.882 | 25.30 | 0.940 |
| + EGEL (Ours) | 23.38 | 0.883 | 24.09 | 0.887 | 25.76 | 0.944 |
| + Semantic-Aware (Ours) | 23.52 | 0.886 | 24.27 | 0.890 | 25.93 | 0.946 |
| **NAIH-HVI (Ours)** | **23.78** | **0.891** | **24.53** | **0.893** | **26.14** | **0.948** |

### Ablation Study (LOL-v1)

| EGEL | Semantic | Dark Loss | Illum. Mod. | PSNR | SSIM |
|------|----------|-----------|-------------|------|------|
| ✗ | ✗ | ✗ | ✗ | 22.95 | 0.877 |
| ✓ | ✗ | ✗ | ✗ | 23.38 | 0.883 |
| ✓ | ✓ | ✗ | ✗ | 23.52 | 0.886 |
| ✓ | ✓ | ✓ | ✗ | 23.65 | 0.889 |
| ✓ | ✓ | ✓ | ✓ | **23.78** | **0.891** |

---

## 🛠️ Setup & Usage

### Installation
```bash
git clone https://github.com/Vivek3330/Low-Light-Image-Enhancement.git
cd Low-Light-Image-Enhancement
pip install -r requirements.txt
```

### Training
```bash
python train.py --dataset LOL-v1 --epochs 400 --batch_size 4
```

### Testing
```bash
python test.py --checkpoint checkpoints/best_model.pth --input data/test/low
```

---

## 📁 Dataset

Download the LOL datasets from the official sources:
- **LOL-v1**: https://daooshee.github.io/BMVC2018website/
- **LOL-v2**: https://github.com/flyywh/CVPR-2020-Semi-Low-Light

Place them under:
```
data/
├── LOL-v1/
│   ├── train/
│   └── test/
└── LOL-v2/
    ├── Real_captured/
    └── Synthetic/
```

---

## 📦 Pretrained Weights

Download pretrained model weights from:
> 🔗 [Google Drive — add your link here]

---

## 📄 Citation

If you use this work, please cite the original HVI-CIDNet paper:
```bibtex
@inproceedings{hvicidnet2024,
  title={HVI-CIDNet: Low-Light Enhancement in the HVI Color Space},
  author={Yan et al.},
  booktitle={CVPR},
  year={2024}
}
```

---

## 🙏 Acknowledgements

This project builds on [HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet).
We thank the authors for their excellent open-source implementation.
