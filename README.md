# ClarityAI — AI-Powered Low-Light Image Enhancement

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://clarity-ai-fairfield-stag.streamlit.app/](https://clarity-ai-fairfield-stag.streamlit.app/)/)

ClarityAI is an interactive Streamlit web app for enhancing dark and low-light photographs using a suite of deep learning and classical computer vision methods. Upload any image and choose from five enhancement algorithms — from state-of-the-art neural networks to instant classical techniques.

> **Presented by:** Sidharth Kumar Pradhan & Naqibahmed Kadri  
> **Guided by:** Dr. Sidike Paheding  
> **School of Engineering & Computing — Fairfield University**

---

## Live Demo

🔗 **[https://clarity-ai-hqdrx4bj3hfjpjrnnwkiky.streamlit.app](https://clarity-ai-hqdrx4bj3hfjpjrnnwkiky.streamlit.app/)**

---

## Features

- **5 enhancement models** — 2 deep learning, 3 classical CV
- **Single Model mode** — run one algorithm and download the result
- **Compare Models mode** — run multiple algorithms side-by-side with brightness metrics
- **Brightness analytics** — resolution, mean brightness score, and dark/bright classification
- **Blend slider** — smoothly transition between original and enhanced output
- **One-click download** — save enhanced images as PNG
- **Auto weight download** — DL model weights are fetched from HuggingFace/GitHub on first run; no manual setup needed

---

## Enhancement Models

| Model | Type | Speed | Quality | Reference |
|---|---|---|---|---|
| **MIRNet** | Deep Learning | ~30s (CPU) | High | ECCV 2020 |
| **Zero-DCE** | Deep Learning | ~0.5s (CPU) | Good | CVPR 2020 |
| **CLAHE** | Traditional CV | Instant | Moderate | Pizer et al. 1987 |
| **Histogram Equalization** | Traditional CV | Instant | Basic | Gonzalez & Woods |
| **Gamma Correction** | Traditional CV | Instant | Basic | Power-Law Transform |

### MIRNet (Multi-scale Residual Block Network)

Learns enriched features via multi-scale residual blocks and dual attention mechanisms. Processes image features at multiple spatial resolutions simultaneously, preserving fine texture while recovering brightness. Model weights auto-downloaded from [HuggingFace](https://huggingface.co/dblasko/mirnet-low-light-img-enhancement).

### Zero-DCE (Zero-Reference Deep Curve Estimation)

A lightweight model that estimates per-image tonal adjustment curves with no paired training data required. Very fast on CPU. Weights auto-downloaded from the [official Zero-DCE GitHub repository](https://github.com/Li-Chongyi/Zero-DCE).

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

Divides the image into small tiles and applies adaptive histogram equalization per tile, with contrast clipping to suppress noise amplification. Clip limit and tile size are tunable in the sidebar.

### Histogram Equalization

Global contrast enhancement by redistributing intensity values in the YCbCr luma channel. A fast, parameter-free baseline.

### Gamma Correction

Auto-estimates an optimal gamma value from mean image brightness and applies a power-law lookup table transform.

---

## Project Structure

```
clarity-ai/
├── streamlit_app.py          # Main Streamlit application (all UI + enhancement logic)
├── requirements.txt          # Python dependencies for Streamlit Cloud
├── packages.txt              # System packages (libGL for headless OpenCV)
├── .streamlit/
│   └── config.toml           # Streamlit theme and server configuration
├── server/
│   └── model/
│       ├── __init__.py
│       ├── MIRNet/
│       │   └── model.py      # MIRNet architecture (PyTorch)
│       └── ZeroDCE/
│           └── model.py      # Zero-DCE architecture (PyTorch)
└── attached_assets/
    └── fairfieldUniversityLogo_*.png  # Sidebar logo
```

> **Model weights** (`models/*.pth`) are excluded from the repository via `.gitignore`.
> They are automatically downloaded at runtime (~258 MB for MIRNet, ~1 MB for Zero-DCE).

---

## Local Setup

### Prerequisites

- Python 3.10+
- pip

### Install & Run

```bash
# Clone the repo
git clone https://github.com/sidharthkumarpradhan/clarity-ai.git
cd clarity-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

On first launch the app will automatically download MIRNet (~258 MB) and Zero-DCE model weights into a local `models/` folder. Subsequent runs use the cached files.

---

## Deploying to Streamlit Cloud

1. Fork or push this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repository and set **Branch** to `main`.
4. Set **Main file path** to `streamlit_app.py`.
5. Click **Deploy** — Streamlit Cloud installs `requirements.txt` and `packages.txt` automatically.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit>=1.30.0` | Web application framework |
| `torch>=2.0.0` | Deep learning inference (MIRNet, Zero-DCE) |
| `torchvision>=0.15.0` | Image tensor transforms |
| `Pillow>=9.0.0` | Image I/O and format handling |
| `numpy>=1.24.0` | Array operations |
| `opencv-python-headless>=4.8.0` | CLAHE, Histogram Equalization, Gamma Correction |

**System package** (`packages.txt`): `libgl1-mesa-glx` — required for headless OpenCV on Streamlit Cloud.

---

## References

- **MIRNet:** Zamir et al., *Learning Enriched Features for Real Image Restoration and Enhancement*, ECCV 2020. [arXiv:2003.06792](https://arxiv.org/abs/2003.06792)
- **Zero-DCE:** Guo et al., *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*, CVPR 2020. [arXiv:2001.06826](https://arxiv.org/abs/2001.06826)
- **CLAHE:** Pizer et al., *Adaptive Histogram Equalization and Its Variations*, Computer Vision, Graphics, and Image Processing, 1987.

---

## License

MIT License. Model weights are subject to their respective upstream licenses.
