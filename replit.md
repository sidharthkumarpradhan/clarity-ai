# ClarityAI - AI Image & Video Enhancement

## Overview
A web application that enhances dark, low-light images using multiple state-of-the-art enhancement models (2 deep learning + 3 traditional CV). Available as both a React web app and a Streamlit app for deployment on Streamlit Cloud. Presented by Sidharth Kumar Pradhan & Naqibahmed Kadri, guided by Dr. Sidike Paheding, School of Engineering & Computing, Fairfield University.

## Architecture
- **Frontend (React)**: React + TypeScript + Vite + Tailwind CSS + shadcn/ui
- **Frontend (Streamlit)**: `streamlit_app.py` - standalone Python app for Streamlit Cloud
- **Backend**: Node.js + Express (TypeScript), spawns persistent Python subprocess for ML inference
- **AI Models**: 5 enhancement models (see table below)
- **Logging**: Custom structured logger (server/logger.ts)

## Models
| Model | Type | Speed | Quality | Source |
|-------|------|-------|---------|--------|
| MIRNet | Deep Learning (PyTorch) | ~30s | High | ECCV 2020, HuggingFace weights |
| Zero-DCE | Deep Learning (PyTorch) | ~0.5s | Good | CVPR 2020, Li-Chongyi/Zero-DCE |
| CLAHE | Traditional CV (OpenCV) | Instant | Moderate | Adaptive Histogram Equalization |
| Histogram EQ | Traditional CV (OpenCV) | Instant | Basic | Global Histogram Equalization |
| Gamma Correction | Traditional CV (OpenCV) | Instant | Basic | Adaptive Power-Law Transform |

## Key Features
- Upload images (JPEG, PNG, WebP, BMP) and videos (MP4, WebM, AVI, MOV, GIF)
- 5 enhancement models with selectable options
- Comparison mode (Streamlit) for side-by-side model evaluation
- Real-time job status polling (React app)
- Download enhanced results
- Drag-and-drop upload interface
- Fairfield University branding with crimson (#a01c2a) color scheme

## Streamlit Cloud Deployment
The app is configured for deployment on https://share.streamlit.io/:
- Main file: `streamlit_app.py`
- Requirements: `streamlit_requirements.txt`
- System packages: `packages.txt`
- Config: `.streamlit/config.toml`
- Model weights are auto-downloaded from HuggingFace/GitHub on first run
- `.gitignore` excludes `models/*.pth` (271MB+ files)

## API Endpoints (React App)
- `GET /api/models` - List all 5 available models
- `GET /api/jobs` - Get all enhancement jobs
- `GET /api/jobs/:id` - Get a specific job
- `DELETE /api/jobs/:id` - Delete a job
- `POST /api/enhance/image` - Upload and enhance image (multipart)
- `POST /api/enhance/video` - Upload and enhance video (multipart)
- `GET /api/enhanced/:filename` - Serve enhanced files

## Environment Variables
- `HF_TOKEN` - Hugging Face API token (for potential future use)

## File Structure
```
streamlit_app.py              - Streamlit app (for Streamlit Cloud deployment)
streamlit_requirements.txt    - Python deps for Streamlit Cloud
packages.txt                  - System deps for Streamlit Cloud
.streamlit/config.toml        - Streamlit theme and server config

server/
  index.ts                    - Express server setup
  routes.ts                   - API routes
  storage.ts                  - In-memory job storage
  enhance.ts                  - Node.js → Python subprocess bridge (5 models)
  logger.ts                   - Structured logger module
  mirnet_server.py            - Persistent Python multi-model inference server
  model/MIRNet/               - MIRNet architecture (PyTorch nn.Module)
  model/ZeroDCE/              - Zero-DCE architecture (PyTorch nn.Module)

models/
  mirnet_finetuned.pth        - MIRNet weights (271MB, auto-downloaded)
  zero_dce.pth                - Zero-DCE weights (320KB, auto-downloaded)

client/src/
  pages/home.tsx              - Main UI page
  App.tsx                     - Root component

shared/
  schema.ts                   - Shared TypeScript types
```

## Known Limitations
- Video enhancement is a placeholder; full implementation requires ffmpeg
- MIRNet inference on CPU takes 30-60 seconds per image
- MIRNet processes at max 400px resolution for performance
- Streamlit Cloud free tier has 1GB RAM limit; MIRNet weights are 271MB
