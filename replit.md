# ClarityAI - AI Image & Video Enhancement

## Overview
A web application that enhances dark, low-light images using the MIRNet deep learning model (PyTorch). Presented by Sidharth Kumar Pradhan & Naqibahmed Kadri, guided by Dr. Sidike Paheding, School of Engineering & Computing, Fairfield University.

## Architecture
- **Frontend**: React + TypeScript + Vite + Tailwind CSS + shadcn/ui
- **Backend**: Node.js + Express (TypeScript), spawns Python subprocess for ML inference
- **AI**: MIRNet model (PyTorch) for low-light image enhancement
  - Model architecture from `dblasko/low-light-event-img-enhancer`
  - Weights from HuggingFace: `dblasko/mirnet-low-light-img-enhancement`
  - Based on Keras MIRNet (https://keras.io/examples/vision/mirnet/)
- **Logging**: Custom structured logger (server/logger.ts) with color-coded levels, timestamps, and JSON metadata

## Key Features
- Upload images (JPEG, PNG, WebP, BMP) and videos (MP4, WebM, AVI, MOV, GIF)
- AI-powered low-light image enhancement using MIRNet
- Real-time job status polling
- Download enhanced results
- Drag-and-drop upload interface
- Preview enhanced images in-browser
- Fairfield University branding with crimson (#a01c2a) color scheme

## Model
| Model | Description | Framework |
|-------|-------------|-----------|
| MIRNet (Low-Light) | Multi-scale Residual Block architecture for low-light image enhancement | PyTorch |

## API Endpoints
- `GET /api/models` - List available AI models
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
server/
  index.ts             - Express server setup
  routes.ts            - API routes with comprehensive logging
  storage.ts           - In-memory job storage
  enhance.ts           - Node.js → Python subprocess bridge for MIRNet
  logger.ts            - Structured logger module
  mirnet_inference.py  - Python MIRNet inference script (PyTorch)
  model/MIRNet/        - MIRNet architecture definition files (PyTorch nn.Module)

models/
  mirnet_finetuned.pth - Pre-trained MIRNet weights (271MB)

client/src/
  pages/home.tsx       - Main UI page
  App.tsx              - Root component

shared/
  schema.ts            - Shared TypeScript types

uploads/               - Temp upload directory (auto-created)
enhanced/              - Output files directory (auto-created)
```

## Known Limitations
- Video enhancement is a placeholder (copies file without AI processing); full implementation requires ffmpeg for frame extraction
- MIRNet inference on CPU takes 30-60 seconds per image depending on resolution
- Model processes images at max 400px resolution for performance; output is resized back to original dimensions
