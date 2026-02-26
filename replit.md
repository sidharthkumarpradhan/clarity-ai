# ClarityAI - AI Image & Video Enhancement

## Overview
A web application that enhances blurry, dark, or low-resolution images and videos using AI super-resolution models. Presented by Sidharth Kumar Pradhan & Naqibahmed Kadri, guided by Dr. Sidike Paheding, School of Engineering & Computing, Fairfield University.

## Architecture
- **Frontend**: React + TypeScript + Vite + Tailwind CSS + shadcn/ui
- **Backend**: Node.js + Express (TypeScript)
- **AI**: Swin2SR models via @huggingface/transformers (Transformers.js, ONNX runtime, runs locally on CPU)
- **File handling**: Multer for uploads, Sharp for image metadata
- **Logging**: Custom structured logger (server/logger.ts) with color-coded levels, timestamps, and JSON metadata

## Key Features
- Upload images (JPEG, PNG, WebP, BMP) and videos (MP4, WebM, AVI, MOV, GIF)
- AI-powered enhancement using Swin2SR models (2x, 4x upscaling)
- Real-time job status polling
- Download enhanced results
- Drag-and-drop upload interface
- Preview enhanced images in-browser
- Fairfield University branding with crimson (#a01c2a) color scheme

## Models Available (ONNX via Xenova)
| Model | HF Model ID | Scale |
|-------|-------------|-------|
| Swin2SR x2 Classical | Xenova/swin2SR-classical-sr-x2-64 | 2x |
| Swin2SR x4 Real-World | Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr | 4x |
| Swin2SR x4 Compressed | Xenova/swin2SR-compressed-sr-x4-48 | 4x |

## API Endpoints
- `GET /api/models` - List available AI models
- `GET /api/jobs` - Get all enhancement jobs
- `GET /api/jobs/:id` - Get a specific job
- `DELETE /api/jobs/:id` - Delete a job
- `POST /api/enhance/image` - Upload and enhance image (multipart)
- `POST /api/enhance/video` - Upload and enhance video (multipart)
- `GET /api/enhanced/:filename` - Serve enhanced files

## Environment Variables
- `HF_TOKEN` - Hugging Face API token (used for model auth)

## File Structure
```
server/
  index.ts      - Express server setup
  routes.ts     - API routes with comprehensive logging
  storage.ts    - In-memory job storage
  enhance.ts    - Transformers.js ONNX model inference
  logger.ts     - Structured logger module

client/src/
  pages/home.tsx  - Main UI page
  App.tsx         - Root component

shared/
  schema.ts     - Shared TypeScript types

uploads/        - Temp upload directory (auto-created)
enhanced/       - Output files directory (auto-created)
```

## Known Limitations
- Video enhancement is a placeholder (copies file without AI processing); full implementation requires ffmpeg for frame extraction
- First model load takes ~10s to download ONNX weights; subsequent requests use cached pipeline
- CPU-only inference; processing time depends on image size
