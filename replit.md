# ClarityAI - AI Image & Video Enhancement

## Overview
A web application that enhances blurry, dark, or low-resolution images and videos using AI super-resolution models from Hugging Face.

## Architecture
- **Frontend**: React + TypeScript + Vite + Tailwind CSS + shadcn/ui
- **Backend**: Node.js + Express (TypeScript)
- **AI**: Hugging Face Inference API using Swin2SR models
- **File handling**: Multer for uploads, Sharp for image processing

## Key Features
- Upload images (JPEG, PNG, WebP, BMP) and videos (MP4, WebM, AVI, MOV, GIF)
- AI-powered enhancement using Swin2SR models (2x, 4x upscaling)
- Real-time job status polling
- Download enhanced results
- Drag-and-drop upload interface
- Preview enhanced images in-browser

## Models Available
| Model | Description | Scale |
|-------|-------------|-------|
| Swin2SR x2 | Fast classical super-resolution | 2x |
| Swin2SR x4 Real-World | Real-world image restoration | 4x |
| Swin2SR x4 Compressed | Compressed artifact removal + upscale | 4x |

## API Endpoints
- `GET /api/models` - List available AI models
- `GET /api/jobs` - Get all enhancement jobs
- `GET /api/jobs/:id` - Get a specific job
- `DELETE /api/jobs/:id` - Delete a job
- `POST /api/enhance/image` - Upload and enhance image (multipart)
- `POST /api/enhance/video` - Upload and enhance video (multipart)
- `GET /api/enhanced/:filename` - Serve enhanced files

## Environment Variables
- `HF_TOKEN` - Hugging Face API token (required)

## File Structure
```
server/
  index.ts      - Express server setup
  routes.ts     - API routes
  storage.ts    - In-memory job storage
  enhance.ts    - HF API integration logic

client/src/
  pages/home.tsx  - Main UI page
  App.tsx         - Root component

shared/
  schema.ts     - Shared TypeScript types

uploads/        - Temp upload directory (auto-created)
enhanced/       - Output files directory (auto-created)
```

## Future Enhancements
- ffmpeg integration for true frame-by-frame video enhancement
- Additional modalities: audio denoising, hyperspectral imaging
- Authentication and user accounts
- Processing history persistence (PostgreSQL)
- Batch processing support
- Edge device optimization (WebGPU, ONNX runtime)
