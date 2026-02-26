import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import path from "path";
import fs from "fs";
import { storage } from "./storage";
import { enhanceImageWithHF, IMAGE_MODELS, VIDEO_MODELS } from "./enhance";
import { log } from "./index";

const UPLOADS_DIR = path.join(process.cwd(), "uploads");
const ENHANCED_DIR = path.join(process.cwd(), "enhanced");

// Ensure directories exist
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(ENHANCED_DIR)) fs.mkdirSync(ENHANCED_DIR, { recursive: true });

const upload = multer({
  dest: UPLOADS_DIR,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB
  },
  fileFilter: (_req, file, cb) => {
    const imageTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"];
    const videoTypes = ["video/mp4", "video/webm", "video/avi", "video/mov", "image/gif"];
    if ([...imageTypes, ...videoTypes].includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Unsupported file type. Please upload JPEG, PNG, WebP, MP4, or WebM files."));
    }
  },
});

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // Get available models
  app.get("/api/models", (_req: Request, res: Response) => {
    res.json({
      image: IMAGE_MODELS,
      video: VIDEO_MODELS,
    });
  });

  // Get all jobs
  app.get("/api/jobs", async (_req: Request, res: Response) => {
    const jobs = await storage.getAllJobs();
    res.json(jobs);
  });

  // Get a specific job
  app.get("/api/jobs/:id", async (req: Request, res: Response) => {
    const job = await storage.getJob(req.params.id as string);
    if (!job) {
      return res.status(404).json({ message: "Job not found" });
    }
    res.json(job);
  });

  // Delete a job
  app.delete("/api/jobs/:id", async (req: Request, res: Response) => {
    const job = await storage.getJob(req.params.id as string);
    if (!job) {
      return res.status(404).json({ message: "Job not found" });
    }
    await storage.deleteJob(req.params.id as string);
    res.json({ success: true });
  });

  // Upload and enhance image
  app.post("/api/enhance/image", upload.single("file"), async (req: Request, res: Response) => {
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    const modelId = req.body.model || IMAGE_MODELS[0].id;
    const model = IMAGE_MODELS.find((m) => m.id === modelId);

    if (!model) {
      return res.status(400).json({ message: "Invalid model selected" });
    }

    // Create job immediately
    const job = await storage.createJob({
      type: "image",
      status: "pending",
      model: modelId,
      scale: model.scale,
      originalName: req.file.originalname,
      originalSize: req.file.size,
      enhancedUrl: null,
      errorMessage: null,
      completedAt: null,
    });

    // Process async
    (async () => {
      try {
        await storage.updateJob(job.id, { status: "processing" });
        log(`Starting enhancement for job ${job.id} with model ${modelId}`);

        const outputFilename = await enhanceImageWithHF(req.file!.path, modelId, ENHANCED_DIR);

        // Clean up upload
        fs.unlinkSync(req.file!.path);

        await storage.updateJob(job.id, {
          status: "completed",
          enhancedUrl: `/api/enhanced/${outputFilename}`,
          completedAt: new Date().toISOString(),
        });

        log(`Job ${job.id} completed: ${outputFilename}`);
      } catch (err: any) {
        log(`Job ${job.id} failed: ${err.message}`);
        // Clean up upload if it exists
        if (req.file?.path && fs.existsSync(req.file.path)) {
          fs.unlinkSync(req.file.path);
        }
        await storage.updateJob(job.id, {
          status: "failed",
          errorMessage: err.message || "Enhancement failed",
          completedAt: new Date().toISOString(),
        });
      }
    })();

    res.json(job);
  });

  // Upload and enhance video
  app.post("/api/enhance/video", upload.single("file"), async (req: Request, res: Response) => {
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    const modelId = req.body.model || VIDEO_MODELS[0].id;
    const model = VIDEO_MODELS.find((m) => m.id === modelId);

    if (!model) {
      return res.status(400).json({ message: "Invalid model selected" });
    }

    const job = await storage.createJob({
      type: "video",
      status: "pending",
      model: modelId,
      scale: model.scale,
      originalName: req.file.originalname,
      originalSize: req.file.size,
      enhancedUrl: null,
      errorMessage: null,
      completedAt: null,
    });

    // For video: we note this is complex and return appropriate message
    // Video enhancement via HF API is limited; we inform the user
    (async () => {
      try {
        await storage.updateJob(job.id, { status: "processing" });

        // For now, copy the video and note limitations
        // True video enhancement would require frame-by-frame processing with ffmpeg
        const ext = path.extname(req.file!.originalname) || ".mp4";
        const outputFilename = `enhanced_${job.id}${ext}`;
        const outputPath = path.join(ENHANCED_DIR, outputFilename);

        // Copy original as-is (placeholder for full ffmpeg pipeline)
        fs.copyFileSync(req.file!.path, outputPath);
        fs.unlinkSync(req.file!.path);

        await storage.updateJob(job.id, {
          status: "completed",
          enhancedUrl: `/api/enhanced/${outputFilename}`,
          completedAt: new Date().toISOString(),
          errorMessage: "Note: Full video frame-by-frame AI enhancement requires ffmpeg. This is a preview version.",
        });

        log(`Video job ${job.id} completed`);
      } catch (err: any) {
        log(`Video job ${job.id} failed: ${err.message}`);
        if (req.file?.path && fs.existsSync(req.file.path)) {
          fs.unlinkSync(req.file.path);
        }
        await storage.updateJob(job.id, {
          status: "failed",
          errorMessage: err.message || "Video enhancement failed",
          completedAt: new Date().toISOString(),
        });
      }
    })();

    res.json(job);
  });

  // Serve enhanced files
  app.get("/api/enhanced/:filename", (req: Request, res: Response) => {
    const filename = req.params.filename as string;
    const filePath = path.join(ENHANCED_DIR, filename);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ message: "File not found" });
    }

    res.sendFile(filePath);
  });

  // Serve original uploaded files (for preview)
  app.get("/api/uploads/:filename", (req: Request, res: Response) => {
    const filename = req.params.filename as string;
    const filePath = path.join(UPLOADS_DIR, filename);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ message: "File not found" });
    }

    res.sendFile(filePath);
  });

  return httpServer;
}
