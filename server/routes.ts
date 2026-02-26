import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import path from "path";
import fs from "fs";
import { storage } from "./storage";
import { enhanceImageWithHF, IMAGE_MODELS, VIDEO_MODELS } from "./enhance";
import { logger } from "./logger";

const MODULE = "routes";
const UPLOADS_DIR = path.join(process.cwd(), "uploads");
const ENHANCED_DIR = path.join(process.cwd(), "enhanced");

if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(ENHANCED_DIR)) fs.mkdirSync(ENHANCED_DIR, { recursive: true });

const upload = multer({
  dest: UPLOADS_DIR,
  limits: {
    fileSize: 50 * 1024 * 1024,
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
  app.get("/api/models", (_req: Request, res: Response) => {
    logger.info(MODULE, "Fetching available models");
    res.json({
      image: IMAGE_MODELS,
      video: VIDEO_MODELS,
    });
  });

  app.get("/api/jobs", async (_req: Request, res: Response) => {
    const jobs = await storage.getAllJobs();
    res.json(jobs);
  });

  app.get("/api/jobs/:id", async (req: Request, res: Response) => {
    const id = req.params.id as string;
    logger.debug(MODULE, `Fetching job`, { id });
    const job = await storage.getJob(id);
    if (!job) {
      logger.warn(MODULE, `Job not found`, { id });
      return res.status(404).json({ message: "Job not found" });
    }
    res.json(job);
  });

  app.delete("/api/jobs/:id", async (req: Request, res: Response) => {
    const id = req.params.id as string;
    logger.info(MODULE, `Deleting job`, { id });
    const job = await storage.getJob(id);
    if (!job) {
      logger.warn(MODULE, `Cannot delete - job not found`, { id });
      return res.status(404).json({ message: "Job not found" });
    }
    await storage.deleteJob(id);
    logger.info(MODULE, `Job deleted successfully`, { id });
    res.json({ success: true });
  });

  app.post("/api/enhance/image", upload.single("file"), async (req: Request, res: Response) => {
    if (!req.file) {
      logger.warn(MODULE, "Image upload attempted with no file");
      return res.status(400).json({ message: "No file uploaded" });
    }

    const modelId = req.body.model || IMAGE_MODELS[0].id;
    const model = IMAGE_MODELS.find((m) => m.id === modelId);

    if (!model) {
      logger.error(MODULE, "Invalid model selected", { modelId });
      return res.status(400).json({ message: "Invalid model selected" });
    }

    logger.info(MODULE, `Image upload received`, {
      filename: req.file.originalname,
      size: req.file.size,
      mimetype: req.file.mimetype,
      model: modelId,
      scale: model.scale,
    });

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

    logger.info(MODULE, `Job created`, { jobId: job.id, model: modelId });

    (async () => {
      try {
        await storage.updateJob(job.id, { status: "processing" });
        logger.info(MODULE, `Job processing started`, { jobId: job.id, model: modelId });

        const outputFilename = await enhanceImageWithHF(req.file!.path, modelId, ENHANCED_DIR);

        if (req.file?.path && fs.existsSync(req.file.path)) {
          fs.unlinkSync(req.file.path);
          logger.debug(MODULE, `Cleaned up uploaded file`, { path: req.file.path });
        }

        await storage.updateJob(job.id, {
          status: "completed",
          enhancedUrl: `/api/enhanced/${outputFilename}`,
          completedAt: new Date().toISOString(),
        });

        logger.info(MODULE, `Job completed successfully`, { jobId: job.id, outputFilename });
      } catch (err: any) {
        logger.error(MODULE, `Job failed`, { jobId: job.id, error: err.message, stack: err.stack?.slice(0, 300) });
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

  app.post("/api/enhance/video", upload.single("file"), async (req: Request, res: Response) => {
    if (!req.file) {
      logger.warn(MODULE, "Video upload attempted with no file");
      return res.status(400).json({ message: "No file uploaded" });
    }

    const modelId = req.body.model || VIDEO_MODELS[0].id;
    const model = VIDEO_MODELS.find((m) => m.id === modelId);

    if (!model) {
      logger.error(MODULE, "Invalid video model selected", { modelId });
      return res.status(400).json({ message: "Invalid model selected" });
    }

    logger.info(MODULE, `Video upload received`, {
      filename: req.file.originalname,
      size: req.file.size,
      model: modelId,
    });

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

    (async () => {
      try {
        await storage.updateJob(job.id, { status: "processing" });
        logger.info(MODULE, `Video processing started`, { jobId: job.id });

        const ext = path.extname(req.file!.originalname) || ".mp4";
        const outputFilename = `enhanced_${job.id}${ext}`;
        const outputPath = path.join(ENHANCED_DIR, outputFilename);

        fs.copyFileSync(req.file!.path, outputPath);
        fs.unlinkSync(req.file!.path);

        await storage.updateJob(job.id, {
          status: "completed",
          enhancedUrl: `/api/enhanced/${outputFilename}`,
          completedAt: new Date().toISOString(),
          errorMessage: "Note: Full video frame-by-frame AI enhancement requires ffmpeg. This is a preview version.",
        });

        logger.info(MODULE, `Video job completed`, { jobId: job.id, outputFilename });
      } catch (err: any) {
        logger.error(MODULE, `Video job failed`, { jobId: job.id, error: err.message });
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

  app.get("/api/enhanced/:filename", (req: Request, res: Response) => {
    const filename = req.params.filename as string;
    const filePath = path.join(ENHANCED_DIR, filename);

    if (!fs.existsSync(filePath)) {
      logger.warn(MODULE, `Enhanced file not found`, { filename });
      return res.status(404).json({ message: "File not found" });
    }

    res.sendFile(filePath);
  });

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
