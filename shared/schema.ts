import { z } from "zod";

export const enhancementJobSchema = z.object({
  id: z.string(),
  type: z.enum(["image", "video"]),
  status: z.enum(["pending", "processing", "completed", "failed"]),
  model: z.string(),
  scale: z.number(),
  originalName: z.string(),
  originalSize: z.number(),
  enhancedUrl: z.string().nullable(),
  errorMessage: z.string().nullable(),
  createdAt: z.string(),
  completedAt: z.string().nullable(),
});

export type EnhancementJob = z.infer<typeof enhancementJobSchema>;

export const createJobSchema = z.object({
  type: z.enum(["image", "video"]),
  model: z.string(),
  scale: z.number(),
});

export type CreateJob = z.infer<typeof createJobSchema>;
