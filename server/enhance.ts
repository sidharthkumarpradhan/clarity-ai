import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";
import sharp from "sharp";
import { logger } from "./logger";

const MODULE = "enhance";

const HF_MODELS: Record<string, string> = {
  "swin2sr-classical-x2": "Xenova/swin2SR-classical-sr-x2-64",
  "swin2sr-realworld-x4": "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
  "swin2sr-compressed-x4": "Xenova/swin2SR-compressed-sr-x4-48",
};

export const IMAGE_MODELS = [
  {
    id: "swin2sr-classical-x2",
    name: "Swin2SR x2 (Classical)",
    description: "Fast classical super-resolution, 2x upscale",
    scale: 2,
    type: "image" as const,
    speed: "Fast",
  },
  {
    id: "swin2sr-realworld-x4",
    name: "Swin2SR x4 (Real-World)",
    description: "Real-world image restoration & 4x super-resolution",
    scale: 4,
    type: "image" as const,
    speed: "Medium",
  },
  {
    id: "swin2sr-compressed-x4",
    name: "Swin2SR x4 (Compressed)",
    description: "Compressed artifact removal + 4x upscale",
    scale: 4,
    type: "image" as const,
    speed: "Medium",
  },
];

export const VIDEO_MODELS = [
  {
    id: "swin2sr-realworld-x4",
    name: "Swin2SR x4 (frame-by-frame)",
    description: "Process each video frame for 4x enhancement",
    scale: 4,
    type: "video" as const,
    speed: "Slow",
  },
];

let pipelineInstance: any = null;
let currentModelId: string | null = null;

async function getUpscaler(modelId: string) {
  const hfModelId = HF_MODELS[modelId];
  if (!hfModelId) {
    throw new Error(`Unknown model: ${modelId}`);
  }

  if (pipelineInstance && currentModelId === modelId) {
    logger.debug(MODULE, `Reusing cached pipeline`, { model: hfModelId });
    return pipelineInstance;
  }

  logger.info(MODULE, `Loading Transformers.js pipeline`, { model: hfModelId });
  const loadStart = Date.now();

  const { pipeline } = await import("@huggingface/transformers");

  try {
    pipelineInstance = await pipeline("image-to-image", hfModelId, {
      dtype: "fp32",
    });
  } catch (loadError: any) {
    logger.error(MODULE, `Failed to load model`, {
      model: hfModelId,
      error: loadError.message,
      elapsed: `${Date.now() - loadStart}ms`,
    });
    throw new Error(`Failed to load AI model "${hfModelId}": ${loadError.message}`);
  }
  currentModelId = modelId;

  logger.info(MODULE, `Pipeline loaded`, { model: hfModelId, elapsed: `${Date.now() - loadStart}ms` });
  return pipelineInstance;
}

export async function enhanceImageWithHF(
  inputPath: string,
  modelId: string,
  outputDir: string
): Promise<string> {
  const hfModelId = HF_MODELS[modelId] || modelId;
  logger.info(MODULE, `Starting image enhancement`, { inputPath, model: hfModelId, modelId });

  const imageBuffer = fs.readFileSync(inputPath);
  logger.info(MODULE, `Read input file`, { size: imageBuffer.length, path: inputPath });

  const metadata = await sharp(imageBuffer).metadata();
  logger.info(MODULE, `Input image metadata`, {
    width: metadata.width,
    height: metadata.height,
    format: metadata.format,
    channels: metadata.channels,
  });

  const upscaler = await getUpscaler(modelId);

  logger.info(MODULE, `Running super-resolution inference`, { model: hfModelId });
  const inferStart = Date.now();

  const result = await upscaler(inputPath);

  const inferElapsed = Date.now() - inferStart;
  logger.info(MODULE, `Inference completed`, { model: hfModelId, elapsed: `${inferElapsed}ms` });

  const outputFilename = `enhanced_${randomUUID()}.png`;
  const outputPath = path.join(outputDir, outputFilename);

  await result.save(outputPath);

  const outputStats = fs.statSync(outputPath);
  const outputMeta = await sharp(outputPath).metadata();

  logger.info(MODULE, `Image enhanced and saved`, {
    outputPath,
    outputFilename,
    inputSize: imageBuffer.length,
    outputSize: outputStats.size,
    inputDimensions: `${metadata.width}x${metadata.height}`,
    outputDimensions: `${outputMeta.width}x${outputMeta.height}`,
    totalElapsed: `${Date.now() - inferStart}ms`,
  });

  return outputFilename;
}
