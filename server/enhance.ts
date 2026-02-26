import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";
import { spawn, ChildProcess } from "child_process";
import { createInterface, Interface } from "readline";
import { logger } from "./logger";

const MODULE = "enhance";

export const IMAGE_MODELS = [
  {
    id: "mirnet-low-light",
    name: "MIRNet (Low-Light Enhancement)",
    description: "Enhances dark, low-light images using Multi-scale Residual Block architecture",
    scale: 1,
    type: "image" as const,
    speed: "Medium",
  },
];

export const VIDEO_MODELS = [
  {
    id: "mirnet-low-light",
    name: "MIRNet (frame-by-frame)",
    description: "Process each video frame for low-light enhancement",
    scale: 1,
    type: "video" as const,
    speed: "Slow",
  },
];

let pythonProcess: ChildProcess | null = null;
let pythonRL: Interface | null = null;
let isReady = false;
let pendingResolve: ((value: any) => void) | null = null;
let pendingReject: ((reason: any) => void) | null = null;
let startingUp = false;

function startPythonServer(): Promise<void> {
  if (startingUp) {
    return new Promise((resolve) => {
      const check = setInterval(() => {
        if (isReady) { clearInterval(check); resolve(); }
      }, 500);
    });
  }

  if (isReady && pythonProcess && !pythonProcess.killed) {
    return Promise.resolve();
  }

  startingUp = true;
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "server", "mirnet_server.py");

    logger.info(MODULE, "Starting persistent MIRNet Python server", { script: scriptPath });

    pythonProcess = spawn("python3", ["-u", scriptPath], {
      cwd: process.cwd(),
      env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1", PYTHONUNBUFFERED: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    });

    pythonProcess.stderr?.on("data", (data: Buffer) => {
      const lines = data.toString().trim().split("\n");
      for (const line of lines) {
        if (line.trim()) {
          logger.info(MODULE, `[python] ${line.trim()}`);
        }
      }
    });

    pythonRL = createInterface({ input: pythonProcess.stdout! });

    pythonRL.on("line", (line: string) => {
      try {
        const data = JSON.parse(line.trim());

        if (data.status === "ready") {
          logger.info(MODULE, "MIRNet Python server is ready (model cached)");
          isReady = true;
          startingUp = false;
          resolve();
          return;
        }

        if (pendingResolve) {
          const res = pendingResolve;
          const rej = pendingReject;
          pendingResolve = null;
          pendingReject = null;

          if (data.success) {
            res(data);
          } else {
            rej!(new Error(data.error || "Unknown inference error"));
          }
        }
      } catch (e: any) {
        logger.error(MODULE, "Failed to parse Python output", { line: line.slice(0, 300) });
        if (pendingReject) {
          const rej = pendingReject;
          pendingResolve = null;
          pendingReject = null;
          rej(new Error("Failed to parse inference result"));
        }
      }
    });

    pythonProcess.on("close", (code: number | null) => {
      logger.warn(MODULE, "Python server process exited", { code });
      isReady = false;
      startingUp = false;
      pythonProcess = null;
      pythonRL = null;

      if (pendingReject) {
        const rej = pendingReject;
        pendingResolve = null;
        pendingReject = null;
        rej(new Error(`Python server exited with code ${code}`));
      }
    });

    pythonProcess.on("error", (err: Error) => {
      logger.error(MODULE, "Failed to start Python server", { error: err.message });
      isReady = false;
      startingUp = false;
      reject(err);
    });
  });
}

function sendInferenceRequest(inputPath: string, outputPath: string): Promise<any> {
  return new Promise((resolve, reject) => {
    if (!pythonProcess || !pythonProcess.stdin || pythonProcess.killed) {
      reject(new Error("Python server is not running"));
      return;
    }

    pendingResolve = resolve;
    pendingReject = reject;

    const request = JSON.stringify({ input_path: inputPath, output_path: outputPath }) + "\n";
    pythonProcess.stdin.write(request);
  });
}

export async function enhanceImageWithHF(
  inputPath: string,
  _modelId: string,
  outputDir: string
): Promise<string> {
  logger.info(MODULE, "Starting MIRNet image enhancement", { inputPath });

  const imageBuffer = fs.readFileSync(inputPath);
  logger.info(MODULE, "Read input file", { size: imageBuffer.length, path: inputPath });

  await startPythonServer();

  const outputFilename = `enhanced_${randomUUID()}.png`;
  const outputPath = path.join(outputDir, outputFilename);

  const startTime = Date.now();

  const result = await sendInferenceRequest(inputPath, outputPath);

  logger.info(MODULE, "MIRNet enhancement completed", {
    outputFilename,
    inputSize: imageBuffer.length,
    outputSize: result.output_size,
    inputDimensions: result.input_dimensions,
    inferenceTime: `${result.inference_time}s`,
    totalTime: `${result.total_time}s`,
    wallTime: `${Date.now() - startTime}ms`,
  });

  return outputFilename;
}
