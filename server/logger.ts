type LogLevel = "DEBUG" | "INFO" | "WARN" | "ERROR";

const COLORS: Record<LogLevel, string> = {
  DEBUG: "\x1b[36m",
  INFO: "\x1b[32m",
  WARN: "\x1b[33m",
  ERROR: "\x1b[31m",
};
const RESET = "\x1b[0m";

function formatTimestamp(): string {
  return new Date().toISOString();
}

function log(level: LogLevel, module: string, message: string, meta?: Record<string, unknown>) {
  const timestamp = formatTimestamp();
  const color = COLORS[level];
  const metaStr = meta ? ` ${JSON.stringify(meta)}` : "";
  console.log(`${color}[${timestamp}] [${level}] [${module}]${RESET} ${message}${metaStr}`);
}

export const logger = {
  debug: (module: string, message: string, meta?: Record<string, unknown>) =>
    log("DEBUG", module, message, meta),
  info: (module: string, message: string, meta?: Record<string, unknown>) =>
    log("INFO", module, message, meta),
  warn: (module: string, message: string, meta?: Record<string, unknown>) =>
    log("WARN", module, message, meta),
  error: (module: string, message: string, meta?: Record<string, unknown>) =>
    log("ERROR", module, message, meta),
};
