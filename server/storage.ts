import { type EnhancementJob } from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  getJob(id: string): Promise<EnhancementJob | undefined>;
  getAllJobs(): Promise<EnhancementJob[]>;
  createJob(data: Omit<EnhancementJob, "id" | "createdAt">): Promise<EnhancementJob>;
  updateJob(id: string, updates: Partial<EnhancementJob>): Promise<EnhancementJob | undefined>;
  deleteJob(id: string): Promise<boolean>;
}

export class MemStorage implements IStorage {
  private jobs: Map<string, EnhancementJob>;

  constructor() {
    this.jobs = new Map();
  }

  async getJob(id: string): Promise<EnhancementJob | undefined> {
    return this.jobs.get(id);
  }

  async getAllJobs(): Promise<EnhancementJob[]> {
    return Array.from(this.jobs.values()).sort(
      (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );
  }

  async createJob(data: Omit<EnhancementJob, "id" | "createdAt">): Promise<EnhancementJob> {
    const id = randomUUID();
    const job: EnhancementJob = {
      ...data,
      id,
      createdAt: new Date().toISOString(),
    };
    this.jobs.set(id, job);
    return job;
  }

  async updateJob(id: string, updates: Partial<EnhancementJob>): Promise<EnhancementJob | undefined> {
    const job = this.jobs.get(id);
    if (!job) return undefined;
    const updated = { ...job, ...updates };
    this.jobs.set(id, updated);
    return updated;
  }

  async deleteJob(id: string): Promise<boolean> {
    return this.jobs.delete(id);
  }
}

export const storage = new MemStorage();
