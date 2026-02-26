import { useState, useRef, useCallback, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { 
  Upload, Sparkles, Image, Video, Trash2, Download, 
  CheckCircle, XCircle, Clock, Loader2,
  Zap, Info, RotateCcw, Eye, EyeOff, GraduationCap, Users
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import type { EnhancementJob } from "@shared/schema";
import fairfieldLogo from "@assets/fairfieldUniversityLogo_1772082834920.png";

interface ModelInfo {
  id: string;
  name: string;
  description: string;
  scale: number;
  type: "image" | "video";
  speed: string;
}

interface ModelsResponse {
  image: ModelInfo[];
  video: ModelInfo[];
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
}

function StatusBadge({ status }: { status: EnhancementJob["status"] }) {
  const configs = {
    pending: { label: "Queued", icon: Clock, className: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400" },
    processing: { label: "Processing", icon: Loader2, className: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400" },
    completed: { label: "Done", icon: CheckCircle, className: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400" },
    failed: { label: "Failed", icon: XCircle, className: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400" },
  };
  const cfg = configs[status];
  const Icon = cfg.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium ${cfg.className}`}>
      <Icon className={`w-3 h-3 ${status === "processing" ? "animate-spin" : ""}`} />
      {cfg.label}
    </span>
  );
}

function JobCard({ job, onDelete }: { job: EnhancementJob; onDelete: (id: string) => void }) {
  const [showOriginal, setShowOriginal] = useState(false);
  const isImage = job.type === "image";

  return (
    <Card className="p-4 gap-3 flex flex-col border-card-border transition-all duration-200">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className={`p-1.5 rounded-md shrink-0 ${isImage ? "bg-primary/10" : "bg-purple-500/10"}`}>
            {isImage ? (
              <Image className="w-4 h-4 text-primary" />
            ) : (
              <Video className="w-4 h-4 text-purple-500" />
            )}
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium truncate" data-testid={`text-job-name-${job.id}`}>
              {job.originalName}
            </p>
            <p className="text-xs text-muted-foreground">
              {formatFileSize(job.originalSize)} · {formatDate(job.createdAt)}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <StatusBadge status={job.status} />
          <Button
            size="icon"
            variant="ghost"
            onClick={() => onDelete(job.id)}
            data-testid={`button-delete-job-${job.id}`}
            className="text-muted-foreground"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Badge variant="secondary" className="text-xs">
          <Zap className="w-3 h-3 mr-1" />
          {job.model.split("/")[1] || job.model}
        </Badge>
        <Badge variant="secondary" className="text-xs">
          {job.scale}x Upscale
        </Badge>
      </div>

      {job.status === "processing" && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
            <span className="text-xs text-muted-foreground">Enhancing with AI... this may take 30-120 seconds</span>
          </div>
          <Progress value={undefined} className="h-1.5" />
        </div>
      )}

      {job.status === "pending" && (
        <div className="flex items-center gap-2">
          <Clock className="w-3.5 h-3.5 text-yellow-500" />
          <span className="text-xs text-muted-foreground">Waiting to start...</span>
        </div>
      )}

      {job.status === "completed" && job.enhancedUrl && isImage && (
        <div className="space-y-2">
          <div className="relative rounded-md bg-muted aspect-video flex items-center justify-center overflow-hidden">
            {!showOriginal ? (
              <img
                src={job.enhancedUrl}
                alt="Enhanced"
                className="max-w-full max-h-48 object-contain rounded-md"
              />
            ) : (
              <div className="flex flex-col items-center gap-2 text-muted-foreground">
                <Eye className="w-8 h-8" />
                <span className="text-xs">Original not stored server-side</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              size="sm"
              variant="outline"
              onClick={() => setShowOriginal(!showOriginal)}
              data-testid={`button-toggle-preview-${job.id}`}
            >
              {showOriginal ? <EyeOff className="w-3.5 h-3.5 mr-1.5" /> : <Eye className="w-3.5 h-3.5 mr-1.5" />}
              {showOriginal ? "Show Enhanced" : "Toggle View"}
            </Button>
            <Button
              size="sm"
              asChild
              data-testid={`button-download-${job.id}`}
            >
              <a href={job.enhancedUrl} download>
                <Download className="w-3.5 h-3.5 mr-1.5" />
                Download
              </a>
            </Button>
          </div>
        </div>
      )}

      {job.status === "completed" && job.enhancedUrl && !isImage && (
        <div className="space-y-2">
          {job.errorMessage && (
            <p className="text-xs text-muted-foreground bg-muted/50 rounded-md p-2">
              <Info className="w-3 h-3 inline mr-1" />
              {job.errorMessage}
            </p>
          )}
          <Button size="sm" asChild data-testid={`button-download-video-${job.id}`}>
            <a href={job.enhancedUrl} download>
              <Download className="w-3.5 h-3.5 mr-1.5" />
              Download Video
            </a>
          </Button>
        </div>
      )}

      {job.status === "failed" && (
        <div className="bg-destructive/10 border border-destructive/20 rounded-md p-3">
          <p className="text-xs text-destructive">
            <XCircle className="w-3.5 h-3.5 inline mr-1.5" />
            {job.errorMessage || "Enhancement failed. Please try again."}
          </p>
        </div>
      )}
    </Card>
  );
}

function DropZone({
  type,
  onFile,
  disabled,
}: {
  type: "image" | "video";
  onFile: (file: File) => void;
  disabled: boolean;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = type === "image"
    ? "image/jpeg,image/jpg,image/png,image/webp,image/bmp"
    : "video/mp4,video/webm,video/avi,video/quicktime,image/gif";

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      data-testid={`dropzone-${type}`}
      className={`
        relative flex flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed 
        transition-all duration-200 cursor-pointer select-none p-10
        ${isDragging
          ? "border-primary bg-primary/5 scale-[1.01]"
          : "border-border hover:border-primary/50 hover:bg-muted/30"
        }
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        disabled={disabled}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }}
        data-testid={`input-file-${type}`}
      />

      <div className={`p-4 rounded-2xl transition-transform duration-200 ${isDragging ? "scale-110" : ""} ${type === "image" ? "bg-primary/10" : "bg-purple-500/10"}`}>
        {type === "image" ? (
          <Image className="w-10 h-10 text-primary" />
        ) : (
          <Video className="w-10 h-10 text-purple-500" />
        )}
      </div>

      <div className="text-center">
        <p className="font-semibold text-foreground">
          Drop your {type} here
        </p>
        <p className="text-sm text-muted-foreground mt-1">
          or <span className="text-primary font-medium">click to browse</span>
        </p>
        <p className="text-xs text-muted-foreground mt-2">
          {type === "image"
            ? "JPEG, PNG, WebP, BMP · Max 50MB"
            : "MP4, WebM, AVI, MOV, GIF · Max 50MB"}
        </p>
      </div>
    </div>
  );
}

function SelectedFileCard({ file, onRemove }: { file: File; onRemove: () => void }) {
  const [preview, setPreview] = useState<string | null>(null);
  const isImage = file.type.startsWith("image/");

  useEffect(() => {
    if (!isImage) return;
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);
  }, [file, isImage]);

  return (
    <div className="relative rounded-xl border border-border bg-card overflow-hidden transition-all duration-200">
      {preview && (
        <div className="aspect-video bg-muted flex items-center justify-center overflow-hidden">
          <img src={preview} alt="Preview" className="max-w-full max-h-full object-contain" />
        </div>
      )}
      <div className="flex items-center justify-between gap-3 p-3">
        <div className="min-w-0">
          <p className="text-sm font-medium truncate">{file.name}</p>
          <p className="text-xs text-muted-foreground">{formatFileSize(file.size)}</p>
        </div>
        <Button size="icon" variant="ghost" onClick={onRemove} data-testid="button-remove-file">
          <RotateCcw className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
}

export default function Home() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<"image" | "video">("image");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");

  const { data: models } = useQuery<ModelsResponse>({
    queryKey: ["/api/models"],
  });

  const { data: jobs = [], isLoading: jobsLoading } = useQuery<EnhancementJob[]>({
    queryKey: ["/api/jobs"],
    refetchInterval: (query) => {
      const data = query.state.data as EnhancementJob[] | undefined;
      const hasActive = data?.some((j) => j.status === "pending" || j.status === "processing");
      return hasActive ? 2000 : false;
    },
  });

  const availableModels = activeTab === "image" ? (models?.image ?? []) : (models?.video ?? []);
  const currentModel = availableModels.find((m) => m.id === selectedModel) ?? availableModels[0];

  const enhanceMutation = useMutation({
    mutationFn: async ({ file, model, type }: { file: File; model: string; type: "image" | "video" }) => {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model", model);
      const res = await apiRequest("POST", `/api/enhance/${type}`, formData);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/jobs"] });
      setSelectedFile(null);
      toast({
        title: "Enhancement started",
        description: "Your file is being processed. Results appear below when ready.",
      });
    },
    onError: (err: Error) => {
      toast({
        title: "Enhancement failed",
        description: err.message,
        variant: "destructive",
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest("DELETE", `/api/jobs/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/jobs"] });
    },
  });

  const handleEnhance = () => {
    if (!selectedFile) return;
    const modelId = selectedModel || currentModel?.id;
    if (!modelId) return;
    enhanceMutation.mutate({ file: selectedFile, model: modelId, type: activeTab });
  };

  const filteredJobs = jobs.filter((j) => j.type === activeTab);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header with University Logo */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img
              src={fairfieldLogo}
              alt="Fairfield University"
              className="h-12 w-auto"
              data-testid="img-university-logo"
            />
            <div className="h-8 w-px bg-border" />
            <div>
              <h1 className="text-lg font-bold text-foreground leading-none" data-testid="text-app-title">
                ClarityAI
              </h1>
              <p className="text-xs text-muted-foreground">AI-Powered Media Enhancement</p>
            </div>
          </div>
          <Badge variant="secondary" className="text-xs">
            <Sparkles className="w-3 h-3 mr-1" />
            MIRNet Model
          </Badge>
        </div>
      </header>

      {/* Hero Section with University Branding */}
      <section className="border-b border-border bg-gradient-to-b from-[#a01c2a]/5 via-[#a01c2a]/3 to-transparent">
        <div className="max-w-6xl mx-auto px-4 py-10 md:py-14">
          <div className="text-center">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-foreground tracking-tight leading-tight">
              Turn blurry into{" "}
              <span className="text-[#a01c2a] dark:text-red-400">crystal clear</span>
            </h2>
            <p className="mt-4 text-base md:text-lg text-muted-foreground max-w-2xl mx-auto">
              Upload your dark or low-light images and videos. Our MIRNet AI model will enhance them to stunning clarity.
            </p>

            {/* Stats Row */}
            <div className="flex items-center justify-center gap-6 sm:gap-10 mt-8 flex-wrap">
              {[
                { label: "MIRNet", sub: "Deep Learning Model" },
                { label: "Low-Light", sub: "Image Enhancement" },
                { label: "PyTorch", sub: "CPU Processing" },
              ].map((item) => (
                <div key={item.label} className="text-center">
                  <p className="text-lg sm:text-xl font-bold text-foreground">{item.label}</p>
                  <p className="text-xs text-muted-foreground">{item.sub}</p>
                </div>
              ))}
            </div>

            {/* Team Credits Card */}
            <div className="mt-8 inline-flex flex-col items-center">
              <Card className="inline-flex flex-col items-center gap-3 px-6 py-4 border-card-border bg-card/80 backdrop-blur-sm max-w-lg">
                <div className="flex items-center gap-2 text-sm text-foreground font-medium">
                  <Users className="w-4 h-4 text-[#a01c2a] dark:text-red-400" />
                  <span>Presented by</span>
                </div>
                <div className="flex flex-col items-center gap-1">
                  <p className="text-sm font-semibold text-foreground" data-testid="text-presenter-1">
                    Sidharth Kumar Pradhan
                  </p>
                  <p className="text-sm font-semibold text-foreground" data-testid="text-presenter-2">
                    Naqibahmed Kadri
                  </p>
                </div>
                <div className="w-full h-px bg-border" />
                <div className="flex flex-col items-center gap-0.5">
                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <GraduationCap className="w-3.5 h-3.5" />
                    <span>Guided by</span>
                  </div>
                  <p className="text-sm font-semibold text-foreground" data-testid="text-advisor">
                    Dr. Sidike Paheding
                  </p>
                  <p className="text-xs text-muted-foreground" data-testid="text-department">
                    School of Engineering & Computing, Fairfield University
                  </p>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8 flex-1 w-full">
        <Tabs value={activeTab} onValueChange={(v) => { setActiveTab(v as "image" | "video"); setSelectedFile(null); setSelectedModel(""); }}>
          <TabsList className="mb-6" data-testid="tabs-mode">
            <TabsTrigger value="image" data-testid="tab-image">
              <Image className="w-4 h-4 mr-2" />
              Images
            </TabsTrigger>
            <TabsTrigger value="video" data-testid="tab-video">
              <Video className="w-4 h-4 mr-2" />
              Videos
              <span className="ml-1 text-[10px] px-1.5 py-0.5 rounded bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400 font-medium leading-none">Beta</span>
            </TabsTrigger>
          </TabsList>

          {(["image", "video"] as const).map((tabType) => (
            <TabsContent key={tabType} value={tabType}>
              <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                {/* Upload Panel */}
                <div className="lg:col-span-2 space-y-4">
                  <div>
                    <h3 className="font-semibold text-foreground mb-1">Upload {tabType === "image" ? "Image" : "Video"}</h3>
                    <p className="text-sm text-muted-foreground">Choose a file to enhance with AI</p>
                  </div>

                  {selectedFile ? (
                    <SelectedFileCard file={selectedFile} onRemove={() => setSelectedFile(null)} />
                  ) : (
                    <DropZone type={tabType} onFile={setSelectedFile} disabled={enhanceMutation.isPending} />
                  )}

                  {availableModels.length > 0 && (
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground flex items-center gap-1.5">
                        AI Model
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button type="button" className="inline-flex">
                              <Info className="w-3.5 h-3.5 text-muted-foreground" />
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="max-w-xs text-xs">Different models are optimized for different types of content.</p>
                          </TooltipContent>
                        </Tooltip>
                      </label>
                      <Select
                        value={selectedModel || (availableModels[0]?.id ?? "")}
                        onValueChange={setSelectedModel}
                      >
                        <SelectTrigger data-testid="select-trigger-model">
                          <SelectValue placeholder="Choose a model" />
                        </SelectTrigger>
                        <SelectContent>
                          {availableModels.map((m) => (
                            <SelectItem key={m.id} value={m.id} data-testid={`option-model-${m.id}`}>
                              <span className="font-medium">{m.name}</span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>

                      {currentModel && (
                        <div className="space-y-1">
                          <p className="text-xs text-muted-foreground">{currentModel.description}</p>
                          <div className="flex flex-wrap gap-2">
                            <Badge variant="secondary" className="text-xs">
                              <Zap className="w-3 h-3 mr-1" />
                              {currentModel.speed} speed
                            </Badge>
                            <Badge variant="secondary" className="text-xs">
                              Enhancement
                            </Badge>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  <Button
                    className="w-full bg-[#a01c2a] hover:bg-[#861724] text-white no-default-hover-elevate"
                    size="lg"
                    onClick={handleEnhance}
                    disabled={!selectedFile || enhanceMutation.isPending}
                    data-testid="button-enhance"
                  >
                    {enhanceMutation.isPending ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Starting...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4 mr-2" />
                        Enhance {tabType === "image" ? "Image" : "Video"}
                      </>
                    )}
                  </Button>

                  {tabType === "video" && (
                    <div className="bg-muted/50 rounded-lg p-3 border border-border">
                      <p className="text-xs text-muted-foreground flex items-start gap-1.5">
                        <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                        <span>
                          Video support uploads your file and applies basic processing. For full frame-by-frame AI enhancement, install ffmpeg on the host machine.
                        </span>
                      </p>
                    </div>
                  )}
                </div>

                {/* Results Panel */}
                <div className="lg:col-span-3 space-y-4">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <h3 className="font-semibold text-foreground">Results</h3>
                      <p className="text-sm text-muted-foreground">
                        {filteredJobs.length === 0 ? "No enhancements yet" : `${filteredJobs.length} job${filteredJobs.length !== 1 ? "s" : ""}`}
                      </p>
                    </div>
                    {filteredJobs.length > 0 && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => queryClient.invalidateQueries({ queryKey: ["/api/jobs"] })}
                        data-testid="button-refresh"
                      >
                        <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
                        Refresh
                      </Button>
                    )}
                  </div>

                  {jobsLoading ? (
                    <div className="flex items-center justify-center py-12">
                      <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                    </div>
                  ) : filteredJobs.length === 0 ? (
                    <Card className="flex flex-col items-center justify-center py-16 gap-3 border-dashed border-card-border bg-card/50">
                      <div className="p-3 rounded-full bg-muted">
                        {tabType === "image" ? (
                          <Image className="w-8 h-8 text-muted-foreground" />
                        ) : (
                          <Video className="w-8 h-8 text-muted-foreground" />
                        )}
                      </div>
                      <p className="text-muted-foreground font-medium">No {tabType}s enhanced yet</p>
                      <p className="text-sm text-muted-foreground text-center max-w-xs">
                        Upload a {tabType} on the left and click "Enhance" to get started
                      </p>
                    </Card>
                  ) : (
                    <div className="space-y-3">
                      {filteredJobs.map((job) => (
                        <JobCard
                          key={job.id}
                          job={job}
                          onDelete={(id) => deleteMutation.mutate(id)}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </main>

      {/* How It Works Section */}
      <section className="border-t border-border bg-card/30">
        <div className="max-w-6xl mx-auto px-4 py-12">
          <h3 className="text-xl font-bold text-foreground text-center mb-8">How it works</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                step: "1",
                title: "Upload your file",
                desc: "Drag and drop or click to upload a blurry, dark, or low-res image or video.",
                icon: Upload,
                color: "bg-[#a01c2a]/10 text-[#a01c2a] dark:text-red-400",
              },
              {
                step: "2",
                title: "AI enhancement",
                desc: "MIRNet deep learning model enhances low-light and degraded images using multi-scale residual blocks.",
                icon: Sparkles,
                color: "bg-purple-500/10 text-purple-500",
              },
              {
                step: "3",
                title: "Download result",
                desc: "Get your enhanced file with improved brightness, contrast, and detail recovery.",
                icon: Download,
                color: "bg-green-500/10 text-green-600 dark:text-green-400",
              },
            ].map((item) => (
              <div key={item.step} className="flex flex-col items-center text-center gap-3 p-4">
                <div className={`p-3 rounded-2xl ${item.color}`}>
                  <item.icon className="w-7 h-7" />
                </div>
                <div>
                  <p className="font-semibold text-foreground">{item.title}</p>
                  <p className="text-sm text-muted-foreground mt-1">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-card">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img
                src={fairfieldLogo}
                alt="Fairfield University"
                className="h-10 w-auto"
              />
              <div className="h-6 w-px bg-border" />
              <span className="text-sm font-semibold text-foreground">ClarityAI</span>
            </div>
            <div className="text-center sm:text-right">
              <p className="text-xs text-muted-foreground">
                School of Engineering & Computing, Fairfield University
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">
                Powered by MIRNet via PyTorch
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
