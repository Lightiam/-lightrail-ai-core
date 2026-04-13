import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Rocket,
  Server,
  Cpu,
  Activity,
  RefreshCw,
  ArrowLeftRight,
  FileText,
  BarChart3,
  Play,
  RotateCcw,
  ChevronDown,
  Copy,
  Check,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  Shield,
  Layers,
  Zap,
} from "lucide-react";
import DashboardHeader from "@/components/dashboard/DashboardHeader";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { toast } from "sonner";

// ─── Types ────────────────────────────────────────────────────────────────────

type ServingMode = "wide-ep" | "disaggregated" | "standard";
type DeploymentStatus =
  | "pending"
  | "provisioning"
  | "running"
  | "degraded"
  | "failed"
  | "rolling-back"
  | "terminated";

interface ObservabilityConfig {
  prometheus_enabled: boolean;
  tracing_enabled: boolean;
  log_level: "debug" | "info" | "warn" | "error";
}

interface DeploymentConfig {
  model: string;
  serving_mode: ServingMode;
  gpu_count: number;
  data_parallel_size: number;
  expert_parallelism: boolean;
  fault_tolerance: boolean;
  observability: ObservabilityConfig;
}

interface Deployment {
  id: string;
  name: string;
  status: DeploymentStatus;
  config: DeploymentConfig;
  replicas: number;
  created_at: string;
  updated_at: string;
  revision: number;
  latency_p50_ms: number | null;
  latency_p99_ms: number | null;
  throughput_tok_s: number | null;
}

interface LogEntry {
  ts: string;
  level: "debug" | "info" | "warn" | "error";
  deployment_id: string | null;
  msg: string;
}

interface MetricSeries {
  name: string;
  value: number;
  unit: string;
  bar_pct: number | null;
}

interface MetricsResponse {
  deployment_id: string | null;
  snapshot_at: string;
  series: MetricSeries[];
}

// ─── Constants ────────────────────────────────────────────────────────────────

const API_BASE = "/inference/llm-serving";

const availableModels = [
  { id: "llama-3.1-405b", name: "Llama 3.1 405B", params: "405B", recommended: true },
  { id: "llama-3.1-70b", name: "Llama 3.1 70B", params: "70B", recommended: false },
  { id: "mixtral-8x22b", name: "Mixtral 8x22B", params: "176B", recommended: false },
  { id: "deepseek-v2", name: "DeepSeek V2", params: "236B", recommended: false },
  { id: "qwen-72b", name: "Qwen 72B", params: "72B", recommended: false },
];

const statusConfig: Record<
  DeploymentStatus,
  { color: string; icon: React.ElementType; label: string }
> = {
  pending: { color: "bg-muted text-muted-foreground", icon: Clock, label: "Pending" },
  provisioning: {
    color: "bg-blue-500/15 text-blue-400 border-blue-500/30",
    icon: Loader2,
    label: "Provisioning",
  },
  running: {
    color: "bg-primary/15 text-primary border-primary/30",
    icon: CheckCircle2,
    label: "Running",
  },
  degraded: {
    color: "bg-yellow-500/15 text-yellow-400 border-yellow-500/30",
    icon: AlertTriangle,
    label: "Degraded",
  },
  failed: {
    color: "bg-destructive/15 text-destructive border-destructive/30",
    icon: XCircle,
    label: "Failed",
  },
  "rolling-back": {
    color: "bg-orange-500/15 text-orange-400 border-orange-500/30",
    icon: RotateCcw,
    label: "Rolling Back",
  },
  terminated: { color: "bg-muted text-muted-foreground", icon: XCircle, label: "Terminated" },
};

const servingModes: { id: ServingMode; label: string; description: string }[] = [
  {
    id: "wide-ep",
    label: "Wide-EP",
    description: "Expert parallelism across all GPUs for MoE models",
  },
  {
    id: "disaggregated",
    label: "Disaggregated Prefill/Decode",
    description: "Separate prefill and decode stages for optimal throughput",
  },
  {
    id: "standard",
    label: "Standard Serving",
    description: "Traditional tensor-parallel serving with auto-batching",
  },
];

const logLevelColor: Record<string, string> = {
  debug: "text-muted-foreground",
  info: "text-primary",
  warn: "text-yellow-400",
  error: "text-destructive",
};

// ─── API helpers ─────────────────────────────────────────────────────────────

async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ─── Component ────────────────────────────────────────────────────────────────

const LLMServing = () => {
  // Deployment list
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [loadingDeployments, setLoadingDeployments] = useState(true);

  // Metrics
  const [metrics, setMetrics] = useState<MetricSeries[]>([]);
  const [loadingMetrics, setLoadingMetrics] = useState(true);

  // Logs
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loadingLogs, setLoadingLogs] = useState(true);

  // Create deployment form
  const [selectedModel, setSelectedModel] = useState("llama-3.1-405b");
  const [deploymentName, setDeploymentName] = useState("");
  const [servingMode, setServingMode] = useState<ServingMode>("wide-ep");
  const [gpuCount, setGpuCount] = useState([8]);
  const [dpSize, setDpSize] = useState([2]);
  const [expertParallel, setExpertParallel] = useState(true);
  const [faultTolerance, setFaultTolerance] = useState(true);
  const [deploying, setDeploying] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);

  // Scale dialog
  const [scaleDialogOpen, setScaleDialogOpen] = useState(false);
  const [selectedDeployment, setSelectedDeployment] = useState<Deployment | null>(null);
  const [scaleReplicas, setScaleReplicas] = useState([2]);
  const [scaling, setScaling] = useState(false);

  // ─── Data fetchers ────────────────────────────────────────────────────────

  const fetchDeployments = useCallback(async () => {
    setLoadingDeployments(true);
    try {
      const data = await apiFetch<{ deployments: Deployment[]; total: number }>("");
      setDeployments(data.deployments);
    } catch (err) {
      toast.error(`Failed to load deployments: ${(err as Error).message}`);
    } finally {
      setLoadingDeployments(false);
    }
  }, []);

  const fetchMetrics = useCallback(async () => {
    setLoadingMetrics(true);
    try {
      const data = await apiFetch<MetricsResponse>("/metrics");
      setMetrics(data.series);
    } catch (err) {
      toast.error(`Failed to load metrics: ${(err as Error).message}`);
    } finally {
      setLoadingMetrics(false);
    }
  }, []);

  const fetchLogs = useCallback(async () => {
    setLoadingLogs(true);
    try {
      const data = await apiFetch<LogEntry[]>("/logs?limit=50");
      setLogs(data);
    } catch (err) {
      toast.error(`Failed to load logs: ${(err as Error).message}`);
    } finally {
      setLoadingLogs(false);
    }
  }, []);

  useEffect(() => {
    fetchDeployments();
    fetchMetrics();
    fetchLogs();
  }, [fetchDeployments, fetchMetrics, fetchLogs]);

  // ─── Actions ──────────────────────────────────────────────────────────────

  const handleDeploy = async () => {
    if (!deploymentName.trim()) {
      toast.error("Deployment name is required");
      return;
    }
    setDeploying(true);
    try {
      await apiFetch<Deployment>("/deploy", {
        method: "POST",
        body: JSON.stringify({
          name: deploymentName.trim(),
          model: selectedModel,
          serving_mode: servingMode,
          gpu_count: gpuCount[0],
          data_parallel_size: dpSize[0],
          expert_parallelism: expertParallel,
          fault_tolerance: faultTolerance,
        }),
      });
      toast.success("Deployment submitted — provisioning GPUs");
      setCreateDialogOpen(false);
      setDeploymentName("");
      await fetchDeployments();
      await fetchLogs();
    } catch (err) {
      toast.error(`Deploy failed: ${(err as Error).message}`);
    } finally {
      setDeploying(false);
    }
  };

  const handleScale = async () => {
    if (!selectedDeployment) return;
    setScaling(true);
    try {
      await apiFetch("/scale", {
        method: "POST",
        body: JSON.stringify({
          deployment_id: selectedDeployment.id,
          replicas: scaleReplicas[0],
        }),
      });
      toast.success(`Scaled ${selectedDeployment.name} to ${scaleReplicas[0]} replicas`);
      setScaleDialogOpen(false);
      await fetchDeployments();
    } catch (err) {
      toast.error(`Scale failed: ${(err as Error).message}`);
    } finally {
      setScaling(false);
    }
  };

  const handleRestart = async (dep: Deployment) => {
    try {
      await apiFetch("/restart", {
        method: "POST",
        body: JSON.stringify({ deployment_id: dep.id }),
      });
      toast.success(`Restart triggered for ${dep.name}`);
      await fetchDeployments();
      await fetchLogs();
    } catch (err) {
      toast.error(`Restart failed: ${(err as Error).message}`);
    }
  };

  const handleRollback = async (dep: Deployment) => {
    try {
      await apiFetch("/rollback", {
        method: "POST",
        body: JSON.stringify({ deployment_id: dep.id }),
      });
      toast.success(`Rollback initiated for ${dep.name}`);
      await fetchDeployments();
      await fetchLogs();
    } catch (err) {
      toast.error(`Rollback failed: ${(err as Error).message}`);
    }
  };

  const handleRefresh = () => {
    fetchDeployments();
    fetchMetrics();
    fetchLogs();
    toast.info("Refreshing deployments…");
  };

  // ─── Derived stats ────────────────────────────────────────────────────────

  const running = deployments.filter((d) => d.status === "running");
  const totalReplicas = deployments.reduce((a, d) => a + d.replicas, 0);
  const totalGpus = deployments.reduce((a, d) => a + d.config.gpu_count * d.replicas, 0);
  const avgP50 =
    running.length > 0
      ? Math.round(
          running.reduce((a, d) => a + (d.latency_p50_ms ?? 0), 0) / running.length
        )
      : 0;

  const overviewStats = [
    {
      label: "Active Deployments",
      value: running.length.toString(),
      icon: Server,
      accent: "text-primary",
    },
    {
      label: "Total Replicas",
      value: totalReplicas.toString(),
      icon: Layers,
      accent: "text-blue-400",
    },
    {
      label: "GPU Allocated",
      value: `${totalGpus}x H100`,
      icon: Cpu,
      accent: "text-purple-400",
    },
    {
      label: "Avg Latency (P50)",
      value: running.length > 0 ? `${avgP50}ms` : "—",
      icon: Zap,
      accent: "text-yellow-400",
    },
  ];

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader
        title="LLM Serving"
        subtitle="Deploy and manage Ray Serve inference workloads"
      />

      <div className="p-6 space-y-6">
        {/* Action Bar */}
        <div className="flex flex-wrap items-center gap-3">
          <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button className="font-mono gap-2">
                <Rocket className="w-4 h-4" />
                Create Deployment
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-xl">
              <DialogHeader>
                <DialogTitle className="font-mono">New LLM Deployment</DialogTitle>
                <DialogDescription>
                  Configure and deploy a new inference serving endpoint.
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-5 py-4">
                {/* Deployment name */}
                <div className="space-y-2">
                  <Label className="text-xs font-mono uppercase tracking-wider text-muted-foreground">
                    Deployment Name
                  </Label>
                  <input
                    className="w-full rounded-md border border-border bg-secondary px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-primary"
                    placeholder="e.g. prod-llama-70b"
                    value={deploymentName}
                    onChange={(e) => setDeploymentName(e.target.value)}
                  />
                </div>

                {/* Model Selection */}
                <div className="space-y-2">
                  <Label className="text-xs font-mono uppercase tracking-wider text-muted-foreground">
                    Model
                  </Label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger className="bg-secondary border-border font-mono">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {availableModels.map((m) => (
                        <SelectItem key={m.id} value={m.id} className="font-mono">
                          <div className="flex items-center gap-2">
                            <span>{m.name}</span>
                            <Badge variant="outline" className="text-[10px] px-1.5">
                              {m.params}
                            </Badge>
                            {m.recommended && (
                              <Badge className="text-[10px] px-1.5 bg-primary/15 text-primary border-primary/30">
                                Recommended
                              </Badge>
                            )}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Serving Mode */}
                <div className="space-y-2">
                  <Label className="text-xs font-mono uppercase tracking-wider text-muted-foreground">
                    Serving Mode
                  </Label>
                  <div className="grid grid-cols-1 gap-2">
                    {servingModes.map((mode) => (
                      <button
                        key={mode.id}
                        onClick={() => setServingMode(mode.id)}
                        className={`text-left p-3 rounded-lg border transition-all ${
                          servingMode === mode.id
                            ? "border-primary bg-primary/10 text-foreground"
                            : "border-border bg-secondary/50 text-muted-foreground hover:border-muted-foreground"
                        }`}
                      >
                        <div className="font-mono text-sm font-medium">{mode.label}</div>
                        <div className="text-xs mt-0.5 opacity-80">{mode.description}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* GPU & Parallelism */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label className="text-xs font-mono uppercase tracking-wider text-muted-foreground">
                      GPU Count: {gpuCount[0]}
                    </Label>
                    <Slider
                      value={gpuCount}
                      onValueChange={setGpuCount}
                      min={1}
                      max={16}
                      step={1}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-xs font-mono uppercase tracking-wider text-muted-foreground">
                      Data Parallel Size: {dpSize[0]}
                    </Label>
                    <Slider value={dpSize} onValueChange={setDpSize} min={1} max={8} step={1} />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center justify-between p-3 rounded-lg border border-border bg-secondary/50">
                    <div>
                      <Label className="text-sm font-mono">Expert Parallel</Label>
                      <p className="text-xs text-muted-foreground">MoE expert sharding</p>
                    </div>
                    <Switch checked={expertParallel} onCheckedChange={setExpertParallel} />
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg border border-border bg-secondary/50">
                    <div>
                      <Label className="text-sm font-mono">Fault Tolerance</Label>
                      <p className="text-xs text-muted-foreground">Auto-recovery on failure</p>
                    </div>
                    <Switch checked={faultTolerance} onCheckedChange={setFaultTolerance} />
                  </div>
                </div>
              </div>

              <DialogFooter>
                <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
                  Cancel
                </Button>
                <Button
                  onClick={handleDeploy}
                  disabled={deploying || !deploymentName.trim()}
                  className="font-mono gap-2"
                >
                  {deploying ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Rocket className="w-4 h-4" />
                  )}
                  {deploying ? "Deploying…" : "Deploy"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <Button variant="outline" className="font-mono gap-2" onClick={handleRefresh}>
            <RefreshCw className="w-4 h-4" />
            Refresh
          </Button>
        </div>

        {/* Deployment Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {overviewStats.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              <Card className="bg-card/50 border-border">
                <CardContent className="p-4 flex items-center gap-4">
                  {loadingDeployments ? (
                    <Skeleton className="h-12 w-full" />
                  ) : (
                    <>
                      <div className={`p-2 rounded-lg bg-secondary ${stat.accent}`}>
                        <stat.icon className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground font-mono uppercase tracking-wider">
                          {stat.label}
                        </p>
                        <p className="text-xl font-bold font-mono text-foreground">{stat.value}</p>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Deployments Table */}
        <Card className="bg-card/50 border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-mono flex items-center gap-2">
              <Server className="w-4 h-4 text-primary" />
              Deployments
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-xs font-mono text-muted-foreground uppercase tracking-wider">
                    <th className="text-left p-4">Name</th>
                    <th className="text-left p-4">Model</th>
                    <th className="text-left p-4">Mode</th>
                    <th className="text-left p-4">Status</th>
                    <th className="text-left p-4">GPUs</th>
                    <th className="text-left p-4">Replicas</th>
                    <th className="text-left p-4">P50 / P99</th>
                    <th className="text-left p-4">Throughput</th>
                    <th className="text-right p-4">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {loadingDeployments ? (
                    Array.from({ length: 3 }).map((_, i) => (
                      <tr key={i} className="border-b border-border/50">
                        <td colSpan={9} className="p-4">
                          <Skeleton className="h-5 w-full" />
                        </td>
                      </tr>
                    ))
                  ) : deployments.length === 0 ? (
                    <tr>
                      <td
                        colSpan={9}
                        className="p-8 text-center text-muted-foreground font-mono text-sm"
                      >
                        No deployments yet — create one to get started.
                      </td>
                    </tr>
                  ) : (
                    deployments.map((dep) => {
                      const sc = statusConfig[dep.status];
                      const StatusIcon = sc.icon;
                      return (
                        <tr
                          key={dep.id}
                          className="border-b border-border/50 hover:bg-secondary/30 transition-colors"
                        >
                          <td className="p-4 font-mono font-medium text-foreground">{dep.name}</td>
                          <td className="p-4 text-muted-foreground">{dep.config.model}</td>
                          <td className="p-4">
                            <Badge variant="outline" className="font-mono text-[10px]">
                              {servingModes.find((m) => m.id === dep.config.serving_mode)?.label}
                            </Badge>
                          </td>
                          <td className="p-4">
                            <Badge className={`gap-1 font-mono text-[10px] border ${sc.color}`}>
                              <StatusIcon
                                className={`w-3 h-3 ${
                                  dep.status === "provisioning" ? "animate-spin" : ""
                                }`}
                              />
                              {sc.label}
                            </Badge>
                          </td>
                          <td className="p-4 font-mono text-muted-foreground">
                            {dep.config.gpu_count}
                          </td>
                          <td className="p-4 font-mono text-muted-foreground">{dep.replicas}</td>
                          <td className="p-4 font-mono text-muted-foreground">
                            {dep.latency_p50_ms != null
                              ? `${dep.latency_p50_ms}ms / ${dep.latency_p99_ms}ms`
                              : "—"}
                          </td>
                          <td className="p-4 font-mono text-muted-foreground">
                            {dep.throughput_tok_s != null
                              ? `${dep.throughput_tok_s} tok/s`
                              : "—"}
                          </td>
                          <td className="p-4 text-right">
                            <div className="flex items-center justify-end gap-1">
                              <Button
                                size="sm"
                                variant="ghost"
                                className="h-7 px-2 font-mono text-xs"
                                title="Scale"
                                onClick={() => {
                                  setSelectedDeployment(dep);
                                  setScaleReplicas([dep.replicas]);
                                  setScaleDialogOpen(true);
                                }}
                              >
                                <ArrowLeftRight className="w-3.5 h-3.5" />
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                className="h-7 px-2 font-mono text-xs"
                                title="Restart"
                                onClick={() => handleRestart(dep)}
                              >
                                <RefreshCw className="w-3.5 h-3.5" />
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                className="h-7 px-2 font-mono text-xs"
                                title="Rollback"
                                onClick={() => handleRollback(dep)}
                              >
                                <RotateCcw className="w-3.5 h-3.5" />
                              </Button>
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Metrics + Logs */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Metrics Panel */}
          <Card className="bg-card/50 border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-mono flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-primary" />
                Serving Metrics
              </CardTitle>
              <CardDescription className="font-mono text-xs">
                Real-time inference performance
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {loadingMetrics ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-6 w-full" />
                ))
              ) : (
                metrics.map((metric, i) => {
                  const barColors = [
                    "bg-primary",
                    "bg-blue-500",
                    "bg-primary",
                    "bg-yellow-500",
                    "bg-purple-500",
                    "bg-blue-400",
                  ];
                  return (
                    <div key={metric.name} className="space-y-1.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-mono text-muted-foreground">{metric.name}</span>
                        <span className="font-mono font-medium text-foreground">
                          {metric.value.toLocaleString()} {metric.unit}
                        </span>
                      </div>
                      {metric.bar_pct != null && (
                        <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${metric.bar_pct}%` }}
                            transition={{ duration: 1, ease: "easeOut" }}
                            className={`h-full rounded-full ${barColors[i % barColors.length]}`}
                          />
                        </div>
                      )}
                    </div>
                  );
                })
              )}
              <Button
                variant="outline"
                className="w-full mt-2 font-mono gap-2 text-xs"
                onClick={() => {
                  fetchMetrics();
                  toast.info("Metrics refreshed");
                }}
              >
                <BarChart3 className="w-3.5 h-3.5" />
                Refresh Metrics
              </Button>
            </CardContent>
          </Card>

          {/* Logs Panel */}
          <Card className="bg-card/50 border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-mono flex items-center gap-2">
                <FileText className="w-4 h-4 text-primary" />
                Deployment Logs
              </CardTitle>
              <CardDescription className="font-mono text-xs">
                Live streaming from Ray Serve controller
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-background rounded-lg border border-border p-3 h-[280px] overflow-y-auto font-mono text-xs space-y-1">
                {loadingLogs ? (
                  Array.from({ length: 6 }).map((_, i) => (
                    <Skeleton key={i} className="h-4 w-full" />
                  ))
                ) : logs.length === 0 ? (
                  <p className="text-muted-foreground text-center py-8">No logs yet.</p>
                ) : (
                  logs.map((log, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: Math.min(i * 0.03, 0.4) }}
                      className="flex gap-2 leading-relaxed"
                    >
                      <span className="text-muted-foreground shrink-0">{log.ts}</span>
                      <span
                        className={`shrink-0 uppercase font-bold ${
                          logLevelColor[log.level] || "text-muted-foreground"
                        }`}
                      >
                        [{log.level}]
                      </span>
                      <span className="text-foreground/80">{log.msg}</span>
                    </motion.div>
                  ))
                )}
              </div>
              <Button
                variant="outline"
                className="w-full mt-3 font-mono gap-2 text-xs"
                onClick={() => {
                  fetchLogs();
                  toast.info("Logs refreshed");
                }}
              >
                <FileText className="w-3.5 h-3.5" />
                Refresh Logs
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Scale Dialog */}
        <Dialog open={scaleDialogOpen} onOpenChange={setScaleDialogOpen}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle className="font-mono">Scale Deployment</DialogTitle>
              <DialogDescription>
                Adjust replicas for{" "}
                <span className="font-mono text-foreground">{selectedDeployment?.name}</span>
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label className="text-xs font-mono uppercase tracking-wider text-muted-foreground">
                  Target Replicas: {scaleReplicas[0]}
                </Label>
                <Slider
                  value={scaleReplicas}
                  onValueChange={setScaleReplicas}
                  min={0}
                  max={16}
                  step={1}
                />
              </div>
              <div className="p-3 rounded-lg border border-border bg-secondary/50 text-xs font-mono text-muted-foreground">
                <p>
                  Current: {selectedDeployment?.replicas} replicas ×{" "}
                  {selectedDeployment?.config.gpu_count} GPUs
                </p>
                <p>
                  Target: {scaleReplicas[0]} replicas ×{" "}
                  {selectedDeployment?.config.gpu_count} GPUs ={" "}
                  {scaleReplicas[0] * (selectedDeployment?.config.gpu_count || 0)} total GPUs
                </p>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setScaleDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                className="font-mono gap-2"
                onClick={handleScale}
                disabled={scaling}
              >
                {scaling ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <ArrowLeftRight className="w-4 h-4" />
                )}
                {scaling ? "Scaling…" : "Apply Scale"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
};

export default LLMServing;
