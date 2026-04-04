"""
LightOS — LLM Serving (Ray Serve + vLLM) backend module
========================================================

Endpoints
---------
GET  /inference/llm-serving              — list all deployments
POST /inference/llm-serving/deploy       — create a new deployment
POST /inference/llm-serving/scale        — scale replicas for a deployment
POST /inference/llm-serving/restart      — restart a deployment
POST /inference/llm-serving/rollback     — rollback to previous config
GET  /inference/llm-serving/logs         — fetch recent deployment logs
GET  /inference/llm-serving/metrics      — fetch serving performance metrics

All state is held in-memory (replace with a real persistence layer as needed).
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger("lightos.llm_serving")

router = APIRouter(prefix="/inference/llm-serving", tags=["llm-serving"])

# ─── Types ────────────────────────────────────────────────────────────────────

ServingMode = Literal["wide-ep", "disaggregated", "standard"]
DeploymentStatus = Literal[
    "pending", "provisioning", "running", "degraded",
    "failed", "rolling-back", "terminated"
]

# ─── DTOs / Schemas ───────────────────────────────────────────────────────────

class ObservabilityConfig(BaseModel):
    prometheus_enabled: bool = Field(default=True)
    tracing_enabled: bool = Field(default=False)
    log_level: Literal["debug", "info", "warn", "error"] = Field(default="info")


class DeployRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="Deployment name (slug)")
    model: str = Field(..., description="Model ID, e.g. 'llama-3.1-405b'")
    serving_mode: ServingMode = Field(default="standard")
    gpu_count: int = Field(default=1, ge=1, le=64)
    data_parallel_size: int = Field(default=1, ge=1, le=32)
    expert_parallelism: bool = Field(default=False)
    fault_tolerance: bool = Field(default=False)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


class ScaleRequest(BaseModel):
    deployment_id: str = Field(..., description="Target deployment ID")
    replicas: int = Field(..., ge=0, le=64, description="Desired replica count (0 = suspend)")


class RestartRequest(BaseModel):
    deployment_id: str = Field(..., description="Target deployment ID")


class RollbackRequest(BaseModel):
    deployment_id: str = Field(..., description="Target deployment ID")
    revision: Optional[int] = Field(
        default=None,
        description="Specific revision to roll back to; omit for previous revision",
    )


class DeploymentConfig(BaseModel):
    model: str
    serving_mode: ServingMode
    gpu_count: int
    data_parallel_size: int
    expert_parallelism: bool
    fault_tolerance: bool
    observability: ObservabilityConfig


class Deployment(BaseModel):
    id: str
    name: str
    status: DeploymentStatus
    config: DeploymentConfig
    replicas: int
    created_at: str
    updated_at: str
    revision: int
    latency_p50_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_tok_s: Optional[float] = None


class LogEntry(BaseModel):
    ts: str
    level: Literal["debug", "info", "warn", "error"]
    deployment_id: Optional[str]
    msg: str


class MetricSeries(BaseModel):
    name: str
    value: float
    unit: str
    bar_pct: Optional[float] = None


class MetricsResponse(BaseModel):
    deployment_id: Optional[str]
    snapshot_at: str
    series: list[MetricSeries]


class DeploymentListResponse(BaseModel):
    deployments: list[Deployment]
    total: int


class ActionResponse(BaseModel):
    ok: bool
    deployment_id: str
    new_status: DeploymentStatus
    message: str


# ─── In-memory store ──────────────────────────────────────────────────────────

_deployments: dict[str, Deployment] = {}
_logs: list[LogEntry] = []
_revisions: dict[str, list[DeploymentConfig]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_demo_state() -> None:
    """Populate realistic initial state so the UI is non-empty on first load."""
    configs = [
        DeployRequest(
            name="prod-llama-405b",
            model="llama-3.1-405b",
            serving_mode="wide-ep",
            gpu_count=8,
            data_parallel_size=2,
            expert_parallelism=True,
            fault_tolerance=True,
        ),
        DeployRequest(
            name="staging-mixtral",
            model="mixtral-8x22b",
            serving_mode="disaggregated",
            gpu_count=4,
            data_parallel_size=1,
            expert_parallelism=True,
            fault_tolerance=False,
        ),
        DeployRequest(
            name="canary-deepseek",
            model="deepseek-v2",
            serving_mode="standard",
            gpu_count=2,
            data_parallel_size=1,
            expert_parallelism=False,
            fault_tolerance=False,
        ),
    ]
    statuses: list[DeploymentStatus] = ["running", "degraded", "provisioning"]
    metrics = [
        {"latency_p50_ms": 42.0, "latency_p99_ms": 128.0, "throughput_tok_s": 1240.0},
        {"latency_p50_ms": 67.0, "latency_p99_ms": 245.0, "throughput_tok_s": 680.0},
        {"latency_p50_ms": None, "latency_p99_ms": None, "throughput_tok_s": None},
    ]
    replicas = [4, 2, 1]

    for req, status, m, r in zip(configs, statuses, metrics, replicas):
        dep_id = f"dep-{uuid.uuid4().hex[:8]}"
        cfg = DeploymentConfig(
            model=req.model,
            serving_mode=req.serving_mode,
            gpu_count=req.gpu_count,
            data_parallel_size=req.data_parallel_size,
            expert_parallelism=req.expert_parallelism,
            fault_tolerance=req.fault_tolerance,
            observability=req.observability,
        )
        _deployments[dep_id] = Deployment(
            id=dep_id,
            name=req.name,
            status=status,
            config=cfg,
            replicas=r,
            created_at=_now(),
            updated_at=_now(),
            revision=1,
            **m,
        )
        _revisions[dep_id] = [cfg]

    _logs.extend([
        LogEntry(ts="12:45:03.221", level="info", deployment_id=None,
                 msg="Deployment prod-llama-405b health check passed (4/4 replicas healthy)"),
        LogEntry(ts="12:44:58.109", level="warn", deployment_id=None,
                 msg="Deployment staging-mixtral replica-1 memory pressure detected (89.2% utilization)"),
        LogEntry(ts="12:44:51.887", level="info", deployment_id=None,
                 msg="Auto-scaler evaluated prod-llama-405b: no scaling action needed (load 62%)"),
        LogEntry(ts="12:44:47.332", level="error", deployment_id=None,
                 msg="Deployment staging-mixtral replica-0 NCCL timeout on rank 3 — retrying collective"),
        LogEntry(ts="12:44:42.001", level="info", deployment_id=None,
                 msg="Provisioning canary-deepseek: requesting 2x H100 from cluster pool-west-2"),
        LogEntry(ts="12:44:38.554", level="info", deployment_id=None,
                 msg="KV cache warmed for prod-llama-405b: 12.4 GB allocated across 8 GPUs"),
        LogEntry(ts="12:44:30.112", level="warn", deployment_id=None,
                 msg="staging-mixtral prefill queue depth exceeding threshold (128 pending)"),
        LogEntry(ts="12:44:22.009", level="info", deployment_id=None,
                 msg="Ray Serve controller healthy — 3 active deployments, 7 replicas total"),
    ])


_seed_demo_state()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_or_404(deployment_id: str) -> Deployment:
    dep = _deployments.get(deployment_id)
    if dep is None:
        raise HTTPException(status_code=404, detail=f"Deployment '{deployment_id}' not found")
    return dep


def _append_log(level: str, msg: str, deployment_id: Optional[str] = None) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    _logs.append(LogEntry(ts=ts, level=level, deployment_id=deployment_id, msg=msg))
    logger.info("[%s] %s", deployment_id or "system", msg)


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("", response_model=DeploymentListResponse, summary="List LLM deployments")
async def list_deployments(
    status: Optional[DeploymentStatus] = Query(default=None, description="Filter by status"),
) -> DeploymentListResponse:
    """Return all deployments, optionally filtered by status."""
    items = list(_deployments.values())
    if status is not None:
        items = [d for d in items if d.status == status]
    return DeploymentListResponse(deployments=items, total=len(items))


@router.post("/deploy", response_model=Deployment, status_code=201,
             summary="Create a new LLM deployment")
async def create_deployment(req: DeployRequest) -> Deployment:
    """
    Submit a new Ray Serve + vLLM deployment request.
    The deployment starts in 'pending' and transitions to 'provisioning'.
    Poll GET /inference/llm-serving until status is 'running' or 'failed'.
    """
    # Reject duplicate names
    for dep in _deployments.values():
        if dep.name == req.name and dep.status != "terminated":
            raise HTTPException(
                status_code=409,
                detail=f"Active deployment with name '{req.name}' already exists",
            )

    dep_id = f"dep-{uuid.uuid4().hex[:8]}"
    cfg = DeploymentConfig(
        model=req.model,
        serving_mode=req.serving_mode,
        gpu_count=req.gpu_count,
        data_parallel_size=req.data_parallel_size,
        expert_parallelism=req.expert_parallelism,
        fault_tolerance=req.fault_tolerance,
        observability=req.observability,
    )
    dep = Deployment(
        id=dep_id,
        name=req.name,
        status="provisioning",
        config=cfg,
        replicas=req.data_parallel_size,
        created_at=_now(),
        updated_at=_now(),
        revision=1,
    )
    _deployments[dep_id] = dep
    _revisions[dep_id] = [cfg]
    _append_log(
        "info",
        f"Deployment {req.name} ({dep_id}) submitted — "
        f"model={req.model}, mode={req.serving_mode}, gpus={req.gpu_count}",
        dep_id,
    )
    return dep


@router.post("/scale", response_model=ActionResponse, summary="Scale deployment replicas")
async def scale_deployment(req: ScaleRequest) -> ActionResponse:
    """
    Adjust the replica count for a running or degraded deployment.
    Scaling to 0 suspends the deployment (status → 'pending').
    """
    dep = _get_or_404(req.deployment_id)
    if dep.status in ("failed", "terminated", "rolling-back"):
        raise HTTPException(
            status_code=422,
            detail=f"Cannot scale a deployment in '{dep.status}' status",
        )

    new_status: DeploymentStatus = "pending" if req.replicas == 0 else dep.status
    dep.replicas = req.replicas
    dep.status = new_status
    dep.updated_at = _now()
    _deployments[req.deployment_id] = dep
    _append_log(
        "info",
        f"Scale applied to {dep.name}: {req.replicas} replicas requested",
        req.deployment_id,
    )
    return ActionResponse(
        ok=True,
        deployment_id=req.deployment_id,
        new_status=new_status,
        message=f"Scaled {dep.name} to {req.replicas} replicas",
    )


@router.post("/restart", response_model=ActionResponse, summary="Restart a deployment")
async def restart_deployment(req: RestartRequest) -> ActionResponse:
    """
    Trigger a rolling restart of all replicas in a deployment.
    The deployment transitions through 'provisioning' before returning to 'running'.
    """
    dep = _get_or_404(req.deployment_id)
    if dep.status == "terminated":
        raise HTTPException(
            status_code=422,
            detail="Cannot restart a terminated deployment",
        )

    dep.status = "provisioning"
    dep.updated_at = _now()
    _deployments[req.deployment_id] = dep
    _append_log(
        "info",
        f"Rolling restart initiated for {dep.name}",
        req.deployment_id,
    )
    return ActionResponse(
        ok=True,
        deployment_id=req.deployment_id,
        new_status="provisioning",
        message=f"Rolling restart triggered for {dep.name}",
    )


@router.post("/rollback", response_model=ActionResponse, summary="Rollback a deployment")
async def rollback_deployment(req: RollbackRequest) -> ActionResponse:
    """
    Roll a deployment back to a previous revision.
    If `revision` is omitted, the previous revision is used.
    The deployment transitions through 'rolling-back' then 'provisioning'.
    """
    dep = _get_or_404(req.deployment_id)
    history = _revisions.get(req.deployment_id, [])
    if len(history) < 2:
        raise HTTPException(
            status_code=409,
            detail="No previous revision available to roll back to",
        )

    target_rev = req.revision if req.revision is not None else len(history) - 1
    if target_rev < 1 or target_rev > len(history):
        raise HTTPException(
            status_code=422,
            detail=f"Revision {target_rev} does not exist (history has {len(history)} entries)",
        )

    dep.config = history[target_rev - 1]
    dep.status = "rolling-back"
    dep.revision = target_rev
    dep.updated_at = _now()
    _deployments[req.deployment_id] = dep
    _append_log(
        "warn",
        f"Rollback to revision {target_rev} initiated for {dep.name}",
        req.deployment_id,
    )
    return ActionResponse(
        ok=True,
        deployment_id=req.deployment_id,
        new_status="rolling-back",
        message=f"Rolling back {dep.name} to revision {target_rev}",
    )


@router.get("/logs", response_model=list[LogEntry], summary="Fetch deployment logs")
async def get_logs(
    deployment_id: Optional[str] = Query(default=None, description="Filter by deployment ID"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum log lines to return"),
) -> list[LogEntry]:
    """
    Return recent log lines from the Ray Serve controller.
    Optionally filter to a specific deployment.
    Returns newest entries last.
    """
    entries = _logs if deployment_id is None else [
        e for e in _logs if e.deployment_id == deployment_id
    ]
    return entries[-limit:]


@router.get("/metrics", response_model=MetricsResponse, summary="Fetch serving metrics")
async def get_metrics(
    deployment_id: Optional[str] = Query(
        default=None,
        description="If provided, return metrics scoped to that deployment; otherwise return cluster-wide metrics",
    ),
) -> MetricsResponse:
    """
    Return a snapshot of LLM serving performance metrics.
    When `deployment_id` is supplied, metrics are scoped to that deployment.
    """
    if deployment_id is not None:
        dep = _get_or_404(deployment_id)
        throughput = dep.throughput_tok_s or 0.0
        p50 = dep.latency_p50_ms or 0.0
        p99 = dep.latency_p99_ms or 0.0
        series = [
            MetricSeries(name="Throughput", value=throughput, unit="tok/s",
                         bar_pct=min(100.0, throughput / 20.0)),
            MetricSeries(name="Latency P50", value=p50, unit="ms"),
            MetricSeries(name="Latency P99", value=p99, unit="ms"),
        ]
    else:
        running = [d for d in _deployments.values() if d.status == "running"]
        total_throughput = sum(d.throughput_tok_s or 0 for d in running)
        avg_p50 = (
            sum(d.latency_p50_ms or 0 for d in running) / max(len(running), 1)
            if running else 0.0
        )
        series = [
            MetricSeries(name="Throughput", value=total_throughput, unit="tok/s",
                         bar_pct=min(100.0, total_throughput / 40.0)),
            MetricSeries(name="Queue Depth", value=12.0, unit="requests", bar_pct=15.0),
            MetricSeries(name="KV Cache Hit Rate", value=94.2, unit="%", bar_pct=94.2),
            MetricSeries(name="GPU Memory", value=72.8, unit="GB", bar_pct=91.0),
            MetricSeries(name="Batch Utilization", value=86.0, unit="%", bar_pct=86.0),
            MetricSeries(name="Avg Latency P50", value=round(avg_p50, 1), unit="ms"),
        ]

    return MetricsResponse(
        deployment_id=deployment_id,
        snapshot_at=_now(),
        series=series,
    )
