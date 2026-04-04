"""
LightOS — FastAPI backend
=========================

Endpoints
---------
GET  /api/compile/ping   — health-check
GET  /api/fabrics        — list all fabric presets
POST /api/compile        — SSE stream of compilation log lines

GET  /inference/llm-serving              — list LLM deployments
POST /inference/llm-serving/deploy       — create a new deployment
POST /inference/llm-serving/scale        — scale replicas
POST /inference/llm-serving/restart      — restart a deployment
POST /inference/llm-serving/rollback     — rollback to previous config
GET  /inference/llm-serving/logs         — fetch deployment logs
GET  /inference/llm-serving/metrics      — fetch serving metrics

Run locally:
 pip install -r requirements.txt
 uvicorn main:app --reload --port 8000
"""

import json
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from compiler_sim import stream_compilation
from fabrics import FABRIC_PRESETS, get_fabric
from llm_serving import router as llm_serving_router

app = FastAPI(title="LightOS API", version="0.3.0")

# Allow any origin in dev (the Vite proxy handles this in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(llm_serving_router)

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/compile/ping")
async def ping():
    return {"status": "ok", "service": "lightrail-compiler", "version": "0.3.0"}

# ─── Fabrics ──────────────────────────────────────────────────────────────────

@app.get("/api/fabrics")
async def list_fabrics():
    return [
        {
            "name": f.name,
            "display": f.display,
            "layers": f.layers,
            "wdm_channels": f.wdm_channels,
            "nodes": f.nodes,
            "topology": f.topology,
            "dispatch_ns": f.dispatch_ns,
            "bw_gbps": f.bw_gbps,
            "congestion_score": f.congestion_score,
            "tags": f.tags,
        }
        for f in FABRIC_PRESETS.values()
    ]

# ─── Compile (SSE stream) ─────────────────────────────────────────────────────

class CompileRequest(BaseModel):
    fabric: str = Field(default="photonic-mesh-20x64", description="Fabric preset name or NxM string")
    function_name: str = Field(default="attention_kernel", description="Kernel / function name")
    cached: bool = Field(default=False, description="Simulate a cache-hit run")
    stream_delay_ms: int = Field(default=120, ge=0, le=2000, description="Inter-line delay (ms)")

async def _sse_generator(req: CompileRequest):
    """Wrap the async generator as an SSE byte stream."""
    async for line in stream_compilation(
        fabric_name=req.fabric,
        function_name=req.function_name,
        cached=req.cached,
        stream_delay_ms=req.stream_delay_ms,
    ):
        payload = json.dumps(line, ensure_ascii=False)
        yield f"data: {payload}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/api/compile")
async def compile_kernel(req: CompileRequest):
    return StreamingResponse(
        _sse_generator(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# ─── Dev entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
