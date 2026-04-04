# LightOS — Claude Code Guide

This file documents conventions for Claude Code sessions working on the LightOS codebase.

---

## Repository layout

```
/
├── src/                        # Hardware RTL (Verilog, Tiny Tapeout)
│   └── project.v               # LR-P8A photonic inference core
├── hardware/                   # KiCAD schematics (LR-P8A PCIe card)
├── test/                       # cocotb RTL test suite
├── docs/                       # Design documentation
├── backend/                    # FastAPI inference API (Python)
│   ├── main.py                 # App entrypoint + compiler endpoints
│   ├── llm_serving.py          # LLM Serving / Ray Serve module  ← NEW
│   ├── fabrics.py              # Photonic fabric presets
│   ├── compiler_sim.py         # Compilation simulation
│   ├── requirements.txt        # Python dependencies
│   └── test_llm_serving.py     # Pytest tests for LLM serving     ← NEW
└── inference/                  # Frontend additions (TypeScript/React)
    └── src/pages/
        └── LLMServing.tsx      # LLM Serving dashboard page        ← NEW
```

---

## Backend conventions (`backend/`)

### Framework
FastAPI ≥ 0.111, Pydantic v2, uvicorn.

### Routing
- Compiler / fabric endpoints live directly in `main.py` under `/api/...`.
- Feature modules are placed in their own file (e.g. `llm_serving.py`) and mounted as an `APIRouter` with a prefix. Register the router in `main.py` via `app.include_router(...)`.

### Models / DTOs
- Use `pydantic.BaseModel` for all request and response schemas.
- Keep request schemas in the same file as their routes (co-location).
- Prefer `Literal[...]` for constrained string enums (avoids a full `enum.Enum` import).

### State
- For demo / simulation features, use module-level `dict` stores.
- Seed realistic initial state from a `_seed_*()` function called at module import time.
- For production persistence, replace the in-memory dicts with your ORM / database layer without changing the API surface.

### Auth
- Dev: no auth (CORS open). Production: add bearer-token middleware in `main.py`.

### Logging
- Use `logging.getLogger("lightos.<module>")` — e.g. `logging.getLogger("lightos.llm_serving")`.

### Running locally
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Running tests
```bash
cd backend
pytest test_llm_serving.py -v
```

---

## LLM Serving feature (`backend/llm_serving.py`)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/inference/llm-serving` | List all deployments (filter by `?status=`) |
| `POST` | `/inference/llm-serving/deploy` | Create a deployment |
| `POST` | `/inference/llm-serving/scale` | Scale replicas (`replicas=0` suspends) |
| `POST` | `/inference/llm-serving/restart` | Rolling restart → `provisioning` |
| `POST` | `/inference/llm-serving/rollback` | Roll back to previous config → `rolling-back` |
| `GET` | `/inference/llm-serving/logs` | Deployment logs (filter by `?deployment_id=`, paginate with `?limit=`) |
| `GET` | `/inference/llm-serving/metrics` | Serving metrics (scoped by `?deployment_id=`) |

### Serving modes
| ID | Description |
|----|-------------|
| `wide-ep` | Expert parallelism across all GPUs (MoE models) |
| `disaggregated` | Separate prefill/decode stages |
| `standard` | Tensor-parallel with auto-batching |

### Deployment status lifecycle
```
         ┌──────────────────────────────────────────┐
         │                                          │
  submit │                                          │ rollback
         ▼                                          │
     pending ──deploy──► provisioning ──►  running ─┤
                              │               │      │
                              │               ▼      │
                              └──────────► degraded  │
                                             │       │
                                             ▼       │
                                           failed    │
                                                     │
                                        rolling-back ◄┘
                                             │
                                             ▼
                                        provisioning
                                             │
                                             ▼
                                          terminated
```

### Adding new fields / endpoints
1. Add the Pydantic schema in `llm_serving.py`.
2. Implement the route on `router` (the module-level `APIRouter`).
3. Add tests in `test_llm_serving.py`.
4. Update this table above.

---

## Frontend conventions (`inference/src/`)

### Framework
React + TypeScript, shadcn-ui components, Tailwind CSS, Framer Motion, sonner toasts.

### API calls
Use the `apiFetch<T>(path, options?)` helper (defined in `LLMServing.tsx`) rather than raw `fetch`. It:
- Prepends `API_BASE` (`/inference/llm-serving`)
- Sets `Content-Type: application/json`
- Throws on non-2xx responses with the server detail message

### Loading states
Always guard data-driven sections with `loadingXxx` booleans and render `<Skeleton>` placeholders while fetching.

### Toast feedback
- Success: `toast.success("…")`
- Error: `toast.error(\`Operation failed: \${err.message}\`)`
- Info: `toast.info("…")`

### Navigation
`LLMServing` is already registered in `DashboardSidebar.tsx` under the Monitoring section at `/dashboard/llm-serving`. Do not move it — preserve the existing navigation structure.

---

## Hardware RTL (`src/`, `test/`, `hardware/`)

No changes to the hardware files are needed for software feature work. See `docs/info.md` for the photonic inference core design documentation.
