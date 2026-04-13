"""
Tests for the LightOS LLM Serving inference API.

Run with:
  cd backend
  pytest test_llm_serving.py -v
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ─── Build a minimal test app that only mounts the llm_serving router ──────

# Import fresh so each test module gets its own seeded state.
# (In production the state lives in the module-level dicts.)
import importlib
import llm_serving as _ls

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(_ls.router)

client = TestClient(app)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_first_deployment_id(status: str | None = None) -> str:
    params = f"?status={status}" if status else ""
    r = client.get(f"/inference/llm-serving{params}")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] > 0, "Expected at least one seeded deployment"
    return data["deployments"][0]["id"]


def create_test_deployment(name: str = "test-deploy", **kwargs) -> dict:
    payload = {
        "name": name,
        "model": "llama-3.1-70b",
        "serving_mode": "standard",
        "gpu_count": 2,
        "data_parallel_size": 1,
        "expert_parallelism": False,
        "fault_tolerance": False,
        **kwargs,
    }
    r = client.post("/inference/llm-serving/deploy", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


# ─── List deployments ─────────────────────────────────────────────────────────

class TestListDeployments:
    def test_returns_200_with_seeded_data(self):
        r = client.get("/inference/llm-serving")
        assert r.status_code == 200
        data = r.json()
        assert "deployments" in data
        assert "total" in data
        assert data["total"] == len(data["deployments"])

    def test_seeded_deployments_have_required_fields(self):
        r = client.get("/inference/llm-serving")
        for dep in r.json()["deployments"]:
            assert "id" in dep
            assert "name" in dep
            assert "status" in dep
            assert "config" in dep
            assert "replicas" in dep
            assert "revision" in dep

    def test_filter_by_status_running(self):
        r = client.get("/inference/llm-serving?status=running")
        assert r.status_code == 200
        for dep in r.json()["deployments"]:
            assert dep["status"] == "running"

    def test_filter_by_status_degraded(self):
        r = client.get("/inference/llm-serving?status=degraded")
        assert r.status_code == 200
        for dep in r.json()["deployments"]:
            assert dep["status"] == "degraded"

    def test_filter_nonexistent_status_returns_empty(self):
        r = client.get("/inference/llm-serving?status=terminated")
        assert r.status_code == 200
        assert r.json()["total"] == 0


# ─── Create deployment ────────────────────────────────────────────────────────

class TestCreateDeployment:
    def test_creates_deployment_returns_201(self):
        dep = create_test_deployment("create-test-1")
        assert dep["id"].startswith("dep-")
        assert dep["name"] == "create-test-1"
        assert dep["status"] == "provisioning"
        assert dep["revision"] == 1

    def test_config_fields_persisted(self):
        dep = create_test_deployment(
            "create-test-2",
            model="mixtral-8x22b",
            serving_mode="wide-ep",
            gpu_count=4,
            data_parallel_size=2,
            expert_parallelism=True,
            fault_tolerance=True,
        )
        cfg = dep["config"]
        assert cfg["model"] == "mixtral-8x22b"
        assert cfg["serving_mode"] == "wide-ep"
        assert cfg["gpu_count"] == 4
        assert cfg["data_parallel_size"] == 2
        assert cfg["expert_parallelism"] is True
        assert cfg["fault_tolerance"] is True

    def test_all_serving_modes_accepted(self):
        for mode in ("wide-ep", "disaggregated", "standard"):
            dep = create_test_deployment(f"mode-test-{mode}", serving_mode=mode)
            assert dep["config"]["serving_mode"] == mode

    def test_duplicate_name_returns_409(self):
        create_test_deployment("dupe-deploy")
        r = client.post(
            "/inference/llm-serving/deploy",
            json={
                "name": "dupe-deploy",
                "model": "llama-3.1-70b",
                "serving_mode": "standard",
                "gpu_count": 1,
                "data_parallel_size": 1,
            },
        )
        assert r.status_code == 409
        assert "already exists" in r.json()["detail"].lower()

    def test_missing_name_returns_422(self):
        r = client.post(
            "/inference/llm-serving/deploy",
            json={"model": "llama-3.1-70b"},
        )
        assert r.status_code == 422

    def test_gpu_count_bounds_enforced(self):
        r = client.post(
            "/inference/llm-serving/deploy",
            json={"name": "gpu-overflow", "model": "llama-3.1-70b", "gpu_count": 999},
        )
        assert r.status_code == 422

    def test_new_deployment_appears_in_list(self):
        name = "list-check-deploy"
        dep = create_test_deployment(name)
        r = client.get("/inference/llm-serving")
        ids = [d["id"] for d in r.json()["deployments"]]
        assert dep["id"] in ids


# ─── Scale deployment ─────────────────────────────────────────────────────────

class TestScaleDeployment:
    def test_scale_running_deployment(self):
        dep_id = get_first_deployment_id("running")
        r = client.post(
            "/inference/llm-serving/scale",
            json={"deployment_id": dep_id, "replicas": 8},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["deployment_id"] == dep_id

    def test_scale_to_zero_sets_pending(self):
        dep = create_test_deployment("scale-zero-test")
        dep_id = dep["id"]
        r = client.post(
            "/inference/llm-serving/scale",
            json={"deployment_id": dep_id, "replicas": 0},
        )
        assert r.status_code == 200
        assert r.json()["new_status"] == "pending"

    def test_scale_nonexistent_returns_404(self):
        r = client.post(
            "/inference/llm-serving/scale",
            json={"deployment_id": "dep-00000000", "replicas": 3},
        )
        assert r.status_code == 404

    def test_scale_replicas_bounds(self):
        dep_id = get_first_deployment_id("running")
        r = client.post(
            "/inference/llm-serving/scale",
            json={"deployment_id": dep_id, "replicas": 999},
        )
        assert r.status_code == 422

    def test_scale_updates_replica_count(self):
        dep = create_test_deployment("scale-count-test")
        dep_id = dep["id"]
        client.post(
            "/inference/llm-serving/scale",
            json={"deployment_id": dep_id, "replicas": 5},
        )
        r = client.get("/inference/llm-serving")
        for d in r.json()["deployments"]:
            if d["id"] == dep_id:
                assert d["replicas"] == 5
                break


# ─── Restart deployment ───────────────────────────────────────────────────────

class TestRestartDeployment:
    def test_restart_running_deployment(self):
        dep_id = get_first_deployment_id("running")
        r = client.post(
            "/inference/llm-serving/restart",
            json={"deployment_id": dep_id},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["new_status"] == "provisioning"

    def test_restart_transitions_to_provisioning(self):
        dep = create_test_deployment("restart-state-test")
        dep_id = dep["id"]
        r = client.post(
            "/inference/llm-serving/restart",
            json={"deployment_id": dep_id},
        )
        assert r.json()["new_status"] == "provisioning"
        # Verify the deployment list reflects the new status
        listing = client.get("/inference/llm-serving")
        for d in listing.json()["deployments"]:
            if d["id"] == dep_id:
                assert d["status"] == "provisioning"
                break

    def test_restart_nonexistent_returns_404(self):
        r = client.post(
            "/inference/llm-serving/restart",
            json={"deployment_id": "dep-deadbeef"},
        )
        assert r.status_code == 404


# ─── Rollback deployment ──────────────────────────────────────────────────────

class TestRollbackDeployment:
    def test_rollback_with_no_history_returns_409(self):
        dep = create_test_deployment("rollback-no-history")
        r = client.post(
            "/inference/llm-serving/rollback",
            json={"deployment_id": dep["id"]},
        )
        assert r.status_code == 409
        assert "no previous revision" in r.json()["detail"].lower()

    def test_rollback_nonexistent_returns_404(self):
        r = client.post(
            "/inference/llm-serving/rollback",
            json={"deployment_id": "dep-ffffffff"},
        )
        assert r.status_code == 404

    def test_rollback_invalid_revision_returns_422(self):
        dep = create_test_deployment("rollback-bad-rev")
        r = client.post(
            "/inference/llm-serving/rollback",
            json={"deployment_id": dep["id"], "revision": 99},
        )
        assert r.status_code in (409, 422)


# ─── Logs ─────────────────────────────────────────────────────────────────────

class TestLogs:
    def test_returns_200_list(self):
        r = client.get("/inference/llm-serving/logs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_seeded_logs_present(self):
        r = client.get("/inference/llm-serving/logs")
        assert len(r.json()) > 0

    def test_log_entries_have_required_fields(self):
        r = client.get("/inference/llm-serving/logs")
        for entry in r.json():
            assert "ts" in entry
            assert "level" in entry
            assert "msg" in entry

    def test_limit_param_respected(self):
        r = client.get("/inference/llm-serving/logs?limit=3")
        assert r.status_code == 200
        assert len(r.json()) <= 3

    def test_deployment_id_filter(self):
        dep_id = get_first_deployment_id("running")
        # Trigger an action that generates a log for this deployment
        client.post(
            "/inference/llm-serving/restart",
            json={"deployment_id": dep_id},
        )
        r = client.get(f"/inference/llm-serving/logs?deployment_id={dep_id}")
        assert r.status_code == 200
        for entry in r.json():
            assert entry["deployment_id"] == dep_id

    def test_actions_append_to_logs(self):
        before = len(client.get("/inference/llm-serving/logs").json())
        dep = create_test_deployment("log-action-test")
        client.post(
            "/inference/llm-serving/restart",
            json={"deployment_id": dep["id"]},
        )
        after = len(client.get("/inference/llm-serving/logs").json())
        # At least the deploy + restart logs should have been added
        assert after > before


# ─── Metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_cluster_metrics_returns_200(self):
        r = client.get("/inference/llm-serving/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "series" in data
        assert "snapshot_at" in data
        assert isinstance(data["series"], list)
        assert len(data["series"]) > 0

    def test_metric_series_fields(self):
        r = client.get("/inference/llm-serving/metrics")
        for s in r.json()["series"]:
            assert "name" in s
            assert "value" in s
            assert "unit" in s

    def test_scoped_metrics_for_running_deployment(self):
        dep_id = get_first_deployment_id("running")
        r = client.get(f"/inference/llm-serving/metrics?deployment_id={dep_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["deployment_id"] == dep_id
        assert len(data["series"]) > 0

    def test_metrics_for_nonexistent_deployment_returns_404(self):
        r = client.get("/inference/llm-serving/metrics?deployment_id=dep-00000000")
        assert r.status_code == 404


# ─── Status transition matrix ─────────────────────────────────────────────────

class TestStatusTransitions:
    """Verify that each action moves the deployment to the correct status."""

    def test_deploy_starts_provisioning(self):
        dep = create_test_deployment("trans-deploy")
        assert dep["status"] == "provisioning"

    def test_restart_from_provisioning_stays_provisioning(self):
        dep = create_test_deployment("trans-restart")
        r = client.post(
            "/inference/llm-serving/restart",
            json={"deployment_id": dep["id"]},
        )
        assert r.json()["new_status"] == "provisioning"

    def test_scale_to_zero_gives_pending(self):
        dep = create_test_deployment("trans-scale-zero")
        r = client.post(
            "/inference/llm-serving/scale",
            json={"deployment_id": dep["id"], "replicas": 0},
        )
        assert r.json()["new_status"] == "pending"

    def test_scale_nonzero_preserves_status(self):
        dep_id = get_first_deployment_id("running")
        r = client.post(
            "/inference/llm-serving/scale",
            json={"deployment_id": dep_id, "replicas": 4},
        )
        assert r.json()["new_status"] == "running"

    def test_rollback_sets_rolling_back(self):
        # Manually inject a deployment with 2 revision entries for rollback to work
        import uuid as _uuid
        from llm_serving import _deployments, _revisions, DeploymentConfig, Deployment, ObservabilityConfig, _now
        dep_id = f"dep-{_uuid.uuid4().hex[:8]}"
        obs = ObservabilityConfig()
        cfg_v1 = DeploymentConfig(
            model="llama-3.1-70b",
            serving_mode="standard",
            gpu_count=2,
            data_parallel_size=1,
            expert_parallelism=False,
            fault_tolerance=False,
            observability=obs,
        )
        cfg_v2 = DeploymentConfig(
            model="llama-3.1-70b",
            serving_mode="wide-ep",
            gpu_count=4,
            data_parallel_size=2,
            expert_parallelism=True,
            fault_tolerance=True,
            observability=obs,
        )
        _deployments[dep_id] = Deployment(
            id=dep_id, name="trans-rollback", status="running", config=cfg_v2,
            replicas=2, created_at=_now(), updated_at=_now(), revision=2,
        )
        _revisions[dep_id] = [cfg_v1, cfg_v2]

        r = client.post(
            "/inference/llm-serving/rollback",
            json={"deployment_id": dep_id},
        )
        assert r.status_code == 200
        assert r.json()["new_status"] == "rolling-back"
