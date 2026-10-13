"""Evaluation run listing endpoints (spec #39). Reads back experiment records written by
evaluation/experiment.py (`rag evaluate` / `rag experiment run`). Read-only and ADMIN-ONLY:
run results can contain dataset questions and retrieved text from every tenant.

The results directory is fixed server-side (data_dir/evaluation/results); it is deliberately
NOT a request parameter, since a caller-controlled directory would be an arbitrary
directory read."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_engine.api.dependencies import get_engine_state, require_admin
from rag_engine.api.engine import EngineState
from rag_engine.evaluation.reports import load_experiments
from rag_engine.exceptions import NotFoundError
from rag_engine.security.authorization import AuthContext

router = APIRouter(prefix="/api/v1", tags=["evaluation"])


@router.get("/evaluation/runs")
async def list_runs(engine: EngineState = Depends(get_engine_state), _admin: AuthContext = Depends(require_admin)) -> dict:
    records = load_experiments(engine.settings.data_dir / "evaluation" / "results")
    runs = [{"experiment_id": r["experiment_id"], "name": r["name"], "timestamp": r["timestamp"]} for r in records]
    return {"count": len(records), "runs": runs}


@router.get("/evaluation/runs/{run_id}")
async def get_run(run_id: str, engine: EngineState = Depends(get_engine_state), _admin: AuthContext = Depends(require_admin)) -> dict:
    for r in load_experiments(engine.settings.data_dir / "evaluation" / "results"):
        if r["experiment_id"] == run_id:
            return r
    raise NotFoundError(f"evaluation run {run_id} not found")
