import asyncio

from api.routes.auth import User, get_current_user, require_admin
from fastapi import APIRouter, Depends, HTTPException, Request

router = APIRouter()


def _manager(request: Request):
    ds = getattr(request.app.state, "detection_service", None)
    mm = getattr(ds, "model_manager", None) if ds else None
    if mm is None:
        raise HTTPException(503, "AI models are unavailable")
    return mm


@router.get("/status")
async def model_status(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    mm = _manager(request)
    status = mm.status()
    status["health"] = mm.health()
    status.pop("config", None)
    return status


@router.get("/metrics")
async def model_metrics(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    return _manager(request).metrics.snapshot()


@router.get("/config")
async def model_config(
    request: Request,
    current_user: User = Depends(require_admin),
):
    return _manager(request).config.to_dict()


@router.post("/reload")
async def reload_models(
    request: Request,
    current_user: User = Depends(require_admin),
):
    mm = _manager(request)
    await asyncio.to_thread(mm.reload)
    return {
        "message": "Models reloaded",
        "health": mm.health(),
        "errors": mm.load_errors,
    }
