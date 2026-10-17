import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from api.routes.auth import User, get_current_user
from api.websocket import manager
from config.settings import settings
from database.database import get_db
from database.models import (
    Alert,
    AlertStatus,
    Camera,
    CameraStatus,
    CameraType,
    Detection,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from services.detection_service import camera_runtime_config
from sqlalchemy.orm import Session

router = APIRouter()

ALLOWED_STREAM_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")


def validate_stream_url(value: Optional[str]) -> Optional[str]:
    if value is None:
        return value
    cleaned = value.strip()
    if cleaned in ("", "simulated"):
        return cleaned
    if not cleaned.lower().startswith(ALLOWED_STREAM_SCHEMES):
        raise ValueError(
            "stream_url must be 'simulated' or use one of: "
            + ", ".join(ALLOWED_STREAM_SCHEMES)
        )
    return cleaned


class CameraCreate(BaseModel):
    name: str
    description: Optional[str] = None
    camera_type: CameraType = CameraType.RTSP
    stream_url: Optional[str] = None
    snapshot_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_meters: float = 0.0
    location_name: Optional[str] = None
    direction_degrees: float = 0.0
    fov_degrees: float = 90.0
    detect_persons: bool = True
    detect_vehicles: bool = True
    detect_drones: bool = True
    resolution_width: int = 1920
    resolution_height: int = 1080
    fps: int = 25
    night_vision: bool = False
    ptz_enabled: bool = False
    geofence_id: Optional[int] = None

    _check_stream_url = field_validator("stream_url")(validate_stream_url)


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    stream_url: Optional[str] = None
    snapshot_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: Optional[str] = None
    direction_degrees: Optional[float] = None
    fov_degrees: Optional[float] = None
    is_active: Optional[bool] = None
    detect_persons: Optional[bool] = None
    detect_vehicles: Optional[bool] = None
    detect_drones: Optional[bool] = None
    night_vision: Optional[bool] = None
    ptz_enabled: Optional[bool] = None
    geofence_id: Optional[int] = None

    _check_stream_url = field_validator("stream_url")(validate_stream_url)


class CameraOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    camera_type: CameraType
    stream_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_meters: float
    location_name: Optional[str] = None
    direction_degrees: float
    fov_degrees: float
    status: CameraStatus
    is_active: bool
    detect_persons: bool
    detect_vehicles: bool
    detect_drones: bool
    resolution_width: int
    resolution_height: int
    fps: int
    night_vision: bool
    ptz_enabled: bool
    geofence_id: Optional[int] = None
    last_seen: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


def _get_detection_service(request: Request):
    return getattr(request.app.state, "detection_service", None)


@router.get("/", response_model=List[CameraOut])
async def list_cameras(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    status: Optional[CameraStatus] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(Camera)
    if status:
        q = q.filter(Camera.status == status)
    if is_active is not None:
        q = q.filter(Camera.is_active == is_active)
    return q.order_by(Camera.id).offset(skip).limit(limit).all()


@router.post("/", response_model=CameraOut, status_code=201)
async def create_camera(
    camera_data: CameraCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    camera = Camera(**camera_data.model_dump())
    camera.status = CameraStatus.INITIALIZING
    db.add(camera)
    db.commit()
    db.refresh(camera)

    ds = _get_detection_service(request)
    if ds and camera.is_active:
        background_tasks.add_task(
            ds.start_camera,
            camera.id,
            camera.stream_url or "simulated",
            camera_runtime_config(camera),
        )

    background_tasks.add_task(
        manager.broadcast,
        {"type": "camera_added", "camera_id": camera.id, "name": camera.name},
    )
    return camera


@router.get("/{camera_id}", response_model=CameraOut)
async def get_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")
    return cam


@router.put("/{camera_id}", response_model=CameraOut)
async def update_camera(
    camera_id: int,
    camera_data: CameraUpdate,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")
    for field, value in camera_data.model_dump(exclude_unset=True).items():
        setattr(cam, field, value)
    db.commit()
    db.refresh(cam)

    ds = _get_detection_service(request)
    if ds and cam.is_active:
        background_tasks.add_task(
            ds.reconfigure_camera,
            cam.id,
            cam.stream_url or "simulated",
            camera_runtime_config(cam),
        )
    return cam


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(
    camera_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    ds = _get_detection_service(request)
    if ds:
        background_tasks.add_task(ds.stop_camera, camera_id)

    db.delete(cam)
    db.commit()


@router.post("/{camera_id}/enable")
async def enable_camera(
    camera_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    cam.is_active = True
    cam.status = CameraStatus.INITIALIZING
    db.commit()
    db.refresh(cam)

    ds = _get_detection_service(request)
    if ds:
        background_tasks.add_task(
            ds.start_camera,
            cam.id,
            cam.stream_url or "simulated",
            camera_runtime_config(cam),
        )
    background_tasks.add_task(
        manager.broadcast_camera_status, str(camera_id), "initializing"
    )
    return {"message": f"Camera '{cam.name}' enabled"}


@router.post("/{camera_id}/disable")
async def disable_camera(
    camera_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    cam.is_active = False
    cam.status = CameraStatus.DISABLED
    db.commit()

    ds = _get_detection_service(request)
    if ds:
        background_tasks.add_task(ds.stop_camera, camera_id)
    background_tasks.add_task(
        manager.broadcast_camera_status, str(camera_id), "disabled"
    )
    return {"message": f"Camera '{cam.name}' disabled"}


@router.get("/{camera_id}/snapshot")
async def get_camera_snapshot(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    snap_dir = Path(settings.SNAPSHOTS_DIR)
    if snap_dir.exists():
        pattern = f"cam{camera_id:02d}_*.jpg"
        snapshots = sorted(snap_dir.glob(pattern), key=os.path.getmtime, reverse=True)
        if snapshots:
            return FileResponse(str(snapshots[0]), media_type="image/jpeg")

    raise HTTPException(404, "No snapshot available yet")


@router.get("/{camera_id}/stats")
async def get_camera_stats(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(404, "Camera not found")

    total_detections = (
        db.query(Detection).filter(Detection.camera_id == camera_id).count()
    )
    total_alerts = db.query(Alert).filter(Alert.camera_id == camera_id).count()
    active_alerts = (
        db.query(Alert)
        .filter(
            Alert.camera_id == camera_id,
            Alert.status == AlertStatus.ACTIVE,
        )
        .count()
    )

    return {
        "camera_id": camera_id,
        "name": cam.name,
        "status": cam.status.value if cam.status else "unknown",
        "total_detections": total_detections,
        "total_alerts": total_alerts,
        "active_alerts": active_alerts,
        "last_seen": cam.last_seen,
        "snapshot_url": f"/api/cameras/{camera_id}/snapshot",
    }
