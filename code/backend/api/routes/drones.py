from datetime import datetime, timedelta, timezone
from typing import List, Optional

from api.routes.auth import User, get_current_user
from database.database import get_db
from database.models import DroneDetection, ThreatLevel
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

router = APIRouter()


class DroneOut(BaseModel):
    id: int
    camera_id: int
    confidence: float
    bounding_box: Optional[dict] = None
    snapshot_path: Optional[str] = None
    estimated_altitude_m: Optional[float] = None
    estimated_speed_ms: Optional[float] = None
    trajectory_data: Optional[list] = None
    drone_type: Optional[str] = None
    is_authorized: bool
    risk_level: ThreatLevel
    ai_analysis: Optional[str] = None
    timestamp: datetime

    model_config = {"from_attributes": True}


@router.get("/", response_model=List[DroneOut])
async def list_drone_detections(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    hours: int = Query(24, ge=1, le=24 * 90),
    camera_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = db.query(DroneDetection).filter(DroneDetection.timestamp >= since)
    if camera_id:
        q = q.filter(DroneDetection.camera_id == camera_id)
    return q.order_by(DroneDetection.timestamp.desc()).offset(skip).limit(limit).all()


@router.get("/active")
async def get_active_drones(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    since = datetime.now(timezone.utc) - timedelta(minutes=5)
    drones = db.query(DroneDetection).filter(DroneDetection.timestamp >= since).all()
    ds = getattr(request.app.state, "detection_service", None)
    live_tracks = ds.get_live_tracks("drone") if ds else []
    return {
        "active_drones": len(drones),
        "live_tracks": live_tracks,
        "detections": [
            DroneOut.model_validate(d).model_dump(mode="json") for d in drones
        ],
    }


@router.get("/stats")
async def drone_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    total = db.query(DroneDetection).count()
    last_24h = (
        db.query(DroneDetection)
        .filter(
            DroneDetection.timestamp >= datetime.now(timezone.utc) - timedelta(hours=24)
        )
        .count()
    )
    authorized = (
        db.query(DroneDetection).filter(DroneDetection.is_authorized.is_(True)).count()
    )
    unauthorized = total - authorized
    return {
        "total_detections": total,
        "last_24h": last_24h,
        "authorized": authorized,
        "unauthorized": unauthorized,
        "threat_percentage": round((unauthorized / max(total, 1)) * 100, 1),
    }


@router.get("/tracks")
async def live_drone_tracks(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    ds = getattr(request.app.state, "detection_service", None)
    tracks = ds.get_live_tracks("drone") if ds else []
    swarm_cameras = ds.swarm_cameras() if ds else []
    return {
        "active_tracks": len(tracks),
        "swarm_detected": bool(swarm_cameras),
        "swarm_cameras": swarm_cameras,
        "tracks": tracks,
    }
