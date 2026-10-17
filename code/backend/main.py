import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_BACKEND_DIR)
for _p in (_CODE_DIR, _BACKEND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging
from contextlib import asynccontextmanager

import uvicorn
from api.routes import alerts, analytics, auth, cameras, drones, geofences, models
from api.routes.auth import user_from_token
from api.websocket import manager
from config.settings import ensure_directories, settings
from database.database import Base, SessionLocal, engine
from database.models import Camera, CameraStatus
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from services.detection_service import DetectionService, camera_runtime_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("quantumfence")

detection_service: DetectionService = None


async def _start_active_cameras(ds: DetectionService) -> None:
    db = SessionLocal()
    try:
        active = (
            db.query(Camera)
            .filter(Camera.is_active == True, Camera.status != CameraStatus.DISABLED)
            .all()
        )
        logger.info(f"Auto-starting {len(active)} active camera(s)...")
        for cam in active:
            await ds.start_camera(
                camera_id=cam.id,
                stream_url=cam.stream_url or "simulated",
                config=camera_runtime_config(cam),
            )
    except Exception as e:
        logger.error(f"Error auto-starting cameras: {e}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detection_service

    logger.info("Starting QuantumFence...")

    ensure_directories()

    Base.metadata.create_all(bind=engine)

    detection_service = DetectionService()
    await detection_service.initialize()
    app.state.detection_service = detection_service

    await _start_active_cameras(detection_service)

    logger.info("QuantumFence operational")
    yield

    logger.info("Shutting down QuantumFence...")
    if detection_service:
        await detection_service.shutdown()


app = FastAPI(
    title="QuantumFence",
    description="Quantum-Accelerated Perimeter Defense AI System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

os.makedirs(settings.SNAPSHOTS_DIR, exist_ok=True)
app.mount(
    "/snapshots",
    StaticFiles(directory=settings.SNAPSHOTS_DIR),
    name="snapshots",
)

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(drones.router, prefix="/api/drones", tags=["Drone Detection"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(geofences.router, prefix="/api/geofences", tags=["Geofences"])
app.include_router(models.router, prefix="/api/models", tags=["AI Models"])

if settings.PROMETHEUS_ENABLED:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(
            app, endpoint="/metrics", include_in_schema=False
        )
    except ImportError:
        logger.warning(
            "prometheus_fastapi_instrumentator not installed, /metrics disabled"
        )


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket, client_id: str, token: str = Query(default=None)
):
    auth_db = SessionLocal()
    try:
        user = user_from_token(token, auth_db)
    finally:
        auth_db.close()
    if user is None:
        await websocket.close(code=4401)
        return

    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get("type", "ping")

            if event_type == "ping":
                await manager.send_personal_message({"type": "pong"}, websocket)

            elif event_type == "subscribe_camera":
                cam_id = data.get("camera_id")
                manager.subscribe_camera(client_id, str(cam_id))
                await manager.send_personal_message(
                    {"type": "subscribed", "camera_id": cam_id}, websocket
                )

            elif event_type == "unsubscribe_camera":
                cam_id = data.get("camera_id")
                manager.unsubscribe_camera(client_id, str(cam_id))
                await manager.send_personal_message(
                    {"type": "unsubscribed", "camera_id": cam_id}, websocket
                )

            elif event_type == "start_camera":
                ds = getattr(app.state, "detection_service", None)
                cam_id = data.get("camera_id")
                if ds and isinstance(cam_id, int):
                    db = SessionLocal()
                    try:
                        cam = db.query(Camera).filter(Camera.id == cam_id).first()
                        params = (
                            (cam.stream_url or "simulated", camera_runtime_config(cam))
                            if cam and cam.is_active
                            else None
                        )
                    finally:
                        db.close()
                    if params:
                        await ds.start_camera(cam_id, params[0], params[1])

    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
        logger.info(f"WS client disconnected: {client_id}")
    except Exception as e:
        manager.disconnect(websocket, client_id)
        logger.warning(f"WS client {client_id} closed after error: {e}")


@app.get("/")
async def root():
    return {
        "system": "QuantumFence",
        "version": "1.0.0",
        "status": "operational",
        "description": "Quantum-Accelerated Perimeter Defense AI System",
    }


@app.get("/health")
async def health():
    ds = getattr(app.state, "detection_service", None)
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "database": "connected",
            "ai_models": ds.model_health() if ds else "unavailable",
            "detection_service": "running" if ds else "stopped",
            "active_cameras": len(ds.camera_processors) if ds else 0,
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
