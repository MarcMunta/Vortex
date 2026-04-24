from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import ControlDependencies
from .routers.autonomy import build_autonomy_router
from .routers.core import build_core_router
from .routers.multimodal import build_multimodal_router
from .routers.training import build_training_router

_CONTROL_CORS_ORIGINS = [
    "http://127.0.0.1:4173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


def create_control_app(deps: ControlDependencies) -> FastAPI:
    app = FastAPI(title="Vortex Control", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CONTROL_CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_core_router(deps))
    app.include_router(build_multimodal_router(deps))
    app.include_router(build_training_router(deps))
    app.include_router(build_autonomy_router(deps))
    return app
