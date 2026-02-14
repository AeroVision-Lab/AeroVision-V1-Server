"""
FastAPI application entry point for Aerovision-V1-Server.
"""

import contextlib
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import api_router
from app.core.config import get_settings
from app.core.logging import logger, setup_logging
from app.core.exceptions import AerovisionException
from app.inference import InferenceFactory

# Get settings
settings = get_settings()

# Setup logging
setup_logging(level=settings.log_level, format_type=settings.log_format)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-powered aviation photography review API",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)


# Include API routes
app.include_router(api_router)


# Exception handlers
@app.exception_handler(AerovisionException)
async def aerovision_exception_handler(request: Request, exc: AerovisionException):
    """统一处理 Aerovision 异常"""
    logger.warning(f"Aerovision 异常: {exc.code} - {exc.message}")

    status_code_map = {
        "IMAGE_LOAD_ERROR": 400,
        "VALIDATION_ERROR": 422,
        "MODEL_NOT_LOADED": 503,
        "INFERENCE_ERROR": 500,
        "RATE_LIMIT_ERROR": 429,
    }

    status_code = status_code_map.get(exc.code, 500)

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
            },
            "detail": str(exc) if settings.debug else None,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.exception(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "内部服务错误",
            "detail": str(exc) if settings.debug else None,
        },
    )


# Lifespan events
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info(f"Starting {settings.app_name} v{settings.version}")

    # Preload models if enabled
    if settings.preload_models and settings.environment != "test":
        try:
            InferenceFactory.preload_models()
        except Exception as e:
            logger.warning(f"Failed to preload models: {e}")

    yield

    logger.info(f"Shutting down {settings.app_name}")


app.router.lifespan_context = lifespan


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1
    )
