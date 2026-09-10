"""Main FastAPI application entrypoint."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure workspace root and backend directory are in sys.path
_current_dir = Path(__file__).resolve().parent
_backend_dir = _current_dir.parent
_root_dir = _backend_dir.parent

for path in (_root_dir, _backend_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from backend.app.core.config import settings
    from backend.app.core.logging import get_logger, setup_logging
    from backend.app.core.init_db import init_db
    from backend.app.api.health import router as health_router
    from backend.app.api.documents import router as documents_router
    from backend.app.api.rag import router as rag_router
    from backend.app.api.chat import router as chat_router
except ImportError:
    from app.core.config import settings
    from app.core.logging import get_logger, setup_logging
    from app.core.init_db import init_db
    from app.api.health import router as health_router
    from app.api.documents import router as documents_router
    from app.api.rag import router as rag_router
    from app.api.chat import router as chat_router

# Configure logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""

    logger.info(
        "Starting %s v%s in %s environment",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
    )

    try:
        init_db()
        logger.info("Database initialization completed successfully.")
    except Exception:
        logger.exception("Database initialization failed.")
        raise

    yield

    logger.info("Shutting down %s", settings.PROJECT_NAME)


# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for the Advanced RAG & Agentic Document QA System",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(rag_router)
app.include_router(chat_router)


@app.get("/", tags=["Root"], summary="Root API Information")
async def root():
    """Root endpoint providing system metadata, version, and documentation links."""

    return {
        "title": settings.PROJECT_NAME,
        "service": settings.SERVICE_NAME,
        "version": settings.VERSION,
        "status": "active",
        "docs_url": "/docs",
        "health_url": "/health",
        "rag_search_url": "/rag/search",
    }