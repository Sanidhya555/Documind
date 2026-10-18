"""FastAPI application factory."""
import os

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title="Enterprise Document Intelligence API",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — restricted to configured origins (not wildcard in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

app.include_router(router, prefix="/api")

# Serve React build as static files if present
_frontend_dist = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
)
_index_file = os.path.join(_frontend_dist, "index.html")

if os.path.exists(_index_file):
    _assets_dir = os.path.join(_frontend_dist, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="frontend-assets")

    @app.get("/")
    def serve_ui():
        return FileResponse(_index_file)
else:
    @app.get("/")
    def api_root():
        return {
            "name": "Enterprise Document Intelligence API",
            "version": "3.0.0",
            "docs": "/docs",
            "health": "/api/health",
        }


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    # Leave room for multipart boundaries and headers around a file. The upload
    # route separately enforces MAX_UPLOAD_BYTES against the file itself.
    max_request_bytes = settings.MAX_UPLOAD_BYTES + 1024 * 1024
    if cl and int(cl) > max_request_bytes:
        raise HTTPException(status_code=413, detail="Request body too large (max upload size is 10 MB)")
    return await call_next(request)
