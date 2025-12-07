"""
HFRS API - Health-aware Food Recommendation System
FastAPI backend with LangGraph multi-agent workflow.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    print("Starting HFRS API...")
    
    # Load ML model on startup (lazy import to avoid circular imports)
    from models.inference import get_recommendation_model
    try:
        model = get_recommendation_model()
        print(f"Model loaded successfully on {settings.device}")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down HFRS API...")


app = FastAPI(
    title=settings.app_name,
    description="AI-powered health-aware food recommendation system with multi-agent LangGraph workflow",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from api import auth, users, recommendations

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["Recommendations"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": settings.app_name}


# Serve static files (frontend build) in production
# Check multiple possible locations for frontend dist
static_paths = [
    os.path.join(os.path.dirname(__file__), "static"),  # Docker: /app/static
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"),  # Local dev: ../frontend/dist
]

# Find the frontend dist directory
frontend_dist = None
for path in static_paths:
    if os.path.exists(path):
        frontend_dist = path
        break

if frontend_dist:
    # Mount static files for assets (js, css, images)
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    
    # Serve index.html for SPA routes (catch-all for non-API routes)
    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Serve index.html for all non-API routes (SPA fallback)."""
        # Check if it's a file request (has extension)
        if "." in full_path.split("/")[-1]:
            file_path = os.path.join(frontend_dist, full_path)
            if os.path.exists(file_path):
                return FileResponse(file_path)
        
        # Return index.html for all other routes (SPA routing)
        return FileResponse(os.path.join(frontend_dist, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
