"""
Backend configuration using pydantic-settings.
Loads environment variables from .env file.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App settings
    app_name: str = "HFRS API"
    debug: bool = False
    
    # Supabase
    supabase_url: str
    supabase_service: str  # Service key for backend
    
    # LLM API Keys
    gemini_key: str
    openai_key: str
    
    # Model settings
    model_checkpoint: str = "checkpoints/best_model.pt"
    device: str = "cpu"  # or "cuda" if available
    
    # LLM settings
    primary_llm: str = "gemini"  # gemini or openai
    fallback_llm: str = "openai"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    
    # CORS - can be set via CORS_ORIGINS env var (comma-separated)
    cors_origins: list[str] = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://localhost",       # Docker nginx (port 80)
        "http://localhost:8000",  # Backend itself (for testing)
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override CORS origins from environment variable if set
        cors_env = os.getenv("CORS_ORIGINS")
        if cors_env:
            self.cors_origins = [origin.strip() for origin in cors_env.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
