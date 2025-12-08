"""
Supabase client configuration.
"""

from supabase import create_client, Client
from functools import lru_cache

from config import get_settings


@lru_cache()
def get_supabase_client() -> Client:
    """Get cached Supabase client instance with service key (admin operations)."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service)


@lru_cache()
def get_supabase_anon_client() -> Client:
    """Get cached Supabase client instance with anon key (public auth operations)."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_anon)
