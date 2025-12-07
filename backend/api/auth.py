"""
Authentication API endpoints using Supabase Auth.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional

from db.supabase import get_supabase_client

router = APIRouter()


class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    """Authentication response."""
    access_token: str
    refresh_token: str
    user_id: str
    email: str


class TokenRefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


@router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """Register a new user."""
    supabase = get_supabase_client()
    
    try:
        # Create user with Supabase Auth
        response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {
                "data": {
                    "full_name": request.full_name
                }
            }
        })
        
        if response.user is None:
            raise HTTPException(status_code=400, detail="Registration failed")
        
        return AuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            user_id=response.user.id,
            email=response.user.email
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Login user and return tokens."""
    supabase = get_supabase_client()
    
    try:
        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        if response.user is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        return AuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            user_id=response.user.id,
            email=response.user.email
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(request: TokenRefreshRequest):
    """Refresh access token."""
    supabase = get_supabase_client()
    
    try:
        response = supabase.auth.refresh_session(request.refresh_token)
        
        if response.user is None:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        return AuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            user_id=response.user.id,
            email=response.user.email
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token refresh failed")


@router.post("/logout")
async def logout():
    """Logout user (client should discard tokens)."""
    return {"message": "Logged out successfully"}


@router.post("/forgot-password")
async def forgot_password(email: EmailStr):
    """Send password reset email."""
    supabase = get_supabase_client()
    
    try:
        supabase.auth.reset_password_email(email)
        return {"message": "Password reset email sent"}
    except Exception as e:
        # Don't reveal if email exists
        return {"message": "If the email exists, a reset link has been sent"}
