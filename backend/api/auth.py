"""
Authentication API endpoints using Supabase Auth.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from typing import Optional

from db.supabase import get_supabase_client, get_supabase_anon_client

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


class LinkEmailRequest(BaseModel):
    """Link email to anonymous account request."""
    email: EmailStr
    password: str


class AnonymousAuthResponse(BaseModel):
    """Anonymous authentication response."""
    access_token: str
    refresh_token: str
    user_id: str
    is_anonymous: bool


@router.post("/anonymous", response_model=AnonymousAuthResponse)
async def anonymous_login():
    """Create anonymous user session.
    
    Note: This endpoint uses the anon key, not the service key.
    """
    supabase = get_supabase_anon_client()
    
    try:
        # Sign in anonymously with Supabase
        response = supabase.auth.sign_in_anonymously()
        
        if response.user is None:
            raise HTTPException(status_code=400, detail="Anonymous sign-in failed")
        
        if response.session is None:
            raise HTTPException(status_code=400, detail="No session created")
        
        return AnonymousAuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            user_id=response.user.id,
            is_anonymous=True
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Anonymous sign-in failed: {str(e)}")


@router.post("/link-email", response_model=AuthResponse)
async def link_email(
    request: LinkEmailRequest,
    authorization: Optional[str] = Header(None)
):
    """Link email and password to current anonymous account.
    
    Requires authentication via Bearer token in Authorization header.
    Updates both Supabase Auth and profiles table.
    """
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.replace('Bearer ', '')
    supabase = get_supabase_client()
    
    try:
        # Verify current user
        user_response = supabase.auth.get_user(token)
        if user_response.user is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_id = user_response.user.id
        
        # Set the session for this request
        supabase.auth.set_session(token, token)
        
        # Update the user with email and password in Supabase Auth
        response = supabase.auth.update_user({
            "email": request.email,
            "password": request.password
        })
        
        if response.user is None:
            raise HTTPException(status_code=400, detail="Failed to link email")
        
        # Update email in profiles table as well
        try:
            supabase.table("profiles").update({
                "email": request.email,
                "updated_at": "now()"
            }).eq("id", user_id).execute()
        except Exception as profile_error:
            # If profile doesn't exist, create it
            try:
                supabase.table("profiles").insert({
                    "id": user_id,
                    "email": request.email,
                    "health_tags": {},
                    "dietary_restrictions": [],
                    "allergies": [],
                    "cuisine_preferences": []
                }).execute()
            except Exception as insert_error:
                print(f"Warning: Failed to update/create profile: {insert_error}")
        
        # Get fresh session after update
        session_response = supabase.auth.get_session()
        
        if session_response.session is None:
            raise HTTPException(status_code=400, detail="Failed to get session after linking")
        
        return AuthResponse(
            access_token=session_response.session.access_token,
            refresh_token=session_response.session.refresh_token,
            user_id=response.user.id,
            email=response.user.email or ""
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to link email: {str(e)}")


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
