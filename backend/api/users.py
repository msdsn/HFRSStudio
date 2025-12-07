"""
User profile API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from db.supabase import get_supabase_client
from db.schemas import UserProfile, HealthProfile, UpdateProfileRequest

router = APIRouter()


async def get_current_user(authorization: str = Header(...)) -> dict:
    """Verify JWT token and return user info."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    supabase = get_supabase_client()
    
    try:
        response = supabase.auth.get_user(token)
        if response.user is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"id": response.user.id, "email": response.user.email}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")


@router.get("/me", response_model=UserProfile)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get current user's profile."""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table("profiles").select("*").eq("id", current_user["id"]).single().execute()
        
        if response.data is None:
            # Create default profile if doesn't exist
            default_profile = {
                "id": current_user["id"],
                "email": current_user["email"],
                "health_tags": {},
                "dietary_restrictions": [],
                "allergies": [],
                "cuisine_preferences": []
            }
            supabase.table("profiles").insert(default_profile).execute()
            return UserProfile(**default_profile)
        
        return UserProfile(**response.data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/me", response_model=UserProfile)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update current user's profile."""
    supabase = get_supabase_client()
    
    try:
        update_data = request.model_dump(exclude_unset=True)
        update_data["updated_at"] = "now()"
        
        response = supabase.table("profiles").update(update_data).eq("id", current_user["id"]).execute()
        
        return UserProfile(**response.data[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/me/health-profile", response_model=UserProfile)
async def update_health_profile(
    health_profile: HealthProfile,
    current_user: dict = Depends(get_current_user)
):
    """Update user's health profile (onboarding step)."""
    supabase = get_supabase_client()
    
    try:
        # Convert health profile to health tags format (NHANES style)
        health_tags = health_profile.to_health_tags()
        
        update_data = {
            "gender": health_profile.gender,
            "age": health_profile.age,
            "race": health_profile.race,
            "education": health_profile.education,
            "household_income": health_profile.household_income,
            "health_tags": health_tags,
            "dietary_restrictions": health_profile.dietary_restrictions,
            "allergies": health_profile.allergies,
            "onboarding_completed": True,
            "updated_at": "now()"
        }
        
        response = supabase.table("profiles").update(update_data).eq("id", current_user["id"]).execute()
        
        return UserProfile(**response.data[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/me/preferences", response_model=UserProfile)
async def update_preferences(
    preferences: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Update user's food preferences."""
    supabase = get_supabase_client()
    
    try:
        update_data = {
            "cuisine_preferences": preferences.get("cuisines", []),
            "disliked_foods": preferences.get("disliked_foods", []),
            "favorite_foods": preferences.get("favorite_foods", []),
            "updated_at": "now()"
        }
        
        response = supabase.table("profiles").update(update_data).eq("id", current_user["id"]).execute()
        
        return UserProfile(**response.data[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/me/complete-onboarding", response_model=UserProfile)
async def complete_onboarding(
    current_user: dict = Depends(get_current_user)
):
    """Mark onboarding as completed."""
    supabase = get_supabase_client()
    
    try:
        update_data = {
            "onboarding_completed": True,
            "updated_at": "now()"
        }
        
        response = supabase.table("profiles").update(update_data).eq("id", current_user["id"]).execute()
        
        return UserProfile(**response.data[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me/food-history")
async def get_food_history(
    current_user: dict = Depends(get_current_user),
    limit: int = 50
):
    """Get user's food consumption history."""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table("food_history").select("*").eq("user_id", current_user["id"]).order("created_at", desc=True).limit(limit).execute()
        
        return {"history": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/me/food-history")
async def add_food_history(
    food_id: str,
    food_name: str,
    rating: Optional[int] = None,
    current_user: dict = Depends(get_current_user)
):
    """Add food to user's history."""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table("food_history").insert({
            "user_id": current_user["id"],
            "food_id": food_id,
            "food_name": food_name,
            "rating": rating
        }).execute()
        
        return {"success": True, "data": response.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
