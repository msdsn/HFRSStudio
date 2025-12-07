"""
Recommendation API endpoints with streaming support.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import asyncio

from api.users import get_current_user
from db.supabase import get_supabase_client
from workflows.recommendation import run_recommendation_workflow, stream_recommendation_workflow

router = APIRouter()


class RecommendationRequest(BaseModel):
    """Request for food recommendations."""
    query: Optional[str] = None  # Natural language query
    num_recommendations: int = 10
    include_explanations: bool = True


class FoodRecommendation(BaseModel):
    """Single food recommendation."""
    food_id: str
    food_name: str
    category: str
    health_score: float
    match_reasons: List[str]
    nutrients: Dict[str, float]
    explanation: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Full recommendation response."""
    recommendations: List[FoodRecommendation]
    agent_outputs: Dict[str, Any]
    workflow_summary: str


@router.post("/generate")
async def generate_recommendations(
    request: RecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate food recommendations using the full agent workflow."""
    supabase = get_supabase_client()
    
    try:
        # Get user profile
        profile_response = supabase.table("profiles").select("*").eq("id", current_user["id"]).single().execute()
        
        if profile_response.data is None:
            raise HTTPException(status_code=400, detail="Please complete your profile first")
        
        user_profile = profile_response.data
        
        # Run the LangGraph workflow
        result = await run_recommendation_workflow(
            user_profile=user_profile,
            query=request.query,
            num_recommendations=request.num_recommendations,
            include_explanations=request.include_explanations
        )
        
        # Log the recommendation
        supabase.table("recommendations_log").insert({
            "user_id": current_user["id"],
            "recommendations": result["recommendations"],
            "agent_outputs": result["agent_outputs"]
        }).execute()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream")
async def generate_recommendations_stream(
    request: RecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Stream recommendations with real-time agent updates."""
    supabase = get_supabase_client()
    
    try:
        # Get user profile
        profile_response = supabase.table("profiles").select("*").eq("id", current_user["id"]).single().execute()
        
        if profile_response.data is None:
            raise HTTPException(status_code=400, detail="Please complete your profile first")
        
        user_profile = profile_response.data
        
        async def event_generator():
            """Generate Server-Sent Events for workflow progress."""
            async for event in stream_recommendation_workflow(
                user_profile=user_profile,
                query=request.query,
                num_recommendations=request.num_recommendations
            ):
                yield f"data: {json.dumps(event)}\n\n"
            
            yield "data: {\"type\": \"done\"}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_recommendation_history(
    current_user: dict = Depends(get_current_user),
    limit: int = 20
):
    """Get user's recommendation history."""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table("recommendations_log").select("*").eq("user_id", current_user["id"]).order("created_at", desc=True).limit(limit).execute()
        
        return {"history": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{recommendation_id}")
async def get_recommendation(
    recommendation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific recommendation by ID."""
    supabase = get_supabase_client()
    
    try:
        response = supabase.table("recommendations_log").select("*").eq("id", recommendation_id).eq("user_id", current_user["id"]).single().execute()
        
        if response.data is None:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
