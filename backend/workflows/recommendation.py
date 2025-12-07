"""
LangGraph Recommendation Workflow.
Multi-agent workflow for generating health-aware food recommendations.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated, AsyncGenerator
from langgraph.graph import StateGraph, END
import asyncio
import json

from agents.nutritionist import NutritionistAgent
from agents.health_advisor import HealthAdvisorAgent
from agents.personalizer import PersonalizerAgent
from agents.critic import CriticAgent
from agents.explainer import ExplainerAgent
from models.inference import get_recommendation_model


class RecommendationState(TypedDict):
    """State schema for the recommendation workflow."""
    # Input
    user_profile: Dict[str, Any]
    user_query: Optional[str]
    num_recommendations: int
    food_history: List[Dict[str, Any]]
    
    # Model predictions
    model_predictions: List[Dict[str, Any]]
    
    # Agent outputs
    nutritionist_analysis: Dict[str, Any]
    personalizer_analysis: Dict[str, Any]
    health_analysis: Dict[str, Any]
    critic_evaluation: Dict[str, Any]
    final_explanation: Dict[str, Any]
    
    # Final output
    final_recommendations: List[Dict[str, Any]]
    workflow_summary: str
    
    # Workflow metadata
    current_step: str
    errors: List[str]


def create_initial_state(
    user_profile: Dict[str, Any],
    query: Optional[str] = None,
    num_recommendations: int = 10,
    food_history: Optional[List[Dict[str, Any]]] = None
) -> RecommendationState:
    """Create initial workflow state."""
    return RecommendationState(
        user_profile=user_profile,
        user_query=query,
        num_recommendations=num_recommendations,
        food_history=food_history or [],
        model_predictions=[],
        nutritionist_analysis={},
        personalizer_analysis={},
        health_analysis={},
        critic_evaluation={},
        final_explanation={},
        final_recommendations=[],
        workflow_summary="",
        current_step="start",
        errors=[]
    )


# Node functions
async def model_prediction_node(state: RecommendationState) -> Dict[str, Any]:
    """Generate initial predictions using MOPI-HFRS model."""
    try:
        print("[MODEL] Starting model prediction...")
        model = get_recommendation_model()
        
        # Compute user embedding
        print("[MODEL] Computing user embedding...")
        user_embedding = model.compute_user_embedding(state['user_profile'])
        print(f"[MODEL] User embedding shape: {user_embedding.shape}")
        
        # Get health filter from user profile
        health_filter = state['user_profile'].get('health_tags', {})
        print(f"[MODEL] Health filter: {health_filter}")
        
        # Get excluded foods from history
        exclude_ids = [item.get('food_id') for item in state['food_history']]
        
        # Generate recommendations - get more to account for filtering invalid foods
        # We need 3-4x more because some might be "Food #..." format and will be filtered
        k_recommendations = max(state['num_recommendations'] * 3, 15)
        print(f"[MODEL] Generating {k_recommendations} recommendations (to filter invalid foods)...")
        recommendations = model.recommend(
            user_embedding=user_embedding,
            k=k_recommendations,  # Get more for filtering invalid "Food #..." entries
            exclude_food_ids=exclude_ids,
            health_filter=health_filter
        )
        
        print(f"[MODEL] Got {len(recommendations)} recommendations")
        if recommendations:
            print(f"[MODEL] First rec: {recommendations[0].get('food_name', 'Unknown')}")
        
        return {
            "model_predictions": recommendations,
            "current_step": "model_prediction_complete"
        }
    except Exception as e:
        import traceback
        print(f"[MODEL] ERROR: {str(e)}")
        traceback.print_exc()
        return {
            "errors": state['errors'] + [f"Model prediction error: {str(e)}"],
            "model_predictions": [],
            "current_step": "model_prediction_error"
        }


async def nutritionist_node(state: RecommendationState) -> Dict[str, Any]:
    """Run nutritionist agent analysis."""
    try:
        agent = NutritionistAgent()
        result = await agent.run(state)
        
        return {
            "nutritionist_analysis": {
                "success": result.success,
                "analysis": result.analysis,
                "data": result.data,
                "confidence": result.confidence
            },
            "current_step": "nutritionist_complete"
        }
    except Exception as e:
        return {
            "errors": state['errors'] + [f"Nutritionist error: {str(e)}"],
            "nutritionist_analysis": {"success": False, "error": str(e)},
            "current_step": "nutritionist_error"
        }


async def personalizer_node(state: RecommendationState) -> Dict[str, Any]:
    """Run personalizer agent analysis."""
    try:
        agent = PersonalizerAgent()
        result = await agent.run(state)
        
        return {
            "personalizer_analysis": {
                "success": result.success,
                "analysis": result.analysis,
                "data": result.data,
                "confidence": result.confidence
            },
            "current_step": "personalizer_complete"
        }
    except Exception as e:
        return {
            "errors": state['errors'] + [f"Personalizer error: {str(e)}"],
            "personalizer_analysis": {"success": False, "error": str(e)},
            "current_step": "personalizer_error"
        }


async def health_advisor_node(state: RecommendationState) -> Dict[str, Any]:
    """Run health advisor agent analysis."""
    try:
        agent = HealthAdvisorAgent()
        result = await agent.run(state)
        
        return {
            "health_analysis": {
                "success": result.success,
                "analysis": result.analysis,
                "data": result.data,
                "confidence": result.confidence
            },
            "current_step": "health_advisor_complete"
        }
    except Exception as e:
        return {
            "errors": state['errors'] + [f"Health advisor error: {str(e)}"],
            "health_analysis": {"success": False, "error": str(e)},
            "current_step": "health_advisor_error"
        }


async def critic_node(state: RecommendationState) -> Dict[str, Any]:
    """Run critic agent evaluation."""
    try:
        agent = CriticAgent()
        result = await agent.run(state)
        
        return {
            "critic_evaluation": {
                "success": result.success,
                "analysis": result.analysis,
                "data": result.data,
                "confidence": result.confidence
            },
            "current_step": "critic_complete"
        }
    except Exception as e:
        return {
            "errors": state['errors'] + [f"Critic error: {str(e)}"],
            "critic_evaluation": {"success": False, "error": str(e)},
            "current_step": "critic_error"
        }


async def explainer_node(state: RecommendationState) -> Dict[str, Any]:
    """Run explainer agent to generate final explanations."""
    try:
        agent = ExplainerAgent()
        result = await agent.run(state)
        
        return {
            "final_explanation": {
                "success": result.success,
                "analysis": result.analysis,
                "data": result.data,
                "confidence": result.confidence
            },
            "current_step": "explainer_complete"
        }
    except Exception as e:
        return {
            "errors": state['errors'] + [f"Explainer error: {str(e)}"],
            "final_explanation": {"success": False, "error": str(e)},
            "current_step": "explainer_error"
        }


async def finalize_node(state: RecommendationState) -> Dict[str, Any]:
    """Finalize recommendations and create output."""
    try:
        print("[FINALIZE] Starting finalize node...")
        
        # Get model predictions first
        model_predictions = state.get('model_predictions', [])
        print(f"[FINALIZE] Model predictions count: {len(model_predictions)}")
        
        # Get critic's final evaluation
        critic_data = state.get('critic_evaluation', {}).get('data', {})
        final_foods = critic_data.get('foods_final_evaluation', [])
        print(f"[FINALIZE] Critic final_foods count: {len(final_foods)}")
        
        # Get explanations
        explanation_data = state.get('final_explanation', {}).get('data', {})
        explanations = explanation_data.get('recommendations_explained', [])
        print(f"[FINALIZE] Explanations count: {len(explanations)}")
        
        # Helper function to check if food has valid name (not "Food #...")
        def is_valid_food(food: Dict[str, Any]) -> bool:
            """Check if food has a valid name (not generic 'Food #...' format)."""
            food_name = food.get('food_name', '')
            # Skip foods with generic "Food #" format
            if isinstance(food_name, str) and food_name.startswith('Food #'):
                return False
            return True
        
        # Merge model predictions with evaluations and explanations
        final_recommendations = []
        num_needed = state['num_recommendations']
        
        # If we have model predictions, use them as base
        if model_predictions:
            # Process more predictions than needed to account for filtering
            processed_count = 0
            for food in model_predictions:
                # Stop if we have enough valid recommendations
                if len(final_recommendations) >= num_needed:
                    break
                
                # Skip invalid foods (generic "Food #" format)
                if not is_valid_food(food):
                    print(f"[FINALIZE] Skipping invalid food: {food.get('food_name')}")
                    continue
                
                # Find matching evaluation
                food_eval = next(
                    (f for f in final_foods if f.get('food_id') == food.get('food_id')),
                    {}
                )
                
                # Find matching explanation
                food_explanation = next(
                    (e for e in explanations if e.get('food_id') == food.get('food_id')),
                    {}
                )
                
                # Skip excluded foods (but only if we have an explicit exclude)
                if food_eval.get('decision') == 'exclude':
                    print(f"[FINALIZE] Skipping excluded food: {food.get('food_name')}")
                    continue
                
                final_food = {
                    **food,
                    "final_score": food_eval.get('final_score', food.get('score', 0)),
                    "rank": len(final_recommendations) + 1,
                    "health_status": food_eval.get('decision', 'include'),
                    "strengths": food_eval.get('combined_strengths', []),
                    "concerns": food_eval.get('combined_concerns', []),
                    "explanation": {
                        "headline": food_explanation.get('headline', f"Great choice: {food.get('food_name', 'Food')}!"),
                        "description": food_explanation.get('explanation', 'A nutritious option for your diet.'),
                        "health_benefits": food_explanation.get('health_benefits', ['Supports overall health']),
                        "serving_suggestion": food_explanation.get('serving_suggestion', 'Enjoy as part of a balanced meal.'),
                        "fun_fact": food_explanation.get('fun_fact', '')
                    }
                }
                final_recommendations.append(final_food)
                processed_count += 1
            
            # If we don't have enough valid recommendations, try to get more from model
            if len(final_recommendations) < num_needed and len(model_predictions) < num_needed * 3:
                print(f"[FINALIZE] Only {len(final_recommendations)} valid recommendations, need {num_needed}")
                print(f"[FINALIZE] Will include remaining valid foods even if less than requested")
        
        print(f"[FINALIZE] Final recommendations count: {len(final_recommendations)}")
        
        # Create workflow summary
        summary_parts = []
        if explanation_data.get('welcome_message'):
            summary_parts.append(explanation_data['welcome_message'])
        if explanation_data.get('overall_summary'):
            summary_parts.append(explanation_data['overall_summary'])
        if explanation_data.get('dietary_tip'):
            summary_parts.append(f"💡 Tip: {explanation_data['dietary_tip']}")
        if explanation_data.get('encouragement'):
            summary_parts.append(explanation_data['encouragement'])
        
        workflow_summary = "\n\n".join(summary_parts) if summary_parts else "Recommendations generated successfully!"
        
        return {
            "final_recommendations": final_recommendations,
            "workflow_summary": workflow_summary,
            "current_step": "complete"
        }
    except Exception as e:
        return {
            "errors": state['errors'] + [f"Finalization error: {str(e)}"],
            "final_recommendations": state.get('model_predictions', [])[:state['num_recommendations']],
            "workflow_summary": "Recommendations generated with some errors.",
            "current_step": "complete_with_errors"
        }


def create_recommendation_workflow() -> StateGraph:
    """Create the LangGraph recommendation workflow."""
    
    # Create workflow graph
    workflow = StateGraph(RecommendationState)
    
    # Add nodes
    workflow.add_node("model_prediction", model_prediction_node)
    workflow.add_node("nutritionist", nutritionist_node)
    workflow.add_node("personalizer", personalizer_node)
    workflow.add_node("health_advisor", health_advisor_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("explainer", explainer_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("model_prediction")
    
    # Add edges - workflow flow
    # After model prediction, run nutritionist and personalizer in parallel (conceptually)
    workflow.add_edge("model_prediction", "nutritionist")
    workflow.add_edge("nutritionist", "personalizer")
    
    # After parallel analysis, run health advisor
    workflow.add_edge("personalizer", "health_advisor")
    
    # Critic evaluates all analyses
    workflow.add_edge("health_advisor", "critic")
    
    # Explainer creates final explanations
    workflow.add_edge("critic", "explainer")
    
    # Finalize creates output
    workflow.add_edge("explainer", "finalize")
    
    # End after finalization
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# Singleton workflow instance
_workflow_instance = None


def get_workflow():
    """Get or create the recommendation workflow."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = create_recommendation_workflow()
    return _workflow_instance


async def run_recommendation_workflow(
    user_profile: Dict[str, Any],
    query: Optional[str] = None,
    num_recommendations: int = 10,
    include_explanations: bool = True
) -> Dict[str, Any]:
    """
    Run the full recommendation workflow.
    
    Args:
        user_profile: User profile dictionary
        query: Optional natural language query
        num_recommendations: Number of recommendations to generate
        include_explanations: Whether to include detailed explanations
        
    Returns:
        Dictionary with recommendations and agent outputs
    """
    workflow = get_workflow()
    
    # Create initial state
    initial_state = create_initial_state(
        user_profile=user_profile,
        query=query,
        num_recommendations=num_recommendations
    )
    
    # Run workflow
    final_state = await workflow.ainvoke(initial_state)
    
    # Extract results
    return {
        "recommendations": final_state.get('final_recommendations', []),
        "workflow_summary": final_state.get('workflow_summary', ''),
        "agent_outputs": {
            "nutritionist": final_state.get('nutritionist_analysis', {}),
            "personalizer": final_state.get('personalizer_analysis', {}),
            "health_advisor": final_state.get('health_analysis', {}),
            "critic": final_state.get('critic_evaluation', {}),
            "explainer": final_state.get('final_explanation', {})
        },
        "errors": final_state.get('errors', [])
    }


async def stream_recommendation_workflow(
    user_profile: Dict[str, Any],
    query: Optional[str] = None,
    num_recommendations: int = 10
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream the recommendation workflow with real-time updates.
    
    Args:
        user_profile: User profile dictionary
        query: Optional query
        num_recommendations: Number of recommendations
        
    Yields:
        Event dictionaries with workflow progress
    """
    workflow = get_workflow()
    
    initial_state = create_initial_state(
        user_profile=user_profile,
        query=query,
        num_recommendations=num_recommendations
    )
    
    # Stream workflow execution
    async for event in workflow.astream(initial_state):
        for node_name, node_output in event.items():
            yield {
                "type": "node_complete",
                "node": node_name,
                "step": node_output.get('current_step', ''),
                "data": {
                    k: v for k, v in node_output.items()
                    if k not in ['current_step', 'errors']
                }
            }
    
    # Final complete event
    yield {"type": "workflow_complete"}
