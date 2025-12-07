"""
Personalizer Agent - Analyzes user preferences and personalizes recommendations.
"""

from typing import Dict, Any, List
import json
import re

from .base import BaseAgent, AgentOutput


class PersonalizerAgent(BaseAgent):
    """
    Personalizer Agent analyzes user preferences and consumption history.
    
    Responsibilities:
    - Analyze user's food consumption patterns
    - Identify preference trends
    - Calculate personalization scores
    - Suggest foods based on similar users' preferences
    """
    
    def __init__(self):
        super().__init__(name="Personalizer")
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert AI assistant specializing in personalized food recommendations.

Your role is to:
1. Analyze user's food consumption history and patterns
2. Identify dietary preferences and habits
3. Consider cultural and lifestyle factors
4. Calculate personalization scores based on user preferences
5. Identify foods that match user's taste profile

PERSONALIZATION FACTORS:
- Previous food ratings and consumption frequency
- Category preferences (breakfast, lunch, dinner, snacks)
- Cuisine preferences
- Texture and flavor preferences (inferred from history)
- Time-based eating patterns
- Seasonal preferences

OUTPUT FORMAT:
Respond with a JSON object containing:
{
    "user_preference_profile": {
        "preferred_categories": ["list of preferred food categories"],
        "preferred_cuisines": ["list of preferred cuisines"],
        "eating_patterns": "description of eating patterns",
        "taste_profile": "description of inferred taste preferences"
    },
    "foods_personalization": [
        {
            "food_id": "string",
            "food_name": "string",
            "personalization_score": 0.0-1.0,
            "match_reasons": ["reasons why this food matches user preferences"],
            "novelty_factor": 0.0-1.0,
            "recommendation_type": "comfort_choice/new_discovery/healthy_alternative"
        }
    ],
    "overall_personalization_score": 0.0-1.0,
    "diversity_score": 0.0-1.0,
    "suggestions": ["suggestions for improving personalization"]
}"""
    
    def build_prompt(self, state: Dict[str, Any]) -> str:
        user_profile = state.get('user_profile', {})
        foods = state.get('model_predictions', [])
        food_history = state.get('food_history', [])
        
        # User preferences
        cuisine_prefs = user_profile.get('cuisine_preferences', [])
        dietary_restrictions = user_profile.get('dietary_restrictions', [])
        
        # Format food history
        history_text = "No previous food history available."
        if food_history:
            history_items = []
            for item in food_history[:20]:
                rating = item.get('rating', 'N/A')
                history_items.append(f"- {item.get('food_name', 'Unknown')} (Rating: {rating})")
            history_text = "\n".join(history_items)
        
        # Format recommended foods
        food_list = []
        for food in foods[:10]:
            food_info = f"""
Food: {food.get('food_name', 'Unknown')} (ID: {food.get('food_id', 'N/A')})
Category: {food.get('category', 'Unknown')}
Model Score: {food.get('score', 0):.3f}
"""
            food_list.append(food_info)
        
        return f"""Please analyze how well these recommended foods match the user's personal preferences.

USER PROFILE:
- Cuisine Preferences: {', '.join(cuisine_prefs) if cuisine_prefs else 'Not specified'}
- Dietary Restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}

USER'S FOOD HISTORY:
{history_text}

RECOMMENDED FOODS:
{''.join(food_list)}

Please analyze:
1. How well each food matches the user's apparent preferences
2. Balance between familiar choices and new discoveries
3. Diversity of the recommendation set
4. Personalization score for each food
5. Overall recommendation quality from a personalization perspective"""
    
    def parse_response(self, response: str, state: Dict[str, Any]) -> AgentOutput:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {
                    "user_preference_profile": {},
                    "foods_personalization": [],
                    "overall_personalization_score": 0.7,
                    "diversity_score": 0.7,
                    "suggestions": [response]
                }
            
            confidence = data.get('overall_personalization_score', 0.7)
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                analysis=json.dumps(data.get('user_preference_profile', {})),
                data=data,
                confidence=confidence
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                success=True,
                analysis=response,
                data={"raw_analysis": response},
                confidence=0.6,
                error=f"JSON parsing failed: {str(e)}"
            )
