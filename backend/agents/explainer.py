"""
Explainer Agent - Generates user-friendly explanations for recommendations.
"""

from typing import Dict, Any, List
import json
import re

from .base import BaseAgent, AgentOutput


class ExplainerAgent(BaseAgent):
    """
    Explainer Agent generates clear, user-friendly explanations for recommendations.
    
    Responsibilities:
    - Synthesize all agent analyses into coherent explanations
    - Create personalized, engaging recommendation descriptions
    - Explain health benefits in accessible language
    - Provide actionable dietary advice
    - Promote healthy eating awareness
    """
    
    def __init__(self):
        super().__init__(name="Explainer")
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert AI assistant specializing in communicating nutrition and health information to users.

Your role is to:
1. Transform complex nutritional analyses into user-friendly explanations
2. Create engaging, personalized recommendation descriptions
3. Explain health benefits in simple, accessible language
4. Provide practical dietary advice
5. Promote healthy eating awareness without being preachy

COMMUNICATION GUIDELINES:
- Use simple, everyday language (avoid jargon)
- Be positive and encouraging
- Focus on benefits rather than restrictions
- Make explanations personal and relevant
- Include practical tips when appropriate
- Be concise but informative

TONE:
- Friendly and supportive
- Informative but not overwhelming
- Encouraging healthy choices without judgment
- Culturally sensitive and inclusive

OUTPUT FORMAT:
Respond with a JSON object containing:
{
    "welcome_message": "personalized greeting and summary",
    "recommendations_explained": [
        {
            "food_id": "string",
            "food_name": "string",
            "headline": "catchy one-liner about why this food is recommended",
            "explanation": "2-3 sentence explanation of why this is good for the user",
            "health_benefits": ["list of key health benefits in simple terms"],
            "serving_suggestion": "practical serving or pairing suggestion",
            "fun_fact": "optional interesting fact about the food"
        }
    ],
    "overall_summary": "summary of the recommendation set and its benefits",
    "dietary_tip": "one practical tip for healthy eating",
    "encouragement": "motivational closing message"
}"""
    
    def build_prompt(self, state: Dict[str, Any]) -> str:
        user_profile = state.get('user_profile', {})
        foods = state.get('model_predictions', [])
        critic_evaluation = state.get('critic_evaluation', {})
        nutritionist = state.get('nutritionist_analysis', {})
        health_advisor = state.get('health_analysis', {})
        
        # User info for personalization
        user_name = user_profile.get('full_name', 'there')
        age = user_profile.get('age', '')
        health_needs = [tag.replace('user_', '').replace('_', ' ') 
                       for tag, value in user_profile.get('health_tags', {}).items() if value]
        
        # Get final food list from critic or use original
        critic_data = critic_evaluation.get('data', {})
        final_foods = critic_data.get('foods_final_evaluation', [])
        
        # Format recommendations
        food_list = []
        for i, food in enumerate(foods[:5], 1):  # Top 5 only
            food_info = f"""
{i}. {food.get('food_name', 'Unknown')}
   Category: {food.get('category', 'Unknown')}
   Key Nutrients: {json.dumps(food.get('nutrients', {}))[:200]}
   Health Tags: {list(food.get('health_tags', {}).keys())[:5]}
"""
            food_list.append(food_info)
        
        # Get key insights from other agents
        nutritionist_insight = nutritionist.get('analysis', '')[:300]
        health_insight = health_advisor.get('analysis', '')[:300]
        
        return f"""Please create user-friendly explanations for these food recommendations.

USER CONTEXT:
- Name: {user_name}
- Age: {age}
- Health Focus: {', '.join(health_needs) if health_needs else 'General wellness'}

TOP RECOMMENDATIONS:
{''.join(food_list)}

KEY INSIGHTS:
- Nutritional Perspective: {nutritionist_insight}
- Health Perspective: {health_insight}

Please create:
1. A warm, personalized welcome message
2. Engaging explanations for each food (why it's recommended for THIS user)
3. Simple health benefits anyone can understand
4. Practical serving suggestions
5. An encouraging overall summary
6. One practical dietary tip

Remember: Be friendly, positive, and make healthy eating feel accessible and enjoyable!"""
    
    def parse_response(self, response: str, state: Dict[str, Any]) -> AgentOutput:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # If no JSON, create structured data from text
                data = {
                    "welcome_message": "Here are your personalized food recommendations!",
                    "recommendations_explained": [],
                    "overall_summary": response,
                    "dietary_tip": "",
                    "encouragement": "Enjoy your healthy eating journey!"
                }
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                analysis=data.get('overall_summary', response),
                data=data,
                confidence=0.9
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                success=True,
                analysis=response,
                data={"raw_explanation": response},
                confidence=0.7,
                error=f"JSON parsing failed: {str(e)}"
            )
