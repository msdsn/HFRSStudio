"""
Health Advisor Agent - Evaluates food-health compatibility based on user's health profile.
"""

from typing import Dict, Any, List
import json
import re

from .base import BaseAgent, AgentOutput


class HealthAdvisorAgent(BaseAgent):
    """
    Health Advisor Agent evaluates how well foods match user's health requirements.
    
    Responsibilities:
    - Analyze user's health conditions and requirements
    - Evaluate food-health compatibility
    - Identify contraindications
    - Provide personalized health advice
    """
    
    def __init__(self):
        super().__init__(name="HealthAdvisor")
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert health advisor AI assistant specializing in personalized nutrition.

Your role is to:
1. Understand user's health profile (conditions, dietary needs, restrictions)
2. Evaluate how well each recommended food aligns with their health requirements
3. Identify any contraindications or foods that may be harmful
4. Provide personalized health-aware food guidance
5. Consider interactions between multiple health conditions

HEALTH CONDITION MAPPINGS:
- High blood pressure → Needs low sodium foods
- Diabetes → Needs low sugar, low carb foods
- High cholesterol → Needs low cholesterol, low saturated fat foods
- Kidney disease → Needs low phosphorus, low potassium, low sodium foods
- Heart disease → Needs low sodium, low saturated fat, low cholesterol foods
- Overweight/Obesity → Needs low calorie foods
- Underweight → Needs high calorie, high protein foods
- Anemia → Needs high iron, high vitamin B12, high folic acid foods
- Pregnancy → Needs high folic acid, high iron, high calcium foods

OUTPUT FORMAT:
Respond with a JSON object containing:
{
    "user_health_summary": "summary of user's health profile and needs",
    "foods_evaluation": [
        {
            "food_id": "string",
            "food_name": "string",
            "health_compatibility_score": 0.0-1.0,
            "suitable": true/false,
            "matching_health_needs": ["list of matching health requirements"],
            "contraindications": ["list of health concerns"],
            "personalized_advice": "specific advice for this user"
        }
    ],
    "overall_compatibility": 0.0-1.0,
    "critical_warnings": ["any critical health warnings"],
    "general_advice": "overall health advice for the user"
}"""
    
    def build_prompt(self, state: Dict[str, Any]) -> str:
        user_profile = state.get('user_profile', {})
        foods = state.get('model_predictions', [])
        nutritionist_analysis = state.get('nutritionist_analysis', {})
        
        # Extract health information
        health_tags = user_profile.get('health_tags', {})
        age = user_profile.get('age', 'Unknown')
        gender = user_profile.get('gender', 'Unknown')
        dietary_restrictions = user_profile.get('dietary_restrictions', [])
        allergies = user_profile.get('allergies', [])
        
        # Format health needs
        health_needs = [tag.replace('user_', '').replace('_', ' ') 
                       for tag, value in health_tags.items() if value]
        
        # Format food list
        food_list = []
        for food in foods[:10]:
            food_tags = [tag.replace('_', ' ') 
                        for tag, value in food.get('health_tags', {}).items() if value]
            food_info = f"""
Food: {food.get('food_name', 'Unknown')} (ID: {food.get('food_id', 'N/A')})
Category: {food.get('category', 'Unknown')}
Health Tags: {', '.join(food_tags) if food_tags else 'None'}
"""
            food_list.append(food_info)
        
        return f"""Please evaluate the health compatibility of these recommended foods for the user.

USER HEALTH PROFILE:
- Age: {age}
- Gender: {gender}
- Health Needs: {', '.join(health_needs) if health_needs else 'No specific needs'}
- Dietary Restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}
- Allergies: {', '.join(allergies) if allergies else 'None'}

RECOMMENDED FOODS:
{''.join(food_list)}

NUTRITIONIST ANALYSIS (for reference):
{json.dumps(nutritionist_analysis.get('data', {}), indent=2)[:1000]}

Please evaluate:
1. How well each food matches the user's health requirements
2. Any foods that should be avoided due to contraindications
3. Personalized advice for each food
4. Overall health compatibility score for the recommendation set"""
    
    def parse_response(self, response: str, state: Dict[str, Any]) -> AgentOutput:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {
                    "user_health_summary": "",
                    "foods_evaluation": [],
                    "overall_compatibility": 0.7,
                    "critical_warnings": [],
                    "general_advice": response
                }
            
            # Calculate confidence based on compatibility score
            confidence = data.get('overall_compatibility', 0.7)
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                analysis=data.get('general_advice', response),
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
