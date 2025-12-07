"""
Nutritionist Agent - Analyzes nutritional values of recommended foods.
"""

from typing import Dict, Any, List
import json
import re

from .base import BaseAgent, AgentOutput


class NutritionistAgent(BaseAgent):
    """
    Nutritionist Agent analyzes the nutritional profile of recommended foods.
    
    Responsibilities:
    - Evaluate macro/micro nutrient content
    - Classify foods according to WHO/FSA standards
    - Identify nutritional strengths and weaknesses
    - Provide nutritional summary for each food
    """
    
    def __init__(self):
        super().__init__(name="Nutritionist")
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert nutritionist AI assistant specializing in analyzing food nutritional values.

Your role is to:
1. Analyze the nutritional content of recommended foods
2. Evaluate macro-nutrients (calories, protein, carbs, fats) and micro-nutrients (vitamins, minerals)
3. Classify foods according to international health standards (WHO, FSA, EU Nutrition guidelines)
4. Identify nutritional strengths (e.g., "high in fiber", "low sodium") and weaknesses
5. Provide clear, concise nutritional summaries

IMPORTANT GUIDELINES:
- Be precise with nutritional assessments
- Use standard serving sizes when applicable
- Reference established dietary guidelines
- Consider both positive and negative nutritional aspects
- Output your analysis in a structured JSON format

OUTPUT FORMAT:
Respond with a JSON object containing:
{
    "foods_analysis": [
        {
            "food_id": "string",
            "food_name": "string",
            "nutritional_profile": "balanced/high-calorie/low-calorie/high-protein/etc",
            "strengths": ["list of nutritional strengths"],
            "weaknesses": ["list of nutritional weaknesses"],
            "key_nutrients": {"nutrient": "level (high/moderate/low)"},
            "health_classification": "WHO/FSA classification",
            "summary": "brief nutritional summary"
        }
    ],
    "overall_assessment": "overall nutritional assessment of the recommendation set",
    "recommendations": ["additional nutritional recommendations"]
}"""
    
    def build_prompt(self, state: Dict[str, Any]) -> str:
        foods = state.get('model_predictions', [])
        
        # Format food list for analysis
        food_list = []
        for food in foods[:10]:  # Analyze top 10 foods
            food_info = f"""
Food ID: {food.get('food_id', 'N/A')}
Food Name: {food.get('food_name', 'Unknown')}
Category: {food.get('category', 'Unknown')}
Nutrients: {json.dumps(food.get('nutrients', {}), indent=2)}
Health Tags: {json.dumps(food.get('health_tags', {}), indent=2)}
"""
            food_list.append(food_info)
        
        foods_text = "\n---\n".join(food_list)
        
        return f"""Please analyze the nutritional profile of the following recommended foods:

{foods_text}

Provide a comprehensive nutritional analysis for each food, including:
1. Overall nutritional profile classification
2. Key nutritional strengths
3. Potential nutritional weaknesses or concerns
4. Important nutrients and their levels
5. Brief health classification

Also provide an overall assessment of how well this set of recommendations covers different nutritional needs."""
    
    def parse_response(self, response: str, state: Dict[str, Any]) -> AgentOutput:
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # If no JSON found, create structured data from text
                data = {
                    "foods_analysis": [],
                    "overall_assessment": response,
                    "recommendations": []
                }
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                analysis=data.get('overall_assessment', response),
                data=data,
                confidence=0.85
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                success=True,
                analysis=response,
                data={"raw_analysis": response},
                confidence=0.7,
                error=f"JSON parsing failed: {str(e)}"
            )
