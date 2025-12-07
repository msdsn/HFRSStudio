"""
Critic Agent - Evaluates and filters recommendations based on all agent analyses.
"""

from typing import Dict, Any, List
import json
import re

from .base import BaseAgent, AgentOutput


class CriticAgent(BaseAgent):
    """
    Critic Agent evaluates outputs from other agents and makes final filtering decisions.
    
    Responsibilities:
    - Review all agent analyses
    - Identify inconsistencies or conflicts
    - Make final filtering decisions
    - Rank foods based on combined scores
    - Ensure recommendation quality
    """
    
    def __init__(self):
        super().__init__(name="Critic")
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert AI critic responsible for evaluating and quality-controlling food recommendations.

Your role is to:
1. Review analyses from Nutritionist, Health Advisor, and Personalizer agents
2. Identify any inconsistencies or conflicts between analyses
3. Make final decisions on which foods to include/exclude
4. Create a combined ranking based on all factors
5. Ensure the final recommendations are high-quality and safe

EVALUATION CRITERIA:
- Health Safety: Foods with contraindications should be flagged or removed
- Nutritional Balance: The set should provide diverse nutrients
- Personalization Quality: Foods should match user preferences
- Overall Quality: Combined score from all factors

DECISION RULES:
1. EXCLUDE foods with critical health warnings
2. DEMOTE foods with moderate health concerns
3. BOOST foods that excel in multiple areas
4. ENSURE diversity in the final set

OUTPUT FORMAT:
Respond with a JSON object containing:
{
    "evaluation_summary": "overall evaluation of the recommendation set",
    "inconsistencies_found": ["list of any inconsistencies between agent analyses"],
    "foods_final_evaluation": [
        {
            "food_id": "string",
            "food_name": "string",
            "final_score": 0.0-1.0,
            "decision": "include/exclude/caution",
            "combined_strengths": ["list of strengths from all analyses"],
            "combined_concerns": ["list of concerns from all analyses"],
            "final_rank": 1-N
        }
    ],
    "excluded_foods": ["list of food IDs excluded with reasons"],
    "quality_score": 0.0-1.0,
    "improvement_suggestions": ["suggestions for improving recommendations"]
}"""
    
    def build_prompt(self, state: Dict[str, Any]) -> str:
        foods = state.get('model_predictions', [])
        nutritionist = state.get('nutritionist_analysis', {})
        health_advisor = state.get('health_analysis', {})
        personalizer = state.get('personalizer_analysis', {})
        
        # Format agent analyses
        nutritionist_summary = json.dumps(nutritionist.get('data', {}), indent=2)[:1500]
        health_summary = json.dumps(health_advisor.get('data', {}), indent=2)[:1500]
        personalizer_summary = json.dumps(personalizer.get('data', {}), indent=2)[:1500]
        
        # Format food list
        food_list = []
        for i, food in enumerate(foods[:10], 1):
            food_info = f"{i}. {food.get('food_name', 'Unknown')} (ID: {food.get('food_id', 'N/A')})"
            food_list.append(food_info)
        
        return f"""Please evaluate and critique the following food recommendations based on all agent analyses.

RECOMMENDED FOODS:
{chr(10).join(food_list)}

NUTRITIONIST ANALYSIS:
{nutritionist_summary}

HEALTH ADVISOR ANALYSIS:
{health_summary}

PERSONALIZER ANALYSIS:
{personalizer_summary}

Please:
1. Identify any inconsistencies between the agent analyses
2. Evaluate each food based on combined factors
3. Make final include/exclude decisions
4. Rank the foods from best to worst
5. Provide an overall quality score for the recommendation set
6. Flag any critical concerns"""
    
    def parse_response(self, response: str, state: Dict[str, Any]) -> AgentOutput:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {
                    "evaluation_summary": response,
                    "inconsistencies_found": [],
                    "foods_final_evaluation": [],
                    "excluded_foods": [],
                    "quality_score": 0.7,
                    "improvement_suggestions": []
                }
            
            confidence = data.get('quality_score', 0.7)
            
            return AgentOutput(
                agent_name=self.name,
                success=True,
                analysis=data.get('evaluation_summary', response),
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
