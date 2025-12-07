"""
Base Agent class for all AI agents in the recommendation workflow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

from utils.llm import get_llm_provider, LLMProvider


class AgentOutput(BaseModel):
    """Standard output format for all agents."""
    agent_name: str
    success: bool
    analysis: str
    data: Dict[str, Any] = {}
    confidence: float = 0.0
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Base class for all AI agents in the workflow.
    Each agent has a specific role and specialized prompt.
    """
    
    def __init__(self, name: str, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the agent.
        
        Args:
            name: Agent name for identification
            llm_provider: Optional custom LLM provider
        """
        self.name = name
        self.llm = llm_provider or get_llm_provider()
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt that defines the agent's role and behavior."""
        pass
    
    @abstractmethod
    def build_prompt(self, state: Dict[str, Any]) -> str:
        """
        Build the user prompt from the current state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str, state: Dict[str, Any]) -> AgentOutput:
        """
        Parse LLM response into structured output.
        
        Args:
            response: Raw LLM response
            state: Current workflow state
            
        Returns:
            Structured AgentOutput
        """
        pass
    
    async def run(self, state: Dict[str, Any]) -> AgentOutput:
        """
        Execute the agent's task.
        
        Args:
            state: Current workflow state
            
        Returns:
            AgentOutput with results
        """
        try:
            # Build prompt from state
            prompt = self.build_prompt(state)
            
            # Generate response
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt
            )
            
            # Parse and return structured output
            return self.parse_response(response, state)
            
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                success=False,
                analysis="",
                error=str(e)
            )
    
    async def run_stream(self, state: Dict[str, Any]):
        """
        Execute agent with streaming response.
        
        Args:
            state: Current workflow state
            
        Yields:
            Response chunks
        """
        try:
            prompt = self.build_prompt(state)
            
            async for chunk in self.llm.generate_stream(
                prompt=prompt,
                system_prompt=self.system_prompt
            ):
                yield chunk
                
        except Exception as e:
            yield f"[Error: {str(e)}]"
