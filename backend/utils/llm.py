"""
Unified LLM Provider with Gemini primary and OpenAI fallback.
Uses LangChain for consistent interface across providers.
"""

from typing import Optional, List, Dict, Any, AsyncGenerator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

from config import get_settings

settings = get_settings()


class LLMProvider:
    """
    Unified LLM interface with automatic fallback.
    Primary: Gemini (Google)
    Fallback: OpenAI GPT-4
    """
    
    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        primary: str = "gemini",
        fallback: str = "openai"
    ):
        """
        Initialize LLM provider.
        
        Args:
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            primary: Primary LLM (gemini or openai)
            fallback: Fallback LLM (gemini or openai)
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.primary_name = primary
        self.fallback_name = fallback
        
        # Initialize primary LLM (Gemini)
        self.primary_llm = self._create_llm(primary)
        
        # Initialize fallback LLM (OpenAI)
        self.fallback_llm = self._create_llm(fallback)
        
        # Track which LLM was used
        self.last_used = None
    
    def _create_llm(self, provider: str):
        """Create LLM instance based on provider name."""
        if provider == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.gemini_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                convert_system_message_to_human=True
            )
        elif provider == "openai":
            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=settings.openai_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _invoke_with_retry(self, llm, messages: List) -> str:
        """Invoke LLM with retry logic."""
        response = await llm.ainvoke(messages)
        return response.content
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate response using primary LLM with fallback.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            context: Optional context dictionary for prompt formatting
            
        Returns:
            Generated response string
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Format prompt with context if provided
        if context:
            prompt = prompt.format(**context)
        
        messages.append(HumanMessage(content=prompt))
        
        # Try primary LLM first
        try:
            self.last_used = self.primary_name
            return await self._invoke_with_retry(self.primary_llm, messages)
        except Exception as e:
            print(f"Primary LLM ({self.primary_name}) failed: {e}")
            
            # Fallback to secondary LLM
            try:
                self.last_used = self.fallback_name
                return await self._invoke_with_retry(self.fallback_llm, messages)
            except Exception as e2:
                print(f"Fallback LLM ({self.fallback_name}) also failed: {e2}")
                raise Exception(f"All LLMs failed. Primary: {e}, Fallback: {e2}")
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response using primary LLM with fallback.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            context: Optional context dictionary
            
        Yields:
            Response chunks as they're generated
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        if context:
            prompt = prompt.format(**context)
        
        messages.append(HumanMessage(content=prompt))
        
        # Try primary LLM first
        try:
            self.last_used = self.primary_name
            async for chunk in self.primary_llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"Primary LLM streaming failed: {e}, trying fallback...")
            
            # Fallback
            try:
                self.last_used = self.fallback_name
                async for chunk in self.fallback_llm.astream(messages):
                    if chunk.content:
                        yield chunk.content
            except Exception as e2:
                yield f"[Error: All LLMs failed]"
    
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Synchronous version of generate."""
        return asyncio.run(self.generate(prompt, system_prompt, context))


# Singleton instance
_llm_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Get or create the singleton LLM provider instance."""
    global _llm_provider
    
    if _llm_provider is None:
        _llm_provider = LLMProvider(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            primary=settings.primary_llm,
            fallback=settings.fallback_llm
        )
    
    return _llm_provider


def reset_llm_provider():
    """Reset the LLM provider instance."""
    global _llm_provider
    _llm_provider = None
