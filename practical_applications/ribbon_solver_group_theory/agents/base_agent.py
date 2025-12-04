"""
Base Agent - Foundation for AI-Driven Truth Space Operations
=============================================================

Provides the abstract base class for all agents with:
- Ollama integration with 12D clock modulation
- Deterministic generation for reproducibility
- Structured result types
- Fallback to symbolic processing when LLM unavailable
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import json
import time
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import clock oracle
try:
    from optimized_clock_oracle import OptimizedClockOracle
    CLOCK_AVAILABLE = True
except ImportError:
    CLOCK_AVAILABLE = False


@dataclass
class AgentResult:
    """Base result class for all agents."""
    agent: str
    success: bool
    timestamp: str
    elapsed_ms: float
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class BaseAgent(ABC):
    """
    Abstract base class for all truth space agents.
    
    Provides:
    - Ollama integration with clock modulation
    - Deterministic generation mode
    - Timing and logging
    - Fallback mechanisms
    """
    
    # Default model preferences (in order of preference)
    DEFAULT_MODELS = ['llama3.2', 'llama3', 'mistral', 'phi3', 'gemma2', 'qwen2']
    
    def __init__(
        self,
        model: str = "llama3.2",
        clock_position: int = 1000,
        verbose: bool = True,
        use_llm: bool = True
    ):
        """
        Initialize the agent.
        
        Args:
            model: Ollama model to use
            clock_position: Starting clock position for determinism
            verbose: Whether to print progress
            use_llm: Whether to use LLM (False = symbolic only)
        """
        self.model = model
        self.clock_position = clock_position
        self.verbose = verbose
        self.use_llm = use_llm and OLLAMA_AVAILABLE
        
        self._start_time = None
        
        # Initialize clock oracle if available
        if CLOCK_AVAILABLE:
            self.clock = OptimizedClockOracle(max_n=10000, lazy=False)
        else:
            self.clock = None
        
        # Verify model if using LLM
        if self.use_llm:
            self._verify_model()
    
    def _verify_model(self):
        """Check if model is available, find fallback if not."""
        if not OLLAMA_AVAILABLE:
            return
        
        try:
            result = ollama.list()
            models_list = result.models if hasattr(result, 'models') else result.get('models', [])
            
            available = []
            for m in models_list:
                if hasattr(m, 'model'):
                    available.append(m.model.split(':')[0])
                else:
                    available.append(m.get('model', m.get('name', '')).split(':')[0])
            
            if self.model not in available and self.model.split(':')[0] not in available:
                # Try fallbacks
                for fallback in self.DEFAULT_MODELS:
                    if fallback in available:
                        self._log(f"Model '{self.model}' not found, using: {fallback}")
                        self.model = fallback
                        return
                
                if available:
                    self._log(f"Using first available model: {available[0]}")
                    self.model = available[0]
                else:
                    self._log("No models available, disabling LLM")
                    self.use_llm = False
                    
        except Exception as e:
            self._log(f"Could not verify model: {e}")
            self.use_llm = False
    
    def _log(self, message: str):
        """Log a message if verbose."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")
    
    def _start_timer(self):
        """Start timing an operation."""
        self._start_time = time.perf_counter()
    
    def _elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000
    
    def _timestamp(self) -> str:
        """Get current timestamp."""
        return time.strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_clock_seed(self) -> int:
        """Get deterministic seed from clock position."""
        if self.clock:
            return self.clock.get_seed(self.clock_position)
        else:
            # Fallback: simple deterministic seed
            return (self.clock_position * 1000) % (2**31)
    
    def _get_clock_temperature(self) -> float:
        """Get clock-modulated temperature (for non-deterministic mode)."""
        if self.clock:
            return self.clock.get_temperature(self.clock_position)
        else:
            return 0.7
    
    def _generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 200,
        deterministic: bool = True
    ) -> str:
        """
        Generate text using Ollama with clock modulation.
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            deterministic: If True, use temperature=0 for reproducibility
            
        Returns:
            Generated text
        """
        if not self.use_llm:
            return ""
        
        options = {
            'num_predict': max_tokens,
            'seed': self._get_clock_seed(),
        }
        
        if deterministic:
            options['temperature'] = 0.0
            options['top_k'] = 1
        else:
            options['temperature'] = self._get_clock_temperature()
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                system=system_prompt or self._default_system_prompt(),
                options=options,
                stream=False
            )
            return response['response']
        except Exception as e:
            self._log(f"Generation error: {e}")
            return ""
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for the agent."""
        return "You are a precise mathematical assistant. Respond concisely and accurately."
    
    def advance_clock(self, steps: int = 1):
        """Advance the clock position."""
        self.clock_position += steps
    
    def _make_result(
        self,
        success: bool,
        data: Dict = None,
        error: str = None
    ) -> AgentResult:
        """Create a standardized result."""
        return AgentResult(
            agent=self.__class__.__name__,
            success=success,
            timestamp=self._timestamp(),
            elapsed_ms=self._elapsed_ms(),
            data=data or {},
            error=error
        )
    
    @abstractmethod
    def process(self, input_data: Any) -> AgentResult:
        """
        Main processing method - must be implemented by subclasses.
        
        Args:
            input_data: Input to process (type depends on agent)
            
        Returns:
            AgentResult with processing results
        """
        pass
