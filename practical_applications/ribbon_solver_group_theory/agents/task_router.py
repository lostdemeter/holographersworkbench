"""
Task Router Agent
=================

Classifies user intent and routes to appropriate handlers.

Task Types:
- EXPLORE: Navigate truth space to find related truths
- SIMPLIFY: Find simpler form of an expression
- DISCOVER: Find new identities or formulas
- VERIFY: Check if an identity is valid
- TRANSLATE: Convert between representations
- ANALYZE: Analyze properties of an expression
"""

import re
from enum import Enum, auto
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult


class TaskType(Enum):
    """Types of truth space tasks."""
    EXPLORE = auto()      # Navigate truth space
    SIMPLIFY = auto()     # Find simpler form
    DISCOVER = auto()     # Find new identities
    VERIFY = auto()       # Check validity
    TRANSLATE = auto()    # Convert representations
    ANALYZE = auto()      # Analyze properties
    UNKNOWN = auto()      # Could not classify


@dataclass
class RoutingResult:
    """Result of task routing."""
    task_type: TaskType
    confidence: float
    extracted_formula: Optional[str]
    extracted_target: Optional[str]
    original_query: str


class TaskRouter(BaseAgent):
    """
    Routes user queries to appropriate handlers.
    
    Uses pattern matching first (fast, deterministic), falls back to
    LLM classification for ambiguous queries.
    """
    
    # Pattern-based classification (fast path)
    PATTERNS = {
        TaskType.EXPLORE: [
            r"find.*identit",
            r"explore",
            r"navigate",
            r"what.*related",
            r"discover.*near",
            r"orbit",
            r"nearby",
        ],
        TaskType.SIMPLIFY: [
            r"simplif",
            r"reduce",
            r"simpler",
            r"optimize",
            r"what.*equal",
            r"what.*is\s+\w+\s*[\*\+\-\/\^]",  # "what is x * y"
        ],
        TaskType.DISCOVER: [
            r"find.*formula",
            r"discover.*new",
            r"search.*for",
            r"mine.*identit",
            r"generate",
        ],
        TaskType.VERIFY: [
            r"verify",
            r"check",
            r"is.*true",
            r"does.*equal",
            r"prove",
            r"valid",
        ],
        TaskType.TRANSLATE: [
            r"translate",
            r"convert",
            r"express.*as",
            r"what.*mean",
            r"ribbon speech",
            r"in english",
        ],
        TaskType.ANALYZE: [
            r"analyze",
            r"what.*type",
            r"classify",
            r"properties",
            r"dominant",
            r"category",
        ],
    }
    
    # Formula extraction patterns
    FORMULA_PATTERNS = [
        r"(?:of|for|about)\s+[\"']?([^\"'\?]+)[\"']?",  # "of phi * phi"
        r"[\"']([^\"']+)[\"']",  # quoted formula
        r":\s*(.+?)(?:\?|$)",  # after colon
        r"(?:expression|formula|identity)\s+(.+?)(?:\?|$)",
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _default_system_prompt(self) -> str:
        return """You are a task classifier for a mathematical truth space system.
Given a user query, classify it into one of these categories:
- EXPLORE: User wants to navigate truth space or find related truths
- SIMPLIFY: User wants to simplify or reduce an expression
- DISCOVER: User wants to find new identities or formulas
- VERIFY: User wants to check if something is true
- TRANSLATE: User wants to convert between representations
- ANALYZE: User wants to understand properties of an expression

Respond with ONLY the category name (e.g., "EXPLORE")."""
    
    def process(self, query: str) -> AgentResult:
        """
        Route a user query to the appropriate task type.
        
        Args:
            query: Natural language query from user
            
        Returns:
            AgentResult with routing information
        """
        self._start_timer()
        self._log(f"Routing query: {query[:50]}...")
        
        # Try pattern-based classification first (fast)
        task_type, confidence = self._classify_by_pattern(query)
        
        # If low confidence, try LLM
        if confidence < 0.5 and self.use_llm:
            llm_type = self._classify_by_llm(query)
            if llm_type != TaskType.UNKNOWN:
                task_type = llm_type
                confidence = 0.7  # LLM classification confidence
        
        # Extract formula from query
        formula = self._extract_formula(query)
        target = self._extract_target(query)
        
        routing = RoutingResult(
            task_type=task_type,
            confidence=confidence,
            extracted_formula=formula,
            extracted_target=target,
            original_query=query
        )
        
        self._log(f"Routed to: {task_type.name} (confidence: {confidence:.2f})")
        
        return self._make_result(
            success=task_type != TaskType.UNKNOWN,
            data={
                'task_type': task_type.name,
                'confidence': confidence,
                'extracted_formula': formula,
                'extracted_target': target,
                'original_query': query,
            }
        )
    
    def _classify_by_pattern(self, query: str) -> Tuple[TaskType, float]:
        """Classify using regex patterns."""
        query_lower = query.lower()
        
        best_type = TaskType.UNKNOWN
        best_score = 0.0
        
        for task_type, patterns in self.PATTERNS.items():
            matches = sum(1 for p in patterns if re.search(p, query_lower))
            score = matches / len(patterns) if patterns else 0
            
            if score > best_score:
                best_score = score
                best_type = task_type
        
        # Boost confidence if multiple patterns match
        confidence = min(best_score * 2, 1.0) if best_score > 0 else 0.0
        
        return best_type, confidence
    
    def _classify_by_llm(self, query: str) -> TaskType:
        """Classify using LLM."""
        response = self._generate(
            prompt=f"Classify this query: {query}",
            max_tokens=20,
            deterministic=True
        )
        
        response_upper = response.upper().strip()
        
        for task_type in TaskType:
            if task_type.name in response_upper:
                return task_type
        
        return TaskType.UNKNOWN
    
    def _extract_formula(self, query: str) -> Optional[str]:
        """Extract mathematical formula from query."""
        for pattern in self.FORMULA_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                formula = match.group(1).strip()
                # Clean up
                formula = formula.rstrip('?.,!')
                if formula:
                    return formula
        
        # Try to find math-like content
        # Look for operators or Greek letters
        math_pattern = r'[\w\d]+\s*[\*\+\-\/\^\=]+\s*[\w\d]+'
        match = re.search(math_pattern, query)
        if match:
            return match.group(0)
        
        # Look for function calls
        func_pattern = r'(?:sin|cos|tan|exp|log|sqrt|phi|π)\s*[\(\*\+]'
        match = re.search(func_pattern, query, re.IGNORECASE)
        if match:
            # Extract surrounding context
            start = max(0, match.start() - 5)
            end = min(len(query), match.end() + 20)
            return query[start:end].strip()
        
        return None
    
    def _extract_target(self, query: str) -> Optional[str]:
        """Extract target (e.g., 'pi' in 'find formula for pi')."""
        target_patterns = [
            r"for\s+(\w+)",
            r"of\s+(\w+)\s*$",
            r"about\s+(\w+)",
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
