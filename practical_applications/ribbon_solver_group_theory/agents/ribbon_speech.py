"""
Ribbon Speech Translator Agent
==============================

Translates between mathematical formulas and Ribbon Speech - a natural language
representation grounded in the 6 mathematical anchors.

Ribbon Speech describes WHAT a formula DOES, not just what it IS.

Anchors:
- zero: Identity, origin, nothing
- sierpinski: Pattern, recursion, self-similarity
- phi: Growth, harmony, proportion
- e_inv: Decay, change, transformation
- cantor: Discrete, boundary, ratio
- sqrt2_inv: Bridge, connection, inverse

Example:
- "φ * φ" → "growth multiplied by growth, yielding greater harmony"
- "sin²(x) + cos²(x)" → "pattern squared combined with pattern squared, unity preserved"
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult


@dataclass
class RibbonSpeech:
    """Ribbon Speech representation of a formula."""
    formula: str
    speech: str
    anchor_weights: Dict[str, float]
    dominant_anchor: str
    concepts: List[str]


class RibbonSpeechTranslator(BaseAgent):
    """
    Translates between formulas and Ribbon Speech.
    
    Ribbon Speech is a semantic language that describes mathematical
    expressions in terms of their conceptual meaning, grounded in
    the 6 mathematical anchors.
    """
    
    # Anchor meanings
    ANCHOR_MEANINGS = {
        'zero': ('identity', 'origin', 'nothing', 'null', 'start'),
        'sierpinski': ('pattern', 'recursion', 'fractal', 'self-similar', 'cycle'),
        'phi': ('growth', 'harmony', 'proportion', 'golden', 'beauty'),
        'e_inv': ('decay', 'change', 'transformation', 'rate', 'flow'),
        'cantor': ('discrete', 'boundary', 'ratio', 'limit', 'step'),
        'sqrt2_inv': ('bridge', 'connection', 'inverse', 'link', 'balance'),
    }
    
    # Symbol to anchor mapping
    SYMBOL_TO_ANCHOR = {
        # Constants
        'phi': 'phi', 'φ': 'phi', 'golden': 'phi',
        'pi': 'sierpinski', 'π': 'sierpinski',
        'e': 'e_inv',
        '0': 'zero', '1': 'zero',
        '2': 'sqrt2_inv',
        'i': 'sqrt2_inv',  # imaginary unit bridges real/complex
        
        # Operations
        '+': 'phi',  # combining = growth
        '-': 'cantor',  # separating = boundary
        '*': 'phi',  # multiplication = growth
        '/': 'cantor',  # division = ratio
        '**': 'sierpinski',  # power = pattern
        '=': 'sqrt2_inv',  # equality = connection
        
        # Functions
        'sin': 'sierpinski', 'cos': 'sierpinski', 'tan': 'sierpinski',
        'exp': 'e_inv', 'log': 'e_inv', 'ln': 'e_inv',
        'sqrt': 'cantor',
        'arctan': 'sqrt2_inv', 'arcsin': 'sqrt2_inv', 'arccos': 'sqrt2_inv',
    }
    
    # Anchor to speech fragments
    ANCHOR_SPEECH = {
        'zero': ['origin', 'identity', 'nothing', 'the void', 'emptiness'],
        'sierpinski': ['pattern', 'cycle', 'recursion', 'self-similar form', 'fractal structure'],
        'phi': ['growth', 'harmony', 'golden proportion', 'expansion', 'flourishing'],
        'e_inv': ['change', 'decay', 'transformation', 'flow', 'rate of becoming'],
        'cantor': ['boundary', 'discrete step', 'ratio', 'limit', 'threshold'],
        'sqrt2_inv': ['bridge', 'connection', 'inverse relation', 'balance', 'link'],
    }
    
    # Operation speech
    OPERATION_SPEECH = {
        '+': 'combined with',
        '-': 'separated from',
        '*': 'multiplied by',
        '/': 'divided by',
        '**': 'raised to',
        '=': 'equals',
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _default_system_prompt(self) -> str:
        return """You are a Ribbon Speech translator. Convert mathematical formulas 
into poetic, meaningful descriptions using these anchor concepts:

- zero: identity, origin, nothing
- sierpinski: pattern, recursion, cycles
- phi: growth, harmony, proportion
- e_inv: decay, change, transformation
- cantor: discrete, boundary, ratio
- sqrt2_inv: bridge, connection, inverse

Example:
Formula: φ * φ
Ribbon Speech: "Growth multiplied by growth yields greater harmony"

Formula: sin²(x) + cos²(x)
Ribbon Speech: "Pattern squared combined with pattern squared preserves unity"

Respond with ONLY the Ribbon Speech, no explanation."""
    
    def process(self, formula: str) -> AgentResult:
        """
        Translate a formula to Ribbon Speech.
        
        Args:
            formula: Mathematical formula
            
        Returns:
            AgentResult with Ribbon Speech translation
        """
        self._start_timer()
        self._log(f"Translating: {formula}")
        
        # Analyze anchor weights
        weights = self._compute_anchor_weights(formula)
        dominant = max(weights, key=weights.get)
        
        # Generate speech
        speech = self._generate_speech(formula, weights, dominant)
        
        # Extract concepts
        concepts = self._extract_concepts(formula, weights)
        
        ribbon = RibbonSpeech(
            formula=formula,
            speech=speech,
            anchor_weights=weights,
            dominant_anchor=dominant,
            concepts=concepts
        )
        
        self._log(f"Speech: {speech}")
        
        return self._make_result(
            success=True,
            data={
                'formula': formula,
                'speech': speech,
                'anchor_weights': weights,
                'dominant_anchor': dominant,
                'concepts': concepts,
            }
        )
    
    def translate_to_formula(self, speech: str) -> AgentResult:
        """
        Translate Ribbon Speech back to a formula.
        
        Args:
            speech: Ribbon Speech description
            
        Returns:
            AgentResult with formula
        """
        self._start_timer()
        self._log(f"Reverse translating: {speech[:50]}...")
        
        # Use LLM for this complex task
        if self.use_llm:
            prompt = f"""Convert this Ribbon Speech back to a mathematical formula:
"{speech}"

Respond with ONLY the formula."""
            
            formula = self._generate(prompt, max_tokens=50, deterministic=True)
            formula = formula.strip()
        else:
            # Fallback: pattern matching
            formula = self._reverse_translate_patterns(speech)
        
        return self._make_result(
            success=bool(formula),
            data={
                'speech': speech,
                'formula': formula,
            }
        )
    
    def _compute_anchor_weights(self, formula: str) -> Dict[str, float]:
        """Compute anchor weights for a formula."""
        weights = {anchor: 0.0 for anchor in self.ANCHOR_MEANINGS}
        
        formula_lower = formula.lower()
        
        # Count symbol contributions
        for symbol, anchor in self.SYMBOL_TO_ANCHOR.items():
            count = formula_lower.count(symbol.lower())
            if count > 0:
                weights[anchor] += count
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Default to balanced
            weights = {k: 1/6 for k in weights}
        
        return weights
    
    def _generate_speech(self, formula: str, weights: Dict[str, float], 
                         dominant: str) -> str:
        """Generate Ribbon Speech for a formula."""
        
        # Try LLM for natural speech
        if self.use_llm:
            llm_speech = self._generate(
                prompt=f"Convert to Ribbon Speech: {formula}",
                max_tokens=100,
                deterministic=True
            )
            if llm_speech and len(llm_speech) > 10:
                return llm_speech.strip().strip('"')
        
        # Fallback: rule-based generation
        return self._rule_based_speech(formula, weights, dominant)
    
    def _rule_based_speech(self, formula: str, weights: Dict[str, float],
                           dominant: str) -> str:
        """Generate speech using rules."""
        parts = []
        
        # Parse formula into components
        # This is simplified - full implementation would use proper parsing
        
        # Handle common patterns
        if 'phi' in formula.lower() or 'φ' in formula:
            if '**2' in formula or '²' in formula:
                parts.append("golden growth squared")
            elif '*' in formula:
                parts.append("growth multiplied by growth")
            else:
                parts.append("golden proportion")
        
        if 'sin' in formula.lower() and 'cos' in formula.lower():
            if '+' in formula:
                parts.append("patterns combined in unity")
            else:
                parts.append("cyclic patterns intertwined")
        
        if 'exp' in formula.lower():
            parts.append("exponential transformation")
        
        if 'log' in formula.lower():
            parts.append("decay measure")
        
        if not parts:
            # Generic based on dominant anchor
            anchor_desc = self.ANCHOR_SPEECH[dominant][0]
            parts.append(f"expression of {anchor_desc}")
        
        # Add dominant anchor flavor
        if dominant == 'phi':
            suffix = ", yielding harmony"
        elif dominant == 'sierpinski':
            suffix = ", in recursive pattern"
        elif dominant == 'e_inv':
            suffix = ", through transformation"
        elif dominant == 'cantor':
            suffix = ", at the boundary"
        elif dominant == 'sqrt2_inv':
            suffix = ", bridging realms"
        else:
            suffix = ""
        
        return ' '.join(parts) + suffix
    
    def _extract_concepts(self, formula: str, weights: Dict[str, float]) -> List[str]:
        """Extract key concepts from formula."""
        concepts = []
        
        # Add concepts from significant anchors
        for anchor, weight in weights.items():
            if weight > 0.1:  # Threshold for significance
                concepts.extend(self.ANCHOR_MEANINGS[anchor][:2])
        
        return list(set(concepts))
    
    def _reverse_translate_patterns(self, speech: str) -> str:
        """Pattern-based reverse translation."""
        speech_lower = speech.lower()
        
        # Look for anchor keywords
        if 'growth' in speech_lower and 'squared' in speech_lower:
            return 'phi**2'
        if 'growth' in speech_lower and 'multiplied' in speech_lower:
            return 'phi * phi'
        if 'pattern' in speech_lower and 'unity' in speech_lower:
            return 'sin(x)**2 + cos(x)**2'
        if 'golden' in speech_lower:
            return 'phi'
        
        return ""
    
    def compare_speech(self, speech1: str, speech2: str) -> float:
        """
        Compare two Ribbon Speech descriptions for similarity.
        
        Returns similarity score 0-1.
        """
        # Extract concepts from both
        concepts1 = set()
        concepts2 = set()
        
        for anchor, meanings in self.ANCHOR_MEANINGS.items():
            for meaning in meanings:
                if meaning in speech1.lower():
                    concepts1.add(anchor)
                if meaning in speech2.lower():
                    concepts2.add(anchor)
        
        if not concepts1 or not concepts2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(concepts1 & concepts2)
        union = len(concepts1 | concepts2)
        
        return intersection / union if union > 0 else 0.0
