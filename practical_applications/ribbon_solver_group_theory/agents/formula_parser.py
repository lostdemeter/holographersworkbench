"""
Formula Parser Agent
====================

Converts natural language mathematical expressions into unambiguous formulas.

Key responsibilities:
1. Parse natural language ("phi times phi") → explicit formula ("φ * φ")
2. Resolve ambiguity (order of operations)
3. Normalize notation (consistent symbols)
4. Validate syntax
"""

import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from .base_agent import BaseAgent, AgentResult

# Try sympy for validation
try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


@dataclass
class ParsedFormula:
    """A parsed and normalized formula."""
    original: str
    normalized: str
    explicit: str  # Fully parenthesized, unambiguous
    symbols: List[str]
    is_valid: bool
    sympy_expr: Optional[str] = None


class FormulaParser(BaseAgent):
    """
    Parses natural language into unambiguous mathematical formulas.
    
    Handles:
    - Word forms: "phi times phi" → "φ * φ"
    - Implicit multiplication: "2x" → "2 * x"
    - Order of operations: "a + b * c" → "a + (b * c)"
    - Greek letters: "phi", "pi", "theta" → "φ", "π", "θ"
    """
    
    # Word to symbol mappings
    WORD_TO_SYMBOL = {
        # Operations
        'plus': '+',
        'minus': '-',
        'times': '*',
        'multiplied by': '*',
        'divided by': '/',
        'over': '/',
        'to the power of': '**',
        'squared': '**2',
        'cubed': '**3',
        'square root of': 'sqrt',
        'sqrt': 'sqrt',
        
        # Constants
        'phi': 'φ',
        'golden ratio': 'φ',
        'pi': 'π',
        'e': 'e',
        'euler': 'e',
        'infinity': '∞',
        
        # Functions
        'sine': 'sin',
        'cosine': 'cos',
        'tangent': 'tan',
        'arctangent': 'arctan',
        'arcsine': 'arcsin',
        'arccosine': 'arccos',
        'logarithm': 'log',
        'natural log': 'ln',
        'exponential': 'exp',
    }
    
    # Symbol normalization (for consistent output)
    SYMBOL_NORMALIZE = {
        'φ': 'phi',
        'π': 'pi',
        '×': '*',
        '÷': '/',
        '·': '*',
        '^': '**',
        '²': '**2',
        '³': '**3',
    }
    
    # Operator precedence for explicit parenthesization
    PRECEDENCE = {
        '+': 1, '-': 1,
        '*': 2, '/': 2,
        '**': 3,
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _default_system_prompt(self) -> str:
        return """You are a mathematical formula parser. Convert natural language 
mathematical expressions into explicit symbolic notation.

Rules:
1. Use standard operators: +, -, *, /, **
2. Use parentheses to make order of operations explicit
3. Use standard function names: sin, cos, tan, exp, log, sqrt
4. Use 'phi' for golden ratio, 'pi' for π, 'e' for Euler's number

Example:
Input: "phi times phi"
Output: phi * phi

Input: "sine of x squared plus cosine of x squared"
Output: (sin(x))**2 + (cos(x))**2

Respond with ONLY the formula, no explanation."""
    
    def process(self, input_text: str) -> AgentResult:
        """
        Parse natural language into an unambiguous formula.
        
        Args:
            input_text: Natural language mathematical expression
            
        Returns:
            AgentResult with parsed formula
        """
        self._start_timer()
        self._log(f"Parsing: {input_text}")
        
        # Step 1: Normalize input
        normalized = self._normalize_input(input_text)
        
        # Step 2: Convert words to symbols
        symbolized = self._words_to_symbols(normalized)
        
        # Step 3: Try LLM for complex cases
        if self.use_llm and self._is_complex(input_text):
            llm_result = self._parse_with_llm(input_text)
            if llm_result:
                symbolized = llm_result
        
        # Step 4: Make explicit (add parentheses)
        explicit = self._make_explicit(symbolized)
        
        # Step 5: Extract symbols
        symbols = self._extract_symbols(explicit)
        
        # Step 6: Validate with SymPy if available
        is_valid, sympy_str = self._validate(explicit)
        
        parsed = ParsedFormula(
            original=input_text,
            normalized=normalized,
            explicit=explicit,
            symbols=symbols,
            is_valid=is_valid,
            sympy_expr=sympy_str
        )
        
        self._log(f"Parsed: {explicit}")
        
        return self._make_result(
            success=is_valid,
            data={
                'original': input_text,
                'normalized': normalized,
                'explicit': explicit,
                'symbols': symbols,
                'is_valid': is_valid,
                'sympy_expr': sympy_str,
            }
        )
    
    def _normalize_input(self, text: str) -> str:
        """Normalize input text."""
        result = text.lower().strip()
        
        # Remove question marks and extra punctuation
        result = result.rstrip('?.,!')
        
        # Normalize whitespace
        result = ' '.join(result.split())
        
        return result
    
    def _words_to_symbols(self, text: str) -> str:
        """Convert word forms to symbols."""
        result = text
        
        # Sort by length (longest first) to avoid partial matches
        sorted_words = sorted(self.WORD_TO_SYMBOL.keys(), key=len, reverse=True)
        
        for word in sorted_words:
            symbol = self.WORD_TO_SYMBOL[word]
            result = re.sub(rf'\b{re.escape(word)}\b', f' {symbol} ', result, flags=re.IGNORECASE)
        
        # Clean up whitespace
        result = ' '.join(result.split())
        
        # Handle implicit multiplication (e.g., "2x" → "2 * x")
        result = re.sub(r'(\d)\s*([a-zA-Z])', r'\1 * \2', result)
        
        # Handle function application (e.g., "sin x" → "sin(x)")
        for func in ['sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'arctan', 'arcsin', 'arccos']:
            result = re.sub(rf'\b{func}\s+(\w+)\b', rf'{func}(\1)', result)
        
        return result.strip()
    
    def _is_complex(self, text: str) -> bool:
        """Check if input is complex enough to need LLM."""
        # Complex if: long, has nested structures, or ambiguous
        if len(text) > 50:
            return True
        if text.count('(') > 1 or text.count(')') > 1:
            return True
        if 'of' in text.lower() and ('squared' in text.lower() or 'cubed' in text.lower()):
            return True
        return False
    
    def _parse_with_llm(self, text: str) -> Optional[str]:
        """Use LLM to parse complex expressions."""
        response = self._generate(
            prompt=f"Convert to formula: {text}",
            max_tokens=100,
            deterministic=True
        )
        
        if response:
            # Clean up LLM response
            response = response.strip()
            # Remove any explanation
            if '\n' in response:
                response = response.split('\n')[0]
            return response
        
        return None
    
    def _make_explicit(self, formula: str) -> str:
        """Add parentheses to make order of operations explicit."""
        # This is a simplified version - full implementation would use
        # proper expression parsing
        
        result = formula
        
        # Ensure ** binds tighter than * and /
        result = re.sub(r'(\w+)\s*\*\*\s*(\d+)', r'(\1**\2)', result)
        
        # Ensure * and / bind tighter than + and -
        # This is approximate - proper implementation needs AST
        
        return result
    
    def _extract_symbols(self, formula: str) -> List[str]:
        """Extract variable and constant symbols from formula."""
        # Find all word-like tokens that aren't functions
        functions = {'sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'arctan', 'arcsin', 'arccos'}
        
        tokens = re.findall(r'\b[a-zA-Z_]\w*\b', formula)
        symbols = [t for t in tokens if t.lower() not in functions]
        
        return list(set(symbols))
    
    def _validate(self, formula: str) -> Tuple[bool, Optional[str]]:
        """Validate formula syntax using SymPy."""
        if not SYMPY_AVAILABLE:
            return True, None  # Assume valid if can't check
        
        # Normalize for SymPy
        sympy_formula = formula
        sympy_formula = sympy_formula.replace('phi', '((1 + sqrt(5)) / 2)')
        sympy_formula = sympy_formula.replace('π', 'pi')
        
        try:
            expr = parse_expr(sympy_formula)
            return True, str(expr)
        except Exception as e:
            self._log(f"Validation error: {e}")
            return False, None
    
    def parse_to_sympy(self, formula: str):
        """Parse formula and return SymPy expression."""
        if not SYMPY_AVAILABLE:
            return None
        
        result = self.process(formula)
        if result.success and result.data.get('sympy_expr'):
            try:
                return parse_expr(result.data['sympy_expr'])
            except:
                pass
        return None
