"""
Base Processor
==============

Abstract base class for all processors.

All processors share:
1. Access to the TruthGroup (core structure)
2. Common result types
3. Logging and reporting
4. Numerical verification utilities
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import TruthGroup, GroupElement, AnchorVector


@dataclass
class ProcessorResult:
    """Base result class for all processors."""
    processor: str
    success: bool
    timestamp: str
    elapsed_time: float
    findings: List[Dict]
    summary: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.to_json())


class BaseProcessor(ABC):
    """
    Abstract base class for all processors.
    
    Processors operate on the TruthGroup to perform specific tasks.
    """
    
    def __init__(self, group: TruthGroup = None, verbose: bool = True):
        """
        Initialize processor with a TruthGroup.
        
        Args:
            group: The TruthGroup to operate on (created if None)
            verbose: Whether to print progress
        """
        self.group = group if group is not None else TruthGroup()
        self.verbose = verbose
        self._start_time = None
    
    def _log(self, message: str):
        """Log a message if verbose."""
        if self.verbose:
            print(message)
    
    def _start_timer(self):
        """Start timing an operation."""
        self._start_time = time.time()
    
    def _elapsed(self) -> float:
        """Get elapsed time since timer started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def _timestamp(self) -> str:
        """Get current timestamp."""
        return time.strftime("%Y-%m-%d %H:%M:%S")
    
    def _verify_numerically(self, expr1: str, expr2: str, 
                            test_points: List[float] = None) -> bool:
        """
        Verify two expressions are numerically equal.
        
        Args:
            expr1: First expression
            expr2: Second expression
            test_points: Points to test at (default: [0.1, 0.5, 1.0, 2.0])
            
        Returns:
            True if expressions match at all test points
        """
        if test_points is None:
            test_points = [0.1, 0.5, 1.0, 2.0]
        
        context = {
            'x': 0, 'np': np, 'pi': np.pi, 'e': np.e,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'arctan': np.arctan, 'arcsin': np.arcsin, 'arccos': np.arccos,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'sec': lambda x: 1/np.cos(x), 'csc': lambda x: 1/np.sin(x), 
            'cot': lambda x: 1/np.tan(x),
            'phi': (1 + np.sqrt(5)) / 2,
        }
        
        try:
            for x in test_points:
                context['x'] = x
                v1 = eval(self._normalize_expr(expr1), {"__builtins__": {}}, context)
                v2 = eval(self._normalize_expr(expr2), {"__builtins__": {}}, context)
                if abs(v1 - v2) > 1e-10:
                    return False
            return True
        except:
            return False
    
    def _normalize_expr(self, expr: str) -> str:
        """Normalize expression for eval."""
        result = expr
        # Handle multiplication symbols first
        result = result.replace('·', '*')
        result = result.replace('×', '*')
        
        # Handle function notation like sin²(x) -> (sin(x))**2
        # Must do this BEFORE replacing bare ²
        for func in ['sinh', 'cosh', 'tanh', 'sin', 'cos', 'tan', 'sec', 'csc', 'cot']:
            pattern = func + '²('
            while pattern in result:
                start = result.find(pattern)
                # Find matching closing paren
                depth = 1
                i = start + len(pattern)
                while i < len(result) and depth > 0:
                    if result[i] == '(':
                        depth += 1
                    elif result[i] == ')':
                        depth -= 1
                    i += 1
                # Extract the argument
                arg = result[start + len(pattern):i-1]
                # Replace with (func(arg))**2
                old = result[start:i]
                new = f'({func}({arg}))**2'
                result = result.replace(old, new, 1)
        
        # Handle remaining superscript 2 (e.g., x²)
        result = result.replace('²', '**2')
        
        return result
    
    @abstractmethod
    def process(self, *args, **kwargs) -> ProcessorResult:
        """
        Main processing method - must be implemented by subclasses.
        """
        pass
