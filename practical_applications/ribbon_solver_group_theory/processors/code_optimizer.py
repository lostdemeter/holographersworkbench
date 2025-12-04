"""
Code Optimizer Processor
========================

Scans code for optimization opportunities using group-theoretic analysis.

This processor:
1. Extracts mathematical expressions from code
2. Maps them to group elements
3. Finds symmetries (simplifications)
4. Suggests optimizations

Ported from ribbon_solver2/tools/auto_optimizer.py with group theory integration.
"""

import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .base import BaseProcessor, ProcessorResult


@dataclass
class Optimization:
    """A single optimization recommendation."""
    original: str
    optimized: str
    category: str  # 'inverse', 'pythagorean', 'algebraic', 'exponential'
    speedup: float
    verified: bool
    location: Optional[str] = None
    description: str = ""
    group_info: Dict = None


@dataclass
class OptimizationResult(ProcessorResult):
    """Result from code optimization."""
    source: str
    total_expressions: int
    optimizations: List[Dict]
    estimated_speedup: float


class CodeOptimizer(BaseProcessor):
    """
    Code optimization using group-theoretic symmetry detection.
    
    Usage:
        optimizer = CodeOptimizer()
        result = optimizer.optimize_file("path/to/code.py")
        result = optimizer.optimize_expressions(["arctan(tan(x))", "sin²(x) + cos²(x)"])
    """
    
    # Known optimization patterns with group-theoretic categories
    PATTERNS = {
        # Inverse compositions (identity anchor)
        r'arctan\s*\(\s*tan\s*\(([^)]+)\)\s*\)': {
            'replacement': r'\1',
            'category': 'inverse',
            'description': 'Inverse composition: arctan(tan(x)) = x',
            'speedup': 100.0,
        },
        r'arcsin\s*\(\s*sin\s*\(([^)]+)\)\s*\)': {
            'replacement': r'\1',
            'category': 'inverse',
            'description': 'Inverse composition: arcsin(sin(x)) = x',
            'speedup': 100.0,
        },
        r'arccos\s*\(\s*cos\s*\(([^)]+)\)\s*\)': {
            'replacement': r'\1',
            'category': 'inverse',
            'description': 'Inverse composition: arccos(cos(x)) = x',
            'speedup': 100.0,
        },
        r'log\s*\(\s*exp\s*\(([^)]+)\)\s*\)': {
            'replacement': r'\1',
            'category': 'inverse',
            'description': 'Inverse composition: log(exp(x)) = x',
            'speedup': 100.0,
        },
        r'exp\s*\(\s*log\s*\(([^)]+)\)\s*\)': {
            'replacement': r'\1',
            'category': 'inverse',
            'description': 'Inverse composition: exp(log(x)) = x',
            'speedup': 100.0,
        },
        r'sqrt\s*\(([^)]+)\)\s*\*\*\s*2': {
            'replacement': r'\1',
            'category': 'inverse',
            'description': 'Inverse composition: sqrt(x)**2 = x',
            'speedup': 50.0,
        },
        
        # Pythagorean identities (unity anchor)
        r'sin\s*\(([^)]+)\)\s*\*\*\s*2\s*\+\s*cos\s*\(\1\)\s*\*\*\s*2': {
            'replacement': '1',
            'category': 'pythagorean',
            'description': 'Pythagorean identity: sin²(x) + cos²(x) = 1',
            'speedup': 200.0,
        },
        r'cos\s*\(([^)]+)\)\s*\*\*\s*2\s*\+\s*sin\s*\(\1\)\s*\*\*\s*2': {
            'replacement': '1',
            'category': 'pythagorean',
            'description': 'Pythagorean identity: cos²(x) + sin²(x) = 1',
            'speedup': 200.0,
        },
        r'1\s*-\s*sin\s*\(([^)]+)\)\s*\*\*\s*2': {
            'replacement': r'cos(\1)**2',
            'category': 'pythagorean',
            'description': 'Pythagorean identity: 1 - sin²(x) = cos²(x)',
            'speedup': 50.0,
        },
        r'1\s*-\s*cos\s*\(([^)]+)\)\s*\*\*\s*2': {
            'replacement': r'sin(\1)**2',
            'category': 'pythagorean',
            'description': 'Pythagorean identity: 1 - cos²(x) = sin²(x)',
            'speedup': 50.0,
        },
        
        # Exponential identities (growth anchor)
        r'exp\s*\(([^)]+)\)\s*\*\s*exp\s*\(-\1\)': {
            'replacement': '1',
            'category': 'exponential',
            'description': 'Exponential identity: exp(x)*exp(-x) = 1',
            'speedup': 100.0,
        },
        r'exp\s*\(([^)]+)\)\s*\*\s*exp\s*\(([^)]+)\)': {
            'replacement': r'exp(\1 + \2)',
            'category': 'exponential',
            'description': 'Exponential identity: exp(a)*exp(b) = exp(a+b)',
            'speedup': 50.0,
        },
        
        # Algebraic simplifications (identity/stability anchors)
        r'([a-zA-Z_]\w*)\s*\*\s*1(?!\d)': {
            'replacement': r'\1',
            'category': 'algebraic',
            'description': 'Multiply by 1',
            'speedup': 1.0,
        },
        r'([a-zA-Z_]\w*)\s*\+\s*0(?!\d)': {
            'replacement': r'\1',
            'category': 'algebraic',
            'description': 'Add 0',
            'speedup': 1.0,
        },
        r'([a-zA-Z_]\w*)\s*-\s*\1(?!\w)': {
            'replacement': '0',
            'category': 'algebraic',
            'description': 'Subtract self: x - x = 0',
            'speedup': 2.0,
        },
        r'([a-zA-Z_]\w*)\s*/\s*\1(?!\w)': {
            'replacement': '1',
            'category': 'algebraic',
            'description': 'Divide by self: x / x = 1',
            'speedup': 2.0,
        },
        
        # Gaussian to Lorentzian approximation
        r'np\.exp\s*\(\s*-([^)]+)\s*\*\*\s*2\s*/\s*(\d+\.?\d*)\s*\)': {
            'replacement': r'1.0 / (1.0 + \1**2 / \2)',
            'category': 'approximation',
            'description': 'Lorentzian approximation (1.5× faster, corr>0.99)',
            'speedup': 50.0,
        },
    }
    
    def __init__(self, group=None, verbose=True):
        super().__init__(group, verbose)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._compiled = {}
        for pattern, info in self.PATTERNS.items():
            try:
                self._compiled[pattern] = (re.compile(pattern), info)
            except re.error:
                self._log(f"Warning: Invalid pattern: {pattern}")
    
    def optimize_file(self, file_path: str, output: str = None) -> OptimizationResult:
        """
        Scan a file for optimization opportunities.
        
        Args:
            file_path: Path to Python file
            output: Optional path to save JSON report
            
        Returns:
            OptimizationResult with found optimizations
        """
        self._start_timer()
        self._log(f"Scanning {file_path} for optimizations...")
        
        path = Path(file_path)
        if not path.exists():
            return OptimizationResult(
                processor='CodeOptimizer',
                success=False,
                timestamp=self._timestamp(),
                elapsed_time=self._elapsed(),
                findings=[],
                summary=f"File not found: {file_path}",
                source=file_path,
                total_expressions=0,
                optimizations=[],
                estimated_speedup=1.0,
            )
        
        content = path.read_text()
        optimizations = self._find_optimizations(content, file_path)
        
        total_speedup = sum(o['speedup'] for o in optimizations) if optimizations else 0
        
        result = OptimizationResult(
            processor='CodeOptimizer',
            success=len(optimizations) > 0,
            timestamp=self._timestamp(),
            elapsed_time=self._elapsed(),
            findings=optimizations,
            summary=f"Found {len(optimizations)} optimizations, estimated {total_speedup:.0f} FLOPS saved",
            source=file_path,
            total_expressions=len(self._extract_expressions(content)),
            optimizations=optimizations,
            estimated_speedup=1.0 + total_speedup / 1000,
        )
        
        if output:
            result.save(output)
        
        self._log(f"\n{result.summary}")
        return result
    
    def optimize_expressions(self, expressions: List[str], 
                             output: str = None) -> OptimizationResult:
        """
        Analyze a list of expressions for optimizations.
        
        Args:
            expressions: List of mathematical expressions
            output: Optional path to save JSON report
            
        Returns:
            OptimizationResult with found optimizations
        """
        self._start_timer()
        self._log(f"Analyzing {len(expressions)} expressions...")
        
        optimizations = []
        for expr in expressions:
            opts = self._analyze_expression(expr)
            optimizations.extend(opts)
        
        total_speedup = sum(o['speedup'] for o in optimizations) if optimizations else 0
        
        result = OptimizationResult(
            processor='CodeOptimizer',
            success=len(optimizations) > 0,
            timestamp=self._timestamp(),
            elapsed_time=self._elapsed(),
            findings=optimizations,
            summary=f"Found {len(optimizations)} optimizations",
            source="expressions",
            total_expressions=len(expressions),
            optimizations=optimizations,
            estimated_speedup=1.0 + total_speedup / 1000,
        )
        
        if output:
            result.save(output)
        
        return result
    
    def process(self, target: str, output: str = None) -> OptimizationResult:
        """
        Main processing method.
        
        Args:
            target: File path or expression
            output: Optional output path
        """
        if Path(target).exists():
            return self.optimize_file(target, output)
        else:
            return self.optimize_expressions([target], output)
    
    def _find_optimizations(self, content: str, source: str) -> List[Dict]:
        """Find all optimizations in content."""
        optimizations = []
        
        for pattern, (compiled, info) in self._compiled.items():
            for match in compiled.finditer(content):
                original = match.group(0)
                optimized = compiled.sub(info['replacement'], original)
                
                # Get group-theoretic info
                g = self.group.element(original)
                
                opt = Optimization(
                    original=original,
                    optimized=optimized,
                    category=info['category'],
                    speedup=info['speedup'],
                    verified=self._verify_numerically(original, optimized),
                    location=f"{source}:{content[:match.start()].count(chr(10)) + 1}",
                    description=info['description'],
                    group_info={
                        'position': g.position.to_dict(),
                        'dominant_anchor': g.position.dominant_anchor()[0].name,
                    }
                )
                optimizations.append(asdict(opt))
        
        return optimizations
    
    def _analyze_expression(self, expr: str) -> List[Dict]:
        """Analyze a single expression."""
        optimizations = []
        
        for pattern, (compiled, info) in self._compiled.items():
            match = compiled.search(expr)
            if match:
                original = match.group(0)
                optimized = compiled.sub(info['replacement'], original)
                
                g = self.group.element(original)
                
                opt = Optimization(
                    original=original,
                    optimized=optimized,
                    category=info['category'],
                    speedup=info['speedup'],
                    verified=self._verify_numerically(original, optimized),
                    description=info['description'],
                    group_info={
                        'position': g.position.to_dict(),
                        'dominant_anchor': g.position.dominant_anchor()[0].name,
                    }
                )
                optimizations.append(asdict(opt))
        
        return optimizations
    
    def _extract_expressions(self, content: str) -> List[str]:
        """Extract mathematical expressions from code."""
        # Simple extraction - look for function calls
        pattern = r'(?:sin|cos|tan|exp|log|sqrt|arctan|arcsin|arccos)\s*\([^)]+\)'
        return re.findall(pattern, content)
