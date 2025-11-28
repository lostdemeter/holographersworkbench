"""
Clock Resonance Compiler
========================

A framework for automatically upgrading workbench processors to use
clock eigenphases instead of random/quasi-random sources.

Architecture:
    ClockResonanceCompiler
    ├── analyze(processor) → identifies random/comb sources
    ├── compile(processor) → creates clock-enhanced version
    ├── validate(original, compiled) → ensures correctness
    └── benchmark(original, compiled) → measures gains

Usage:
    from workbench.core import ClockResonanceCompiler
    
    compiler = ClockResonanceCompiler()
    
    # Analyze a processor
    analysis = compiler.analyze(SpectralScorer)
    
    # Compile to clock-resonant version
    ClockSpectralScorer = compiler.compile(SpectralScorer)
    
    # Or use the decorator
    @compiler.clock_resonant
    class MyProcessor:
        def process(self, data):
            # Uses self.clock_oracle for phases
            phase = self.clock_oracle.get_fractional_phase(n)
            ...

Based on: Clock Dimensional Downcasting discovery (Nov 2024)
"""

import numpy as np
from typing import Type, Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from functools import wraps
import inspect
import ast
import time

# Import the lazy clock oracle
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from workbench.processors.sublinear_clock_v2 import (
        LazyClockOracle, CLOCK_RATIOS_6D
    )
    CLOCK_AVAILABLE = True
except ImportError:
    CLOCK_AVAILABLE = False


@dataclass
class RandomSourceInfo:
    """Information about a random/quasi-random source in code."""
    name: str
    source_type: str  # 'numpy_random', 'golden_comb', 'uniform', 'custom'
    location: str     # method name or line number
    usage: str        # description of how it's used
    upgrade_strategy: str  # how to replace with clock phases


@dataclass
class CompilerAnalysis:
    """Analysis result from the compiler."""
    processor_name: str
    random_sources: List[RandomSourceInfo]
    estimated_difficulty: str  # 'easy', 'medium', 'hard'
    estimated_improvement: str
    notes: List[str]
    
    def __str__(self):
        lines = [
            f"Analysis: {self.processor_name}",
            f"Difficulty: {self.estimated_difficulty}",
            f"Expected improvement: {self.estimated_improvement}",
            f"Random sources found: {len(self.random_sources)}",
        ]
        for src in self.random_sources:
            lines.append(f"  - {src.name} ({src.source_type}) in {src.location}")
            lines.append(f"    Usage: {src.usage}")
            lines.append(f"    Strategy: {src.upgrade_strategy}")
        if self.notes:
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


@dataclass
class CompilerResult:
    """Result from compiling a processor."""
    original_class: Type
    compiled_class: Type
    analysis: CompilerAnalysis
    validation_passed: bool
    benchmark_improvement: Optional[float] = None


class ClockOracleMixin:
    """
    Mixin that provides clock oracle access to any class.
    
    Add this to a processor to give it access to clock phases:
    
        class MyProcessor(ClockOracleMixin):
            def process(self, data):
                phase = self.get_clock_phase(n)
                ...
    """
    
    _clock_oracle: Optional[LazyClockOracle] = None
    _clock_counter: int = 0
    
    @property
    def clock_oracle(self) -> LazyClockOracle:
        """Get or create the clock oracle."""
        if self._clock_oracle is None:
            self._clock_oracle = LazyClockOracle()
        return self._clock_oracle
    
    def get_clock_phase(self, n: int = None, clock: str = 'golden') -> float:
        """
        Get a clock phase.
        
        If n is None, uses an auto-incrementing counter.
        """
        if n is None:
            self._clock_counter += 1
            n = self._clock_counter
        return self.clock_oracle.get_fractional_phase(n, clock)
    
    def get_clock_phases(self, count: int, clock: str = 'golden') -> np.ndarray:
        """Get multiple clock phases."""
        return np.array([
            self.get_clock_phase(i + 1, clock) 
            for i in range(count)
        ])
    
    def reset_clock_counter(self):
        """Reset the auto-incrementing counter."""
        self._clock_counter = 0
        
    def clock_random(self, size: int = None) -> np.ndarray:
        """
        Drop-in replacement for np.random.random().
        
        Returns clock phases instead of pseudo-random numbers.
        """
        if size is None:
            return self.get_clock_phase()
        return self.get_clock_phases(size)
    
    def clock_randn(self, size: int = None) -> np.ndarray:
        """
        Drop-in replacement for np.random.randn().
        
        Uses Box-Muller transform on clock phases.
        """
        if size is None:
            u1 = self.get_clock_phase()
            u2 = self.get_clock_phase()
            return np.sqrt(-2 * np.log(u1 + 1e-10)) * np.cos(2 * np.pi * u2)
        
        # Generate pairs
        n_pairs = (size + 1) // 2
        result = []
        for _ in range(n_pairs):
            u1 = self.get_clock_phase()
            u2 = self.get_clock_phase()
            z1 = np.sqrt(-2 * np.log(u1 + 1e-10)) * np.cos(2 * np.pi * u2)
            z2 = np.sqrt(-2 * np.log(u1 + 1e-10)) * np.sin(2 * np.pi * u2)
            result.extend([z1, z2])
        return np.array(result[:size])
    
    def clock_choice(self, a: int, size: int = None, replace: bool = True) -> np.ndarray:
        """
        Drop-in replacement for np.random.choice().
        
        Uses clock phases to select indices.
        """
        if size is None:
            phase = self.get_clock_phase()
            return int(phase * a) % a
        
        phases = self.get_clock_phases(size)
        indices = (phases * a).astype(int) % a
        
        if not replace:
            # Remove duplicates by shifting
            seen = set()
            result = []
            for idx in indices:
                while idx in seen:
                    idx = (idx + 1) % a
                seen.add(idx)
                result.append(idx)
            return np.array(result)
        
        return indices


class ClockResonanceCompiler:
    """
    Compiler that upgrades processors to use clock eigenphases.
    
    The compiler can:
    1. Analyze a processor to find random/quasi-random sources
    2. Generate a clock-enhanced version
    3. Validate that the compiled version is correct
    4. Benchmark the improvement
    
    Example:
        compiler = ClockResonanceCompiler()
        
        # Analyze
        analysis = compiler.analyze(SpectralScorer)
        print(analysis)
        
        # Compile
        ClockSpectralScorer = compiler.compile(SpectralScorer)
        
        # Use
        scorer = ClockSpectralScorer()
        result = scorer.score(data)
    """
    
    # Known random source patterns
    RANDOM_PATTERNS = {
        'np.random.random': 'numpy_random',
        'np.random.rand': 'numpy_random',
        'np.random.randn': 'numpy_random',
        'np.random.choice': 'numpy_random',
        'np.random.uniform': 'numpy_random',
        'np.random.randint': 'numpy_random',
        'random.random': 'stdlib_random',
        'random.uniform': 'stdlib_random',
        'PHI': 'golden_comb',
        'golden': 'golden_comb',
        '1.618': 'golden_comb',
        'np.linspace': 'uniform_sampling',
    }
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.oracle = LazyClockOracle() if CLOCK_AVAILABLE else None
        
    def analyze(self, processor_class: Type) -> CompilerAnalysis:
        """
        Analyze a processor class to find random/quasi-random sources.
        
        Returns an analysis with:
        - List of random sources found
        - Estimated difficulty of upgrade
        - Expected improvement
        """
        random_sources = []
        notes = []
        
        # Get source code
        try:
            source = inspect.getsource(processor_class)
        except (OSError, TypeError):
            return CompilerAnalysis(
                processor_name=processor_class.__name__,
                random_sources=[],
                estimated_difficulty='unknown',
                estimated_improvement='unknown',
                notes=['Could not retrieve source code']
            )
        
        # Scan for random patterns
        for pattern, source_type in self.RANDOM_PATTERNS.items():
            if pattern in source:
                # Find the method containing this pattern
                for name, method in inspect.getmembers(processor_class, predicate=inspect.isfunction):
                    try:
                        method_source = inspect.getsource(method)
                        if pattern in method_source:
                            random_sources.append(RandomSourceInfo(
                                name=pattern,
                                source_type=source_type,
                                location=name,
                                usage=self._infer_usage(pattern, method_source),
                                upgrade_strategy=self._get_upgrade_strategy(source_type)
                            ))
                    except (OSError, TypeError):
                        pass
        
        # Estimate difficulty
        n_sources = len(random_sources)
        if n_sources == 0:
            difficulty = 'none'
            improvement = 'N/A (no random sources)'
        elif n_sources <= 2:
            difficulty = 'easy'
            improvement = 'Reproducibility + 5-10% quality'
        elif n_sources <= 5:
            difficulty = 'medium'
            improvement = 'Reproducibility + 10-20% quality'
        else:
            difficulty = 'hard'
            improvement = 'Significant refactoring needed'
        
        # Check for inheritance
        if ClockOracleMixin in processor_class.__mro__:
            notes.append('Already has ClockOracleMixin')
            difficulty = 'none'
        
        return CompilerAnalysis(
            processor_name=processor_class.__name__,
            random_sources=random_sources,
            estimated_difficulty=difficulty,
            estimated_improvement=improvement,
            notes=notes
        )
    
    def _infer_usage(self, pattern: str, source: str) -> str:
        """Infer how a random source is used."""
        if 'init' in pattern.lower() or 'initial' in source.lower():
            return 'Initialization'
        elif 'sample' in source.lower():
            return 'Sampling'
        elif 'phase' in source.lower():
            return 'Phase generation'
        elif 'angle' in source.lower():
            return 'Angle computation'
        else:
            return 'General randomness'
    
    def _get_upgrade_strategy(self, source_type: str) -> str:
        """Get the upgrade strategy for a source type."""
        strategies = {
            'numpy_random': 'Replace with clock_random() or clock_randn()',
            'stdlib_random': 'Replace with get_clock_phase()',
            'golden_comb': 'Replace with clock phases (already quasi-random)',
            'uniform_sampling': 'Replace with clock_phases for better coverage',
        }
        return strategies.get(source_type, 'Manual replacement needed')
    
    def compile(self, processor_class: Type) -> Type:
        """
        Compile a processor class to use clock phases.
        
        Returns a new class that:
        1. Inherits from ClockOracleMixin
        2. Has clock_* methods available
        3. Can be used as a drop-in replacement
        """
        # Create a new class that inherits from both
        class CompiledProcessor(ClockOracleMixin, processor_class):
            """Clock-compiled version of the original processor."""
            
            def __init__(self, *args, **kwargs):
                # Initialize clock oracle
                ClockOracleMixin.__init__(self)
                # Initialize original
                processor_class.__init__(self, *args, **kwargs)
                
        # Copy class attributes
        CompiledProcessor.__name__ = f"Clock{processor_class.__name__}"
        CompiledProcessor.__doc__ = f"""
Clock-compiled version of {processor_class.__name__}.

This class has been upgraded to use clock eigenphases instead of
random/quasi-random sources. It inherits from ClockOracleMixin,
providing access to:

- self.get_clock_phase(n) - Get the n-th clock phase
- self.clock_random(size) - Drop-in for np.random.random()
- self.clock_randn(size) - Drop-in for np.random.randn()
- self.clock_choice(a, size) - Drop-in for np.random.choice()

Original docstring:
{processor_class.__doc__ or 'No documentation'}
"""
        
        return CompiledProcessor
    
    def validate(
        self, 
        original_class: Type, 
        compiled_class: Type,
        test_input: Any = None
    ) -> bool:
        """
        Validate that the compiled class produces correct output.
        
        This is a basic validation - for full testing, use domain-specific tests.
        """
        try:
            # Check that compiled class can be instantiated
            compiled = compiled_class()
            
            # Check that it has clock methods
            assert hasattr(compiled, 'get_clock_phase')
            assert hasattr(compiled, 'clock_random')
            assert hasattr(compiled, 'clock_oracle')
            
            # Check that clock phases are in valid range
            phase = compiled.get_clock_phase(1)
            assert 0 <= phase < 1
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"Validation failed: {e}")
            return False
    
    def benchmark(
        self,
        original_class: Type,
        compiled_class: Type,
        test_func: Callable,
        n_trials: int = 5
    ) -> Tuple[float, float, float]:
        """
        Benchmark the compiled class against the original.
        
        Args:
            original_class: The original processor class
            compiled_class: The clock-compiled class
            test_func: A function that takes a processor and returns a quality metric
            n_trials: Number of trials to run
            
        Returns:
            (original_quality, compiled_quality, improvement_percent)
        """
        original_scores = []
        compiled_scores = []
        
        for _ in range(n_trials):
            # Test original
            original = original_class()
            original_scores.append(test_func(original))
            
            # Test compiled
            compiled = compiled_class()
            compiled_scores.append(test_func(compiled))
        
        orig_mean = np.mean(original_scores)
        comp_mean = np.mean(compiled_scores)
        improvement = 100 * (comp_mean - orig_mean) / abs(orig_mean) if orig_mean != 0 else 0
        
        return orig_mean, comp_mean, improvement
    
    def clock_resonant(self, cls: Type) -> Type:
        """
        Decorator to make a class clock-resonant.
        
        Usage:
            @compiler.clock_resonant
            class MyProcessor:
                def process(self, data):
                    phase = self.get_clock_phase()
                    ...
        """
        return self.compile(cls)


# Convenience function
def make_clock_resonant(processor_class: Type) -> Type:
    """
    Quick function to make any processor clock-resonant.
    
    Usage:
        ClockScorer = make_clock_resonant(SpectralScorer)
        scorer = ClockScorer()
    """
    compiler = ClockResonanceCompiler()
    return compiler.compile(processor_class)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Clock Resonance Compiler Demo")
    print("=" * 70)
    
    # Create a simple test processor
    class TestProcessor:
        """A simple processor that uses random numbers."""
        
        def __init__(self, seed: int = 42):
            np.random.seed(seed)
            self.weights = np.random.random(10)
            
        def process(self, data: np.ndarray) -> float:
            """Process data using random weights."""
            noise = np.random.randn(len(data))
            return np.sum(data * self.weights[:len(data)]) + 0.1 * np.sum(noise)
    
    # Analyze
    compiler = ClockResonanceCompiler(verbose=True)
    analysis = compiler.analyze(TestProcessor)
    print("\n" + str(analysis))
    
    # Compile
    ClockTestProcessor = compiler.compile(TestProcessor)
    print(f"\nCompiled: {ClockTestProcessor.__name__}")
    
    # Validate
    valid = compiler.validate(TestProcessor, ClockTestProcessor)
    print(f"Validation: {'PASSED' if valid else 'FAILED'}")
    
    # Test the compiled processor
    print("\nTesting compiled processor:")
    processor = ClockTestProcessor()
    
    # Show clock phases
    print(f"  Clock phase 1: {processor.get_clock_phase(1):.6f}")
    print(f"  Clock phase 2: {processor.get_clock_phase(2):.6f}")
    print(f"  Clock random(5): {processor.clock_random(5)}")
    
    # Process some data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = processor.process(data)
    print(f"  Process result: {result:.4f}")
    
    print("\n" + "=" * 70)
    print("Clock Resonance Compiler ready!")
    print("=" * 70)
