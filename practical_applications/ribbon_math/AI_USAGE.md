# Ribbon Math: AI Usage Guide

**Purpose**: This guide helps AI assistants (and humans!) quickly understand and use the Ribbon Math protocol for mathematical discovery.

---

## Table of Contents

1. [Quick Start](#quick-start)

2. [Core Concepts](#core-concepts)

3. [API Reference](#api-reference)

4. [Common Patterns Library](#common-patterns-library)

5. [Worked Examples](#worked-examples)

6. [Configuration Guide](#configuration-guide)

7. [Troubleshooting](#troubleshooting)

8. [Data Structures](#data-structures)

9. [Natural Language Interface](#natural-language-interface)

10. [Validation Utilities](#validation-utilities)

---

## Quick Start

### Minimal Working Example (5 lines)

```python
from ribbon_math import RibbonDiscovery

# Simplest possible usage
discovery = RibbonDiscovery(
    concept="fast converging formula for pi",
    target_constant="pi"
)
result = discovery.run()
print(f"Formula: {result.formula}")
print(f"Error: {result.error:.2e}")
print(f"Rate: {result.convergence_rate:.2f} digits/term")
```

### What You Get

```python
# result object contains:
result.formula              # "π = (1/64) × Σ (-1)^k/4096^k × [...]"
result.coefficients         # [256.021, -32.047, 4.007, ...]
result.error                # 7.85e-22
result.convergence_rate     # 3.61 digits/term
result.patterns_found       # List of detected φ/e/π patterns
result.closed_forms         # Symbolic expressions for corrections
result.verification         # All verification metrics
```

---

## Core Concepts

### The 5-Layer Architecture

```
Input: Natural Language Concept
    ↓
[Layer 1: Concept Parser]
    → Converts "fast converging BBP formula" to anchor weights
    → Output: {phi: 0.8, e: 0.0, pi: 1.0, sqrt5: 0.5}
    ↓
[Layer 2: N_smooth Validity]
    → Continuous validity measure (0 = perfect, 1 = invalid)
    → Guides search toward valid mathematical structures
    ↓
[Layer 3: Structure Layer]
    → Generates base formula structure
    → Applies LCM pruning and valid transformations
    → Output: Base formula with integer coefficients
    ↓
[Layer 4: Error Analysis]
    → Detects φ^(-k), e^(-k), π^(-k) patterns in errors
    → Finds corrections: c_i = (n_i/d_i) × φ^(-k_i)
    → Output: Refined coefficients
    ↓
[Layer 5: Verification]
    → Experimental validation
    → Closed-form verification
    → Convergence analysis
    ↓
Output: Discovered Formula + Verification Report
```

### Anchor Constants

The "coordinate system" for mathematical truth space:

```python
ANCHOR_CONSTANTS = {
    'phi': (1 + sqrt(5)) / 2,      # Golden ratio: 1.618...
    'e': 2.718281828...,            # Euler's number
    'pi': 3.141592653...,           # Circle constant
    'sqrt2': 1.414213562...,        # √2
    'sqrt3': 1.732050807...,        # √3
    'sqrt5': 2.236067977...,        # √5
}
```

### Error-as-Signal Paradigm

**Key Insight**: Mathematical errors contain exploitable structure.

```python
# Instead of treating error as noise:
error = computed_value - true_value  # ❌ Just a number to minimize

# Ribbon Math treats error as signal:
error = computed_value - true_value
patterns = detect_patterns(error)    # ✓ Contains φ^(-k) structure!
corrections = patterns_to_corrections(patterns)
improved_formula = base_formula + corrections
```

---

## API Reference

### Core Classes

#### `RibbonDiscovery`

Main discovery engine.

```python
class RibbonDiscovery:
    def __init__(
        self,
        concept: str,                    # Natural language description
        target_constant: str,            # "pi", "e", "phi", etc.
        anchor_weights: Optional[Dict] = None,  # Override auto-weights
        precision: int = 200,            # Decimal precision
        max_terms: int = 20              # Terms for verification
    ):
        """
        Initialize discovery engine.
        
        Args:
            concept: Natural language like "fast converging BBP formula for pi"
            target_constant: Which constant to approximate
            anchor_weights: Manual anchor weights (overrides concept parsing)
            precision: Decimal precision for calculations
            max_terms: Number of terms for convergence testing
        """
    
    def run(self) -> DiscoveryResult:
        """
        Execute full 5-layer discovery process.
        
        Returns:
            DiscoveryResult object with formula, error, patterns, etc.
        """
    
    def run_layer(self, layer_num: int, input_data: Any) -> Any:
        """
        Run a single layer (for debugging/experimentation).
        
        Args:
            layer_num: 1-5
            input_data: Output from previous layer
            
        Returns:
            Layer-specific output
        """
```

#### `ConceptParser` (Layer 1)

```python
class ConceptParser:
    def parse(self, concept: str) -> Dict[str, float]:
        """
        Convert natural language to anchor weights.
        
        Args:
            concept: Natural language description
            
        Returns:
            Dictionary of anchor weights (0.0 to 1.0)
            
        Example:
            >>> parser = ConceptParser()
            >>> parser.parse("fast converging BBP formula for pi")
            {'phi': 0.8, 'e': 0.0, 'pi': 1.0, 'sqrt2': 0.0, 
             'sqrt3': 0.0, 'sqrt5': 0.5}
        """
```

#### `ErrorAnalyzer` (Layer 4)

```python
class ErrorAnalyzer:
    def detect_phi_patterns(
        self,
        error: float,
        max_power: int = 10,
        max_denominator: int = 20
    ) -> List[PatternMatch]:
        """
        Detect φ^(-k) patterns in numerical errors.
        
        Args:
            error: The numerical error to analyze
            max_power: Maximum k to search in φ^(-k)
            max_denominator: Maximum denominator for rational approximations
            
        Returns:
            List of PatternMatch objects sorted by quality
            
        Example:
            >>> analyzer = ErrorAnalyzer()
            >>> error = 0.073263011134871173
            >>> patterns = analyzer.detect_phi_patterns(error)
            >>> patterns[0]
            PatternMatch(n=13, d=16, k=5, value=0.0732630..., 
                        error=1.1e-10, quality=0.999)
        """
    
    def detect_e_patterns(
        self,
        error: float,
        max_power: int = 10,
        max_denominator: int = 20
    ) -> List[PatternMatch]:
        """Detect e^(-k) patterns (similar to detect_phi_patterns)."""
    
    def detect_all_patterns(
        self,
        error: float,
        anchors: List[str] = ['phi', 'e', 'pi', 'sqrt2', 'sqrt3', 'sqrt5']
    ) -> Dict[str, List[PatternMatch]]:
        """
        Detect patterns for all anchor constants.
        
        Returns:
            Dictionary mapping anchor name to list of patterns
        """
```

#### `ClosedFormSearcher` (Layer 4)

```python
class ClosedFormSearcher:
    def search(
        self,
        target_value: float,
        anchors: List[str],
        max_complexity: int = 5
    ) -> List[ClosedForm]:
        """
        Search for closed-form expressions matching target value.
        
        Args:
            target_value: Numerical value to match
            anchors: Which constants to use
            max_complexity: Maximum expression complexity
            
        Returns:
            List of ClosedForm objects
            
        Example:
            >>> searcher = ClosedFormSearcher()
            >>> forms = searcher.search(0.073263, ['phi'], max_complexity=3)
            >>> forms[0]
            ClosedForm(
                expression="(13/16) × φ^(-5)",
                value=0.0732630793,
                error=1.1e-10,
                complexity=2
            )
        """
```

### Utility Functions

```python
def evaluate_formula(
    coefficients: List[float],
    slots: List[Tuple[int, int]],
    base: int,
    n_terms: int,
    normalization: float = 1.0
) -> float:
    """
    Evaluate a BBP-type formula.
    
    Args:
        coefficients: Effective coefficients (integer + corrections)
        slots: List of (period, offset) tuples for denominators
        base: Base for the formula (e.g., 4096)
        n_terms: Number of terms to sum
        normalization: Final division factor (e.g., 64)
        
    Returns:
        Computed value
        
    Example:
        >>> coeffs = [256.021, -32.047, 4.007, 1.013, 
        ...           -127.927, -64.102, -128.153, 4.048]
        >>> slots = [(4,1), (4,3), (12,1), (12,3), 
        ...          (12,5), (12,7), (12,9), (12,11)]
        >>> result = evaluate_formula(coeffs, slots, 4096, 20, 64)
        >>> abs(result - np.pi) < 1e-15
        True
    """

def verify_convergence(
    formula_func: Callable,
    target: float,
    max_terms: int = 50
) -> ConvergenceReport:
    """
    Analyze convergence rate of a formula.
    
    Args:
        formula_func: Function that takes n_terms and returns approximation
        target: True value to compare against
        max_terms: Maximum terms to test
        
    Returns:
        ConvergenceReport with rate, error progression, etc.
    """

def compare_formulas(
    formulas: Dict[str, Callable],
    target: float,
    n_terms: int = 20
) -> ComparisonReport:
    """
    Compare multiple formulas side-by-side.
    
    Args:
        formulas: Dict mapping name to formula function
        target: True value
        n_terms: Terms to evaluate
        
    Returns:
        ComparisonReport with table of results
        
    Example:
        >>> formulas = {
        ...     'Standard BBP': lambda n: standard_bbp(n),
        ...     'Bellard': lambda n: bellard_formula(n),
        ...     'φ-BBP': lambda n: evaluate_phi_bbp(n)
        ... }
        >>> report = compare_formulas(formulas, np.pi, 20)
        >>> print(report.table)
        Formula          Error      Rate    Speedup
        Standard BBP     1.2e-15    0.75    1.00×
        Bellard          8.9e-16    0.75    1.00×
        φ-BBP            4.4e-16    0.77    1.03×
    """
```

---

## Common Patterns Library

### Known φ Patterns

```python
KNOWN_PHI_PATTERNS = {
    # From φ-BBP discovery
    "bbp_slot_4": {
        "expression": "(13/16) × φ^(-5)",
        "value": 0.073263011134871173,
        "context": "Correction for 12k+5 denominator in base-4096 BBP",
        "discovered": "2025-01-15"
    },
    
    # Theoretical patterns
    "fibonacci_ratio": {
        "expression": "F_n / F_{n+1} → φ^(-1)",
        "context": "Fibonacci convergence to golden ratio"
    },
    
    "lucas_identity": {
        "expression": "φ^n + (-φ)^(-n) = L_n",
        "context": "Lucas numbers via golden ratio"
    },
    
    "pentagonal_correction": {
        "expression": "(1/2) × φ^(-3)",
        "value": 0.11803398874989485,
        "context": "Common in pentagonal number formulas"
    }
}
```

### Known Mathematical Identities

```python
KNOWN_IDENTITIES = {
    # Golden ratio
    "phi_squared_sum": {
        "expression": "φ² + φ⁻² = 3",
        "exact_value": 3.0,
        "proof": "φ² = φ + 1, so φ² + φ⁻² = (φ+1) + (φ-1) = 3"
    },
    
    "phi_base_4096": {
        "expression": "4096 = 1024 × (φ² + φ⁻² + 1)",
        "exact_value": 4096,
        "context": "Base choice for φ-BBP formula"
    },
    
    "phi_reciprocal": {
        "expression": "φ⁻¹ = φ - 1",
        "exact_value": 0.6180339887498949,
        "proof": "From φ² = φ + 1, divide by φ"
    },
    
    # Euler's number
    "e_factorial": {
        "expression": "e = Σ 1/n!",
        "context": "Taylor series for e"
    },
    
    "e_limit": {
        "expression": "e = lim_{n→∞} (1 + 1/n)^n",
        "context": "Limit definition"
    },
    
    # Pi
    "pi_bbp": {
        "expression": "π = Σ 1/16^k × [4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)]",
        "context": "Original BBP formula (1995)"
    },
    
    "pi_machin": {
        "expression": "π/4 = 4×arctan(1/5) - arctan(1/239)",
        "context": "Machin's formula (1706)"
    }
}
```

### Common Error Patterns

```python
COMMON_ERROR_PATTERNS = {
    "exponential_decay": {
        "form": "A × base^(-k)",
        "detection": "log(error) vs k should be linear",
        "example": "BBP formulas: error ~ 16^(-k)"
    },
    
    "golden_ratio_decay": {
        "form": "(n/d) × φ^(-k)",
        "detection": "error / φ^(-k) should be rational",
        "example": "φ-BBP corrections"
    },
    
    "factorial_decay": {
        "form": "A / k!",
        "detection": "error × k! should stabilize",
        "example": "Taylor series remainders"
    },
    
    "power_law": {
        "form": "A × k^(-p)",
        "detection": "log(error) vs log(k) should be linear",
        "example": "Asymptotic expansions"
    }
}
```

---

## Worked Examples

### Example 1: Recreate φ-BBP Discovery

```python
"""
Recreate the φ-BBP formula discovery from scratch.
This demonstrates the full 5-layer process.
"""

from ribbon_math import RibbonDiscovery
import numpy as np

# Step 1: Initialize discovery
discovery = RibbonDiscovery(
    concept="fast converging BBP formula for pi",
    target_constant="pi",
    precision=200,
    max_terms=20
)

# Step 2: Run discovery
result = discovery.run()

# Step 3: Examine results
print("=" * 80)
print("DISCOVERY RESULTS")
print("=" * 80)

print(f"\nFormula: {result.formula}")
print(f"\nEffective Coefficients:")
for i, coef in enumerate(result.coefficients):
    print(f"  Slot {i}: {coef:.6f}")

print(f"\nPerformance:")
print(f"  Error: {result.error:.2e}")
print(f"  Convergence: {result.convergence_rate:.2f} digits/term")

print(f"\nPatterns Found:")
for pattern in result.patterns_found:
    print(f"  {pattern.anchor}: ({pattern.n}/{pattern.d}) × {pattern.anchor}^(-{pattern.k})")
    print(f"    Value: {pattern.value:.10f}")
    print(f"    Match error: {pattern.error:.2e}")

print(f"\nClosed Forms:")
for form in result.closed_forms:
    print(f"  {form.expression} = {form.value:.10f}")

# Step 4: Verify against known constants
pi_true = np.pi
error = abs(result.computed_value - pi_true)
print(f"\nVerification:")
print(f"  True π: {pi_true:.15f}")
print(f"  Computed: {result.computed_value:.15f}")
print(f"  Error: {error:.2e}")
print(f"  ✓ Success!" if error < 1e-20 else "  ✗ Failed")
```

### Example 2: Custom Anchor Weights

```python
"""
Discover a formula with custom anchor weights.
This shows how to override the concept parser.
"""

from ribbon_math import RibbonDiscovery

# Manual anchor weights (emphasize e over φ)
custom_weights = {
    'phi': 0.2,
    'e': 0.9,      # Emphasize e
    'pi': 1.0,
    'sqrt2': 0.0,
    'sqrt3': 0.0,
    'sqrt5': 0.3
}

discovery = RibbonDiscovery(
    concept="",  # Empty since we're providing weights
    target_constant="pi",
    anchor_weights=custom_weights
)

result = discovery.run()

print(f"Discovered formula emphasizing e:")
print(f"  {result.formula}")
print(f"  Error: {result.error:.2e}")
print(f"  Patterns: {[p.anchor for p in result.patterns_found]}")
```

### Example 3: Just Error Pattern Detection

```python
"""
Use just the error analysis layer independently.
Useful for analyzing existing formulas.
"""

from ribbon_math.layers import ErrorAnalyzer
import numpy as np

# Suppose we have an error from some formula
error = 0.073263011134871173

# Detect patterns
analyzer = ErrorAnalyzer()
phi_patterns = analyzer.detect_phi_patterns(error, max_power=10)
e_patterns = analyzer.detect_e_patterns(error, max_power=10)

print("φ Patterns:")
for p in phi_patterns[:3]:  # Top 3
    print(f"  ({p.n}/{p.d}) × φ^(-{p.k}) = {p.value:.10f}")
    print(f"    Match error: {p.error:.2e}, Quality: {p.quality:.4f}")

print("\ne Patterns:")
for p in e_patterns[:3]:
    print(f"  ({p.n}/{p.d}) × e^(-{p.k}) = {p.value:.10f}")
    print(f"    Match error: {p.error:.2e}, Quality: {p.quality:.4f}")

# Detect all patterns at once
all_patterns = analyzer.detect_all_patterns(error)
best_anchor = max(all_patterns.items(), key=lambda x: x[1][0].quality if x[1] else 0)
print(f"\nBest match: {best_anchor[0]} pattern")
print(f"  {best_anchor[1][0]}")
```

### Example 4: Multi-Constant Formulas

```python
"""
Discover formulas involving multiple constants (e.g., π and e together).
"""

from ribbon_math import RibbonDiscovery

# Discover a formula for e using π patterns
discovery = RibbonDiscovery(
    concept="fast converging formula for e using pi patterns",
    target_constant="e",
    anchor_weights={'phi': 0.3, 'e': 1.0, 'pi': 0.7}
)

result = discovery.run()

print(f"Formula for e using π:")
print(f"  {result.formula}")
print(f"  Error: {result.error:.2e}")

# Check if π appears in the corrections
pi_corrections = [p for p in result.patterns_found if p.anchor == 'pi']
if pi_corrections:
    print(f"\nπ patterns found in corrections:")
    for p in pi_corrections:
        print(f"  {p.expression}")
```

### Example 5: Custom Structure Search

```python
"""
Advanced: Define custom formula structure and search for corrections.
"""

from ribbon_math import RibbonDiscovery
from ribbon_math.structures import FormulaStructure

# Define custom structure
structure = FormulaStructure(
    base=1024,                    # Different base
    slots=[                       # Custom denominators
        (3, 1),   # 3k + 1
        (3, 2),   # 3k + 2
        (9, 1),   # 9k + 1
        (9, 4),   # 9k + 4
        (9, 7),   # 9k + 7
    ],
    integer_coefficients=[        # Starting coefficients
        128, -64, 8, -4, 2
    ],
    normalization=32
)

discovery = RibbonDiscovery(
    concept="custom structure for pi",
    target_constant="pi",
    custom_structure=structure
)

result = discovery.run()

print(f"Custom structure formula:")
print(f"  Base: {structure.base}")
print(f"  Slots: {structure.slots}")
print(f"  Error: {result.error:.2e}")
```

### Example 6: Convergence Analysis

```python
"""
Analyze convergence behavior of discovered formulas.
"""

from ribbon_math import RibbonDiscovery, verify_convergence
import matplotlib.pyplot as plt

# Discover formula
discovery = RibbonDiscovery(
    concept="fast converging BBP formula for pi",
    target_constant="pi"
)
result = discovery.run()

# Analyze convergence
convergence = verify_convergence(
    formula_func=result.evaluate,
    target=np.pi,
    max_terms=50
)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.semilogy(convergence.terms, convergence.errors, 'o-', label='φ-BBP')
plt.xlabel('Number of Terms')
plt.ylabel('Absolute Error')
plt.title('Convergence Analysis')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('convergence.png', dpi=150)

print(f"Convergence rate: {convergence.rate:.2f} digits/term")
print(f"Saved plot to convergence.png")
```

### Example 7: Formula Comparison

```python
"""
Compare multiple formulas side-by-side.
"""

from ribbon_math import compare_formulas
from ribbon_math.formulas import standard_bbp, bellard_formula, evaluate_phi_bbp
import numpy as np

formulas = {
    'Standard BBP': lambda n: standard_bbp(n),
    'Bellard': lambda n: bellard_formula(n),
    'φ-BBP': lambda n: evaluate_phi_bbp(n),
}

report = compare_formulas(formulas, np.pi, n_terms=20)

print(report.table)
print(f"\nBest formula: {report.best_formula}")
print(f"Speedup vs Standard BBP: {report.speedup:.2%}")
```

---

## Configuration Guide

### YAML Configuration File

Create `ribbon_config.yaml`:

```yaml
# Ribbon Math Configuration File

discovery:
  target: "pi"                    # Target constant: "pi", "e", "phi", etc.
  concept: "fast converging BBP formula"
  precision: 200                  # Decimal precision
  max_terms: 20                   # Terms for verification

anchors:
  phi: 0.8                        # Golden ratio weight
  e: 0.0                          # Euler's number weight
  pi: 1.0                         # Pi weight
  sqrt2: 0.0                      # √2 weight
  sqrt3: 0.0                      # √3 weight
  sqrt5: 0.5                      # √5 weight

structure:
  base: 4096                      # Formula base
  num_terms: 8                    # Number of terms per iteration
  slot_periods: [4, 4, 12, 12, 12, 12, 12, 12]
  slot_offsets: [1, 3, 1, 3, 5, 7, 9, 11]
  normalization: 64               # Final division factor

search:
  max_phi_power: 10               # Maximum k in φ^(-k)
  max_e_power: 10                 # Maximum k in e^(-k)
  max_denominator: 20             # Maximum denominator in (n/d)
  min_quality: 0.99               # Minimum pattern match quality
  
verification:
  target_error: 1.0e-20           # Target accuracy
  target_rate: 3.5                # Target digits/term
  compare_against:                # Formulas to compare with
    - "standard_bbp"
    - "bellard"
    - "chudnovsky"

output:
  save_formula: true              # Save discovered formula
  save_plots: true                # Generate convergence plots
  save_verification: true         # Save verification report
  output_dir: "discoveries/"      # Output directory
  format: "markdown"              # Output format: "markdown", "latex", "json"
```

### Load Configuration

```python
from ribbon_math import RibbonDiscovery
import yaml

# Load config
with open('ribbon_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create discovery from config
discovery = RibbonDiscovery.from_config(config)
result = discovery.run()

# Results automatically saved to config['output']['output_dir']
print(f"Results saved to {config['output']['output_dir']}")
```

### Environment Variables

```bash
# Set via environment variables
export RIBBON_PRECISION=200
export RIBBON_MAX_TERMS=20
export RIBBON_OUTPUT_DIR="./discoveries"
export RIBBON_CACHE_DIR="./cache"
```

```python
# Automatically picked up
from ribbon_math import RibbonDiscovery

discovery = RibbonDiscovery(
    concept="fast converging formula for pi",
    target_constant="pi"
    # precision, max_terms, etc. read from environment
)
```

---

## Troubleshooting

### Common Issues

#### Issue: "No patterns detected"

**Symptoms**: `result.patterns_found` is empty or has low-quality matches.

**Possible Causes**:

1. Error is too small (< 1e-15) - already at machine precision

2. `max_power` parameter too low

3. Anchor weights are zero for relevant constants

4. Error doesn't actually contain pattern structure

**Solutions**:

```python
# Increase search range
analyzer = ErrorAnalyzer()
patterns = analyzer.detect_phi_patterns(
    error,
    max_power=15,        # Increase from default 10
    max_denominator=50   # Increase from default 20
)

# Check if error is too small
if abs(error) < 1e-14:
    print("Already at machine precision!")

# Verify anchor weights
print(f"Anchor weights: {discovery.anchor_weights}")
# Make sure relevant anchors are non-zero
```

#### Issue: "Formula diverges"

**Symptoms**: Error increases with more terms, or `nan`/`inf` values.

**Possible Causes**:

1. Slot structure doesn't match base

2. Corrections have wrong signs

3. Base is too large

4. Normalization factor incorrect

**Solutions**:

```python
# Verify structure consistency
structure = discovery.structure
print(f"Base: {structure.base}")
print(f"Slots: {structure.slots}")
print(f"Normalization: {structure.normalization}")

# Check for sign errors in corrections
for i, (int_coef, correction) in enumerate(zip(INT_COEFS, PHI_CORRECTIONS)):
    effective = int_coef + correction
    print(f"Slot {i}: {int_coef} + {correction:+.6f} = {effective:.6f}")
    if abs(effective) > abs(int_coef) * 2:
        print(f"  ⚠️  Warning: Large correction!")

# Try smaller base
discovery = RibbonDiscovery(
    concept="...",
    target_constant="pi",
    custom_structure=FormulaStructure(base=1024)  # Instead of 4096
)
```

#### Issue: "Convergence slower than expected"

**Symptoms**: `convergence_rate` is lower than claimed or desired.

**Possible Causes**:

1. Not enough terms evaluated

2. Corrections not fully optimized

3. Wrong target constant

4. Precision too low

**Solutions**:

```python
# Increase terms
result = discovery.run(max_terms=50)  # Instead of 20

# Increase precision
discovery = RibbonDiscovery(
    concept="...",
    target_constant="pi",
    precision=500  # Instead of 200
)

# Verify target
print(f"Target: {discovery.target_constant}")
print(f"Target value: {discovery.target_value}")

# Re-optimize corrections
result = discovery.run(optimize_corrections=True)
```

#### Issue: "Closed forms don't match"

**Symptoms**: `result.closed_forms` are empty or have large errors.

**Possible Causes**:

1. Corrections are numerical approximations, not exact

2. `max_complexity` too low

3. Relevant anchors not included in search

**Solutions**:

```python
# Increase complexity
searcher = ClosedFormSearcher()
forms = searcher.search(
    target_value=correction,
    anchors=['phi', 'e', 'pi'],
    max_complexity=10  # Increase from default 5
)

# Try different anchor combinations
for anchor_set in [['phi'], ['e'], ['pi'], ['phi', 'e'], ['phi', 'pi']]:
    forms = searcher.search(correction, anchor_set)
    if forms and forms[0].error < 1e-10:
        print(f"Found match with {anchor_set}: {forms[0].expression}")
        break
```

#### Issue: "Out of memory"

**Symptoms**: Process killed or `MemoryError`.

**Possible Causes**:

1. Precision too high

2. Search space too large

3. Too many terms

**Solutions**:

```python
# Reduce precision
discovery = RibbonDiscovery(
    concept="...",
    target_constant="pi",
    precision=100  # Instead of 200
)

# Reduce search space
discovery.config.max_phi_power = 8      # Instead of 10
discovery.config.max_denominator = 15   # Instead of 20

# Use iterative approach
for n_terms in [5, 10, 15, 20]:
    result = discovery.run(max_terms=n_terms)
    print(f"{n_terms} terms: error = {result.error:.2e}")
```

### Debugging Tips

#### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ribbon_math')

discovery = RibbonDiscovery(
    concept="...",
    target_constant="pi",
    verbose=True
)

result = discovery.run()
# Will print detailed progress through each layer
```

#### Inspect Intermediate Results

```python
# Run layers individually
discovery = RibbonDiscovery(concept="...", target_constant="pi")

# Layer 1: Concept parsing
weights = discovery.run_layer(1, discovery.concept)
print(f"Anchor weights: {weights}")

# Layer 2: N_smooth validity
validity = discovery.run_layer(2, weights)
print(f"Validity: {validity}")

# Layer 3: Structure generation
structure = discovery.run_layer(3, validity)
print(f"Base formula: {structure}")

# Layer 4: Error analysis
corrections = discovery.run_layer(4, structure)
print(f"Corrections: {corrections}")

# Layer 5: Verification
verification = discovery.run_layer(5, corrections)
print(f"Verification: {verification}")
```

#### Validate Inputs

```python
from ribbon_math.validate import validate_config, validate_structure

# Validate configuration
config = {...}
issues = validate_config(config)
if issues:
    print("Configuration issues:")
    for issue in issues:
        print(f"  - {issue}")

# Validate structure
structure = FormulaStructure(...)
issues = validate_structure(structure)
if issues:
    print("Structure issues:")
    for issue in issues:
        print(f"  - {issue}")
```

---

## Data Structures

### `DiscoveryResult`

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class DiscoveryResult:
    """Result from a discovery session."""
    
    # Formula representation
    formula: str                          # LaTeX or Python expression
    formula_latex: str                    # LaTeX format
    formula_python: str                   # Python code
    
    # Numerical results
    coefficients: List[float]             # Effective coefficients
    integer_coefficients: List[int]       # Base integer coefficients
    corrections: List[float]              # φ/e/π corrections
    
    # Performance metrics
    error: float                          # Absolute error
    relative_error: float                 # Relative error
    convergence_rate: float               # Digits per term
    computed_value: float                 # Actual computed value
    target_value: float                   # True value
    
    # Discovery details
    patterns_found: List['PatternMatch']  # Detected patterns
    closed_forms: List['ClosedForm']      # Symbolic expressions
    anchor_weights: Dict[str, float]      # Used anchor weights
    
    # Structure
    structure: 'FormulaStructure'         # Formula structure
    
    # Verification
    verification: Dict                    # Verification metrics
    comparison: Optional[Dict]            # Comparison with other formulas
    
    # Metadata
    discovery_time: float                 # Time taken (seconds)
    layers_executed: List[str]            # Which layers ran
    config: Dict                          # Configuration used
```

### `PatternMatch`

```python
@dataclass
class PatternMatch:
    """A detected mathematical pattern."""
    
    anchor: str                # 'phi', 'e', 'pi', etc.
    n: int                     # Numerator
    d: int                     # Denominator
    k: int                     # Power (in anchor^(-k))
    
    value: float               # Computed value: (n/d) × anchor^(-k)
    target: float              # Target value we're matching
    error: float               # |value - target|
    relative_error: float      # error / target
    quality: float             # Match quality (0-1, 1 = perfect)
    
    expression: str            # Human-readable: "(13/16) × φ^(-5)"
    
    def __str__(self):
        return f"({self.n}/{self.d}) × {self.anchor}^(-{self.k}) = {self.value:.10f}"
```

### `ClosedForm`

```python
@dataclass
class ClosedForm:
    """A closed-form expression for a value."""
    
    expression: str            # "arctan(1/φ) + log(φ)/2"
    value: float               # Numerical value
    target: float              # Target value
    error: float               # |value - target|
    complexity: int            # Expression complexity (lower = simpler)
    
    # Components
    anchors_used: List[str]    # Which constants appear
    operations: List[str]      # Which operations: +, -, ×, /, ^, log, arctan, etc.
    
    def __str__(self):
        return f"{self.expression} = {self.value:.10f} (error: {self.error:.2e})"
```

### `FormulaStructure`

```python
@dataclass
class FormulaStructure:
    """Structure of a BBP-type formula."""
    
    base: int                              # Base (e.g., 4096)
    slots: List[Tuple[int, int]]           # (period, offset) for denominators
    integer_coefficients: List[int]        # Base integer coefficients
    normalization: float                   # Final division factor
    
    # Optional
    alternating: bool = True               # Whether to use (-1)^k
    starting_k: int = 0                    # Starting index
    
    def num_slots(self) -> int:
        """Number of terms per iteration."""
        return len(self.slots)
    
    def denominator(self, slot: int, k: int) -> int:
        """Compute denominator for given slot and iteration."""
        period, offset = self.slots[slot]
        return period * k + offset
    
    def validate(self) -> List[str]:
        """Validate structure consistency."""
        issues = []
        
        if len(self.slots) != len(self.integer_coefficients):
            issues.append("Number of slots doesn't match number of coefficients")
        
        if self.base <= 1:
            issues.append(f"Base must be > 1, got {self.base}")
        
        if self.normalization <= 0:
            issues.append(f"Normalization must be > 0, got {self.normalization}")
        
        for i, (period, offset) in enumerate(self.slots):
            if period <= 0:
                issues.append(f"Slot {i}: period must be > 0, got {period}")
            if offset < 0:
                issues.append(f"Slot {i}: offset must be >= 0, got {offset}")
            if offset >= period:
                issues.append(f"Slot {i}: offset must be < period, got {offset} >= {period}")
        
        return issues
```

### `ConvergenceReport`

```python
@dataclass
class ConvergenceReport:
    """Analysis of formula convergence."""
    
    terms: List[int]              # Number of terms evaluated
    values: List[float]           # Computed values
    errors: List[float]           # Absolute errors
    relative_errors: List[float]  # Relative errors
    
    rate: float                   # Average digits per term
    rate_std: float               # Standard deviation of rate
    
    best_error: float             # Best error achieved
    best_terms: int               # Terms needed for best error
    
    # Convergence type
    convergence_type: str         # "exponential", "factorial", "power_law", etc.
    convergence_params: Dict      # Type-specific parameters
    
    def plot(self, filename: Optional[str] = None):
        """Generate convergence plot."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.terms, self.errors, 'o-')
        plt.xlabel('Number of Terms')
        plt.ylabel('Absolute Error')
        plt.title(f'Convergence: {self.rate:.2f} digits/term')
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename, dpi=150)
        else:
            plt.show()
```

### `ComparisonReport`

```python
@dataclass
class ComparisonReport:
    """Comparison of multiple formulas."""
    
    formulas: Dict[str, Dict]     # Name -> metrics
    target: float                 # Target value
    n_terms: int                  # Terms evaluated
    
    best_formula: str             # Name of best formula
    best_error: float             # Best error achieved
    
    table: str                    # Formatted comparison table
    
    def __str__(self):
        return self.table
    
    def to_markdown(self) -> str:
        """Export as markdown table."""
        lines = ["| Formula | Error | Rate | Speedup |"]
        lines.append("|---------|-------|------|---------|")
        
        baseline_rate = self.formulas[list(self.formulas.keys())[0]]['rate']
        
        for name, metrics in self.formulas.items():
            speedup = metrics['rate'] / baseline_rate
            lines.append(
                f"| {name} | {metrics['error']:.2e} | "
                f"{metrics['rate']:.2f} | {speedup:.2f}× |"
            )
        
        return "\n".join(lines)
```

---

## Natural Language Interface

### Simple Queries

```python
from ribbon_math import discover

# Simple discovery
result = discover("Find a fast formula for pi")

# With constraints
result = discover("""
Find a formula for e that:
- Converges faster than Taylor series
- Uses golden ratio patterns
- Has at most 6 terms per iteration
""")

# Comparison request
result = discover("""
Compare BBP formulas for pi:
- Standard BBP
- Bellard's formula
- A new formula using φ patterns
""")
```

### Query Parser

```python
from ribbon_math.nlp import QueryParser

parser = QueryParser()

# Parse natural language
query = "Find a fast converging BBP formula for pi using golden ratio"
parsed = parser.parse(query)

print(f"Target: {parsed.target}")              # "pi"
print(f"Constraints: {parsed.constraints}")    # ["fast converging", "BBP", "golden ratio"]
print(f"Anchor weights: {parsed.anchors}")     # {'phi': 0.8, 'pi': 1.0, ...}
print(f"Structure hints: {parsed.structure}")  # {"type": "BBP"}

# Use parsed query
discovery = RibbonDiscovery.from_query(parsed)
result = discovery.run()
```

### Supported Query Patterns

```python
QUERY_PATTERNS = {
    # Target specification
    "for pi": {"target": "pi"},
    "for e": {"target": "e"},
    "for φ": {"target": "phi"},
    "for the golden ratio": {"target": "phi"},
    
    # Speed/convergence
    "fast": {"anchors": {"phi": 0.8}},
    "fast converging": {"anchors": {"phi": 0.8}},
    "slow": {"anchors": {"phi": 0.2}},
    "exponential convergence": {"convergence_type": "exponential"},
    
    # Structure
    "BBP": {"structure_type": "bbp"},
    "BBP-type": {"structure_type": "bbp"},
    "Machin-like": {"structure_type": "machin"},
    "continued fraction": {"structure_type": "continued_fraction"},
    
    # Anchors
    "using golden ratio": {"anchors": {"phi": 0.9}},
    "using φ": {"anchors": {"phi": 0.9}},
    "using e": {"anchors": {"e": 0.9}},
    "using pi": {"anchors": {"pi": 0.9}},
    
    # Constraints
    "at most N terms": {"max_terms_per_iteration": "N"},
    "base N": {"base": "N"},
    "better than X": {"comparison_target": "X"},
}
```

---

## Validation Utilities

### Formula Validation

```python
from ribbon_math.validate import validate_formula

# Validate a discovered formula
result = discovery.run()

validation = validate_formula(
    formula=result,
    target=np.pi,
    expected_error=1e-20,
    expected_rate=3.5,
    expected_patterns=['phi']
)

print(validation.report)
```

### Validation Report

```python
@dataclass
class ValidationReport:
    """Validation results."""
    
    passed: bool                  # Overall pass/fail
    checks: Dict[str, bool]       # Individual check results
    errors: List[str]             # Error messages
    warnings: List[str]           # Warning messages
    
    report: str                   # Formatted report
    
    def __str__(self):
        return self.report

# Example output:
"""
VALIDATION REPORT
=================

✓ Error check: 7.85e-22 < 1.00e-20
✓ Convergence rate: 3.61 >= 3.50 digits/term
✓ Pattern detection: Found ['phi'] patterns
✓ Closed forms: 8/8 corrections have closed forms
✓ Mathematical identities: All verified
✗ Comparison: Not faster than Bellard (expected 20% faster)

Overall: PASSED (5/6 checks)

Warnings:
  - Convergence rate slightly below theoretical (3.61 vs 3.65)
"""
```

### Automated Testing

```python
from ribbon_math.validate import run_test_suite

# Run full test suite on a discovery
result = discovery.run()
test_results = run_test_suite(result)

print(f"Tests passed: {test_results.passed}/{test_results.total}")
print(f"Coverage: {test_results.coverage:.1%}")

if not test_results.all_passed:
    print("\nFailed tests:")
    for test in test_results.failed:
        print(f"  - {test.name}: {test.error}")
```

---

## Performance Tips

### 1\. Use Caching

```python
from ribbon_math import RibbonDiscovery

discovery = RibbonDiscovery(
    concept="...",
    target_constant="pi",
    cache_dir="./cache",      # Enable caching
    use_cache=True
)

# First run: slow (computes everything)
result1 = discovery.run()

# Second run: fast (uses cache)
result2 = discovery.run()
```

### 2\. Parallel Pattern Search

```python
from ribbon_math.layers import ErrorAnalyzer

analyzer = ErrorAnalyzer(n_jobs=4)  # Use 4 CPU cores

patterns = analyzer.detect_all_patterns(
    error,
    anchors=['phi', 'e', 'pi', 'sqrt2', 'sqrt3', 'sqrt5']
)
# Searches all anchors in parallel
```

### 3\. Incremental Discovery

```python
# Start with low precision, increase gradually
for precision in [50, 100, 200]:
    discovery = RibbonDiscovery(
        concept="...",
        target_constant="pi",
        precision=precision
    )
    result = discovery.run()
    
    if result.error < 1e-20:
        print(f"Sufficient precision: {precision}")
        break
```

### 4\. Early Stopping

```python
discovery = RibbonDiscovery(
    concept="...",
    target_constant="pi",
    early_stopping=True,
    target_error=1e-20
)

result = discovery.run()
# Stops as soon as target_error is reached
```

---

## Appendix: Mathematical Background

### BBP-Type Formulas

BBP (Bailey-Borwein-Plouffe) formulas have the form:

```
α = Σ_{k=0}^∞ 1/b^k × p(k)/q(k)
```

where:

* `b` is the base (integer > 1)

* `p(k)` is a polynomial in k

* `q(k)` is a polynomial in k

**Key property**: Can compute individual hexadecimal digits of π without computing previous digits.

**Original BBP formula** (1995):

```
π = Σ 1/16^k × [4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)]
```

### Golden Ratio Properties

The golden ratio φ = (1 + √5)/2 ≈ 1.618 has unique properties:

```
φ² = φ + 1
φ⁻¹ = φ - 1
φ² + φ⁻² = 3
φⁿ = F_n × φ + F_{n-1}  (Fibonacci)
```

These identities make φ appear naturally in many mathematical contexts.

### Error Analysis Theory

**Exponential convergence**: error \~ C × b^(-k)

* Characteristic of BBP formulas

* log(error) vs k is linear

**Golden ratio convergence**: error \~ (n/d) × φ^(-k)

* Appears in Fibonacci-related formulas

* Slower than exponential but has special structure

**Factorial convergence**: error \~ C / k!

* Characteristic of Taylor series

* Fastest common convergence type

### Pattern Detection Algorithm

```python
def detect_phi_pattern(error, max_power=10, max_denom=20):
    """
    Find (n/d) × φ^(-k) ≈ error
    
    Algorithm:
    1. For each k in 1..max_power:
        2. Compute φ^(-k)
        3. Compute ratio = error / φ^(-k)
        4. Find best rational approximation n/d to ratio
        5. Compute match_error = |error - (n/d) × φ^(-k)|
        6. If match_error < threshold, record pattern
    7. Return best pattern
    """
    best = None
    best_error = float('inf')
    
    for k in range(1, max_power + 1):
        phi_power = PHI ** (-k)
        ratio = error / phi_power
        
        # Find rational approximation using continued fractions
        n, d = best_rational_approx(ratio, max_denom)
        
        candidate_value = (n / d) * phi_power
        match_error = abs(error - candidate_value)
        
        if match_error < best_error:
            best_error = match_error
            best = PatternMatch(
                anchor='phi', n=n, d=d, k=k,
                value=candidate_value, target=error,
                error=match_error, quality=1 - match_error/error
            )
    
    return best
```

---

## Quick Reference Card

### Essential Imports

```python
from ribbon_math import RibbonDiscovery, discover
from ribbon_math.layers import ConceptParser, ErrorAnalyzer, ClosedFormSearcher
from ribbon_math.validate import validate_formula, run_test_suite
from ribbon_math.utils import compare_formulas, verify_convergence
```

### Common Workflows

```python
# 1. Simple discovery
result = discover("fast formula for pi")

# 2. Custom discovery
discovery = RibbonDiscovery(concept="...", target_constant="pi")
result = discovery.run()

# 3. Pattern detection only
analyzer = ErrorAnalyzer()
patterns = analyzer.detect_phi_patterns(error)

# 4. Compare formulas
report = compare_formulas({'BBP': bbp_func, 'φ-BBP': phi_bbp_func}, np.pi)

# 5. Validate discovery
validation = validate_formula(result, np.pi, expected_error=1e-20)
```

### Key Constants

```python
PHI = 1.6180339887498949      # Golden ratio
E = 2.718281828459045         # Euler's number
PI = 3.141592653589793        # Pi
SQRT2 = 1.4142135623730951    # √2
SQRT3 = 1.7320508075688772    # √3
SQRT5 = 2.23606797749979      # √5
```

---

## Additional Resources

### Documentation

* Full API documentation: `docs/api/`

* Tutorial notebooks: `notebooks/`

* Example discoveries: `examples/`

### Papers

* φ-BBP Formula paper: `practical_applications/ribbon_math/discoveries/phi_bbp_paper/`

* Ribbon LCM v5 framework: `practical_applications/ribbon_math/README.md`

### Community

* GitHub issues: Report bugs or request features

* Discussions: Share discoveries and ask questions

---

## Version History

* **v1.0** (2025-01): Initial release with φ-BBP discovery

* **v1.1** (planned): Natural language interface, more anchor constants

* **v2.0** (planned): Multi-constant formulas, continued fractions

---

## License

GNU General Public License v3.0 - see LICENSE for details.

---

## Citation

If you use Ribbon Math in your research, please cite:

```bibtex
@software{ribbon_math_2025,
  title={Ribbon Math: A Protocol for Mathematical Discovery via Error Analysis},
  author={Gushurst, Lesley},
  year={2025},
  url={https://github.com/lostdemeter/holographersworkbench}
}

@article{gushurst2025phibbp,
  title={The φ-BBP Formula: A New Class of Spigot Algorithms for π 
         Incorporating the Golden Ratio},
  author={Gushurst, Lesley},
  year={2025},
  url={https://github.com/lostdemeter/phi_bbp}
}
```

---

**Last Updated**: 2025-01-15\
**Maintained By**: Lesley Gushurst\
**For AI Assistants**: This guide is specifically designed to help you understand and use Ribbon Math effectively. If something is unclear or you need more examples, please let us know!
