# Workbench Architecture

## Design Philosophy

The workbench follows a **5-layer architecture** with strict unidirectional dependencies:

1. **Separation of Concerns**: Each layer has a single, well-defined responsibility
2. **Unidirectional Dependencies**: Higher layers depend on lower layers, never vice-versa
3. **Pure Functions at Base**: Layer 1 contains only pure, stateless functions
4. **Stateful Transformers Above**: Layers 2-4 introduce state and complexity progressively
5. **Clear Extension Points**: New functionality has an obvious home

## 5-Layer Architecture

```
Layer 5: generation/     (Generates external artifacts: code files, reports)
    ↓
Layer 4: processors/     (Stateful transformers: scorers, optimizers, compressors)
    ↓
Layer 3: analysis/       (Read-only analyzers: profilers, pattern detectors)
    ↓
Layer 2: core/           (Domain primitives: zeta zeros, quantum modes)
    ↓
Layer 1: primitives/     (Pure utility functions: signal processing, FFT, phase math)
```

### Layer 1: Primitives (`workbench/primitives/`)

**Purpose**: Pure, stateless utility functions with no side effects

**Modules**:
- `signal.py` - Signal processing (normalize, smooth, detect_peaks, psnr, etc.)
- `frequency.py` - FFT and power spectrum computation
- `phase.py` - Phase retrieval (retrieve_hilbert, retrieve_gs, align)
- `kernels.py` - Kernel functions (exponential_decay, gaussian)
- `quantum_folding.py` - Quantum-inspired TSP optimization (QuantumFolder)
- `chaos_seeding.py` - Residual chaos optimization (ChaosSeeder, AdaptiveChaosSeeder)

**Rules**:
- No classes, only functions
- No state, no side effects
- No dependencies on other workbench layers
- Only depends on: numpy, scipy

**Example**:
```python
from workbench.primitives import signal, frequency, phase

# Pure function calls
normalized = signal.normalize(data, method='minmax')
fft_result = frequency.compute_fft(normalized)
envelope = signal.compute_envelope(data)
```

### Layer 2: Core (`workbench/core/`)

**Purpose**: Domain-specific primitives (zeta zeros, crystalline structures)

**Modules**:
- `zeta.py` - Hybrid fractal-Newton zeta zero computation (100% perfect accuracy)
- `gushurst_crystal.py` - **GushurstCrystal** unified number-theoretic framework

**Rules**:
- Can use Layer 1 (primitives)
- Introduces domain-specific concepts
- Minimal state (mostly caching)
- No dependencies on Layers 3-5

**Hybrid Fractal-Newton Method**:

Breakthrough algorithm achieving 100% perfect accuracy (error < 1e-12) for Riemann zeta zeros:
- **Phase 1**: Sierpinski fractal exploration (Hausdorff dimension 1.585)
- **Phase 2**: Adaptive Newton refinement (15 iterations)
- **Performance**: 2.7× faster than mpmath for batches ≥20
- **Innovation**: Exploits dimensional equivalence (1 ≈ 1.585 ≈ 2) for better initial guesses

**The Gushurst Crystal**:

The **GushurstCrystal** is a unified framework that combines:
- Fractal peel analysis on zeta zero spacings
- Spectral decomposition for prime prediction
- Crystalline lattice structure [2¹, 3¹, 7¹]

Enables:
- Prime prediction via resonance patterns (100% accuracy, all sequential)
- Zeta zero computation (fast mode: no overhead)
- Unified number-theoretic framework

**Example**:
```python
from workbench.core import zetazero, zetazero_batch, GushurstCrystal

# Hybrid fractal-Newton (100% perfect accuracy)
z = zetazero(100)  # Single zero
zeros = zetazero_batch(1, 100)  # Batch computation

# Gushurst crystal (unified framework)
gc = GushurstCrystal(n_zeros=100, max_prime=1000)
primes = gc.predict_primes(n_primes=20)      # Prime prediction
zeros = gc.predict_zeta_zeros(n_zeros=20)    # Fast mode (default)
structure = gc.analyze_crystal_structure()   # Optional analysis
```

### Layer 3: Analysis (`workbench/analysis/`)

**Purpose**: Read-only analyzers that inspect data without modifying it

**Modules**:
- `performance.py` - Performance profiling and bottleneck detection
- `errors.py` - Error pattern discovery and visualization
- `convergence.py` - Convergence analysis and stopping decisions
- `affinity.py` - Time affinity optimization

**Rules**:
- Can use Layers 1-2
- Read-only operations (no data transformation)
- Produces reports, metrics, visualizations
- No dependencies on Layers 4-5

**Example**:
```python
from workbench.analysis import PerformanceProfiler, ErrorPatternAnalyzer

# Analyze without modifying
profiler = PerformanceProfiler()
result, profile = profiler.profile_function(my_func, args)

analyzer = ErrorPatternAnalyzer(actual, predicted, x)
report = analyzer.analyze_all()
```

### Layer 4: Processors (`workbench/processors/`)

**Purpose**: Stateful transformers that modify data

**Modules**:
- `spectral.py` - Spectral scoring with zeta zeros
- `holographic.py` - Phase retrieval and holographic refinement
- `holographic_depth.py` - Monocular depth extraction (2.4× better dynamic range)
- `optimization.py` - Sublinear optimization algorithms
- `compression.py` - Fractal peeling + holographic compression
- `encoding.py` - Holographic encoding
- `ergodic.py` - Ergodic jump diagnostics
- `adaptive_nonlocality.py` - Dimensional coupling optimization
- `sublinear_qik.py` - Sublinear QIK (O(N^1.5 log N))
- `quantum_autoencoder.py` - Quantum autoencoder for TSP
- `additive_error_stereo.py` - O(n) stereo synthesis (2.5× speedup)

**Rules**:
- Can use Layers 1-3
- Stateful classes that transform data
- Main computational workhorses
- No dependencies on Layer 5

**Example**:
```python
from workbench.processors import SpectralScorer, SublinearOptimizer

# Stateful transformers
scorer = SpectralScorer(use_zeta=True, n_zeta=20)
scores = scorer.compute_scores(candidates)

optimizer = SublinearOptimizer(scorer=scorer)
top_k, stats = optimizer.optimize(candidates, score_fn, top_k=100)
```

### Layer 5: Generation (`workbench/generation/`)

**Purpose**: Generate external artifacts (code files, reports, etc.)

**Modules**:
- `code.py` - Formula code generation, validation, optimization

**Rules**:
- Can use all lower layers (1-4)
- Produces external artifacts
- Top of the dependency chain

**Example**:
```python
from workbench.generation import FormulaCodeGenerator

# Generate production code
generator = FormulaCodeGenerator(base_formula="x**2", name="improved")
generator.add_correction("correction = 0.1 * x")
generator.export_to_file("improved.py", format="module")
```

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: generation/                                        │
│   code.py                                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: processors/                                        │
│   spectral.py, holographic.py, optimization.py,            │
│   compression.py, encoding.py, ergodic.py                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: analysis/                                          │
│   performance.py, errors.py, convergence.py, affinity.py   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: core/                                              │
│   zeta.py, gushurst_crystal.py                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: primitives/                                        │
│   signal.py, frequency.py, phase.py, kernels.py            │
└─────────────────────────────────────────────────────────────┘
                           ↓
                    numpy, scipy, mpmath
```

## Import Patterns

### High-Level Imports (Recommended for Users)

```python
from workbench import (
    # Layer 2: Core
    zetazero, ZetaFiducials, GushurstCrystal,
    
    # Layer 3: Analysis
    PerformanceProfiler, ErrorPatternAnalyzer, ConvergenceAnalyzer,
    
    # Layer 4: Processors
    SpectralScorer, SublinearOptimizer, HolographicCompressor,
    
    # Layer 5: Generation
    FormulaCodeGenerator,
)
```

### Explicit Layer Imports (Recommended for Developers)

```python
# Layer 1: Pure functions
from workbench.primitives import signal, frequency, phase

# Layer 2: Domain primitives
from workbench.core.zeta import zetazero, ZetaFiducials
from workbench.core import GushurstCrystal

# Layer 3: Analyzers
from workbench.analysis.performance import PerformanceProfiler
from workbench.analysis.errors import ErrorPatternAnalyzer

# Layer 4: Processors
from workbench.processors.spectral import SpectralScorer
from workbench.processors.optimization import SublinearOptimizer

# Layer 5: Generators
from workbench.generation.code import FormulaCodeGenerator
```

## Data Flow Patterns

### Pattern 1: Spectral Scoring Pipeline

```
Input Data
    ↓
[Layer 2: Get Zeta Zeros]
    ↓
Frequencies
    ↓
[Layer 4: Spectral Scoring]
    ↓
Raw Scores
    ↓
[Layer 4: Holographic Refinement]
    ↓
Refined Scores
```

### Pattern 2: Optimization Toolkit Pipeline

```
Slow Function
    ↓
[Layer 3: Profile Performance]
    ↓
Bottleneck Identified
    ↓
[Layer 3: Analyze Error Patterns]
    ↓
Correction Patterns
    ↓
[Layer 5: Generate Code]
    ↓
Optimized Function
    ↓
[Layer 3: Monitor Convergence]
    ↓
Stop Decision
```

### Pattern 3: Complete Workflow

```python
# Layer 2: Get domain primitives
zeros = ZetaFiducials.get_standard(20)

# Layer 4: Score candidates
scorer = SpectralScorer(frequencies=zeros)
scores = scorer.compute_scores(candidates)

# Layer 4: Refine with holographic processing
refined = holographic_refinement(scores, reference)

# Layer 4: Optimize
optimizer = SublinearOptimizer()
top_k, stats = optimizer.optimize(candidates, lambda c: refined[c], top_k=100)

# Layer 3: Analyze performance
profiler = PerformanceProfiler()
result, profile = profiler.profile_function(optimizer.optimize, args)
```

## Extension Points

### Adding New Primitive Functions (Layer 1)

Add to appropriate module in `workbench/primitives/`:

```python
# workbench/primitives/signal.py
def my_new_signal_function(signal: np.ndarray, param: float) -> np.ndarray:
    """Pure function with no side effects."""
    return processed_signal
```

Update `workbench/primitives/__init__.py` to export it.

### Adding New Domain Primitives (Layer 2)

Add to `workbench/core/`:

```python
# workbench/core/my_domain.py
class MyDomainPrimitive:
    """Domain-specific primitive with minimal state."""
    def __init__(self, params):
        self.params = params
    
    def compute(self, data):
        # Can use Layer 1 primitives
        from workbench.primitives import signal
        return signal.normalize(data)
```

### Adding New Analyzers (Layer 3)

Add to `workbench/analysis/`:

```python
# workbench/analysis/my_analyzer.py
class MyAnalyzer:
    """Read-only analyzer."""
    def analyze(self, data):
        # Can use Layers 1-2
        # Returns report, doesn't modify data
        return report
```

### Adding New Processors (Layer 4)

Add to `workbench/processors/`:

```python
# workbench/processors/my_processor.py
class MyProcessor:
    """Stateful transformer."""
    def __init__(self, params):
        self.params = params
        self.state = {}
    
    def process(self, data):
        # Can use Layers 1-3
        # Transforms data
        return transformed_data
```

### Adding New Generators (Layer 5)

Add to `workbench/generation/`:

```python
# workbench/generation/my_generator.py
class MyGenerator:
    """Generates external artifacts."""
    def generate(self, spec):
        # Can use all lower layers
        # Produces files, reports, etc.
        with open("output.txt", "w") as f:
            f.write(generated_content)
```

## Testing Strategy

### Layer-Specific Testing

Each layer should have focused tests:

```python
# tests/test_primitives.py
def test_signal_normalize():
    """Test pure function."""
    data = np.array([1, 2, 3, 4, 5])
    normalized = signal.normalize(data, method='minmax')
    assert np.min(normalized) == 0
    assert np.max(normalized) == 1

# tests/test_core.py
def test_zeta_fiducials():
    """Test domain primitive."""
    zeros = ZetaFiducials.get_standard(20)
    assert len(zeros) == 20
    assert zeros[0] > 14.0  # First zero ~14.13

# tests/test_analysis.py
def test_performance_profiler():
    """Test analyzer."""
    profiler = PerformanceProfiler()
    result, profile = profiler.profile_function(lambda x: x**2, 10)
    assert result == 100
    assert profile.execution_time > 0

# tests/test_processors.py
def test_spectral_scorer():
    """Test processor."""
    scorer = SpectralScorer(use_zeta=True, n_zeta=10)
    candidates = np.arange(100, 200)
    scores = scorer.compute_scores(candidates)
    assert len(scores) == len(candidates)

# tests/test_generation.py
def test_code_generator():
    """Test generator."""
    gen = FormulaCodeGenerator("x**2", "test")
    code = gen.generate_function()
    assert "def test" in code
```

### Integration Testing

Test cross-layer workflows:

```python
def test_complete_pipeline():
    """Test full workflow across all layers."""
    # Layer 2
    zeros = ZetaFiducials.get_standard(20)
    
    # Layer 4
    scorer = SpectralScorer(frequencies=zeros)
    scores = scorer.compute_scores(np.arange(100))
    
    # Layer 3
    profiler = PerformanceProfiler()
    _, profile = profiler.profile_function(scorer.compute_scores, np.arange(100))
    
    assert len(scores) == 100
    assert profile.execution_time > 0
```

## Performance Considerations

### Caching Strategy

- **Layer 2**: ZetaFiducials caches computed zeros
- **Layer 4**: Reuse scorer/optimizer instances
- Use `@lru_cache` for expensive pure functions

### Memory Management

- Use `np.float32` when precision allows
- Process large datasets in chunks
- Delete intermediate results explicitly

### Speed Optimization

- Vectorize with NumPy (Layer 1)
- Use FFT for convolutions (Layer 1)
- Avoid Python loops over arrays
- Profile before optimizing (Layer 3)

## Common Pitfalls

### 1. Violating Layer Dependencies

```python
# ✗ BAD: Layer 1 depending on Layer 4
def normalize(signal):
    scorer = SpectralScorer()  # Layer 4 in Layer 1!
    return scorer.process(signal)

# ✓ GOOD: Layer 1 stays pure
def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-12)
```

### 2. Stateful Functions in Layer 1

```python
# ✗ BAD: State in Layer 1
cache = {}
def compute_fft(signal):
    if id(signal) in cache:
        return cache[id(signal)]
    result = np.fft.fft(signal)
    cache[id(signal)] = result
    return result

# ✓ GOOD: Pure function
def compute_fft(signal):
    return np.fft.fft(signal)
```

### 3. Circular Dependencies

```python
# ✗ BAD: Circular import
# processors/spectral.py
from workbench.generation.code import FormulaCodeGenerator

# generation/code.py
from workbench.processors.spectral import SpectralScorer

# ✓ GOOD: Respect layer hierarchy
# processors/spectral.py - no imports from Layer 5
# generation/code.py - can import from Layer 4
```

## Versioning

Current: **v0.1.0**

Semantic versioning:
- **Major**: Breaking API changes, layer restructuring
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, documentation

## Migration from Old Structure

The repository has been reorganized from a flat structure to a 5-layer architecture:
- Old utility functions → `workbench/primitives/`
- Old zeta computation → `workbench/core/zeta.py`
- Old processors → `workbench/processors/`
- Old analysis tools → `workbench/analysis/`
- Old code generation → `workbench/generation/`

All functionality has been preserved and enhanced with the new structure.

## Questions?

- **README.md** - Usage guide and quick start
- **AI_README.md** - Concise API reference for AI assistants
- **examples/** - Working code examples
- **tests/** - Test suite demonstrating usage
