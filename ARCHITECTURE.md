# Workbench Architecture

## Design Philosophy

The workbench follows these principles:

1. **Separation of Concerns**: Spectral, holographic, and optimization are distinct
2. **Reusable Components**: Common patterns extracted into shared utilities
3. **Consistent Interfaces**: Same API patterns across all modules
4. **Minimal Dependencies**: Each module depends only on what it needs
5. **Easy Extension**: Clear where new functionality belongs

## Module Dependencies

```
Core Pipeline:
┌─────────────┐
│   utils.py  │  (No dependencies - pure utilities)
└─────────────┘
       ↑
       │
┌─────────────┐
│ spectral.py │  (Uses utils for normalization)
└─────────────┘
       ↑
       │
┌──────────────┐
│holographic.py│  (Uses spectral for some operations)
└──────────────┘
       ↑
       │
┌──────────────┐
│optimization.py│ (Uses both spectral and holographic)
└──────────────┘

Compression Modules:
┌──────────────────┐
│fractal_peeling.py│ (Independent - numpy only)
└──────────────────┘

┌─────────────────────────┐
│holographic_compression.py│ (Independent - numpy, zlib)
└─────────────────────────┘

High-Performance Computing:
┌──────────────┐
│fast_zetas.py │ (Independent - numpy, scipy, mpmath)
└──────────────┘

┌──────────────────┐
│time_affinity.py  │ (Independent - numpy)
└──────────────────┘

Optimization Toolkit:
┌─────────────────────┐
│performance_profiler.py│ (Independent - numpy, tracemalloc)
└─────────────────────┘
       ↓
┌──────────────────────────┐
│error_pattern_visualizer.py│ (Uses scipy for FFT)
└──────────────────────────┘
       ↓
┌─────────────────────────┐
│formula_code_generator.py │ (Uses ast for validation)
└─────────────────────────┘
       ↓
┌────────────────────────┐
│convergence_analyzer.py │ (Uses scipy for curve fitting)
└────────────────────────┘
```

## Core Abstractions

### 1. Spectral Domain

**Purpose**: Transform problems into frequency space

**Key Classes**:
- `ZetaFiducials` - Manages zeta zeros (the "carrier frequencies")
- `SpectralScorer` - Scores using oscillatory patterns
- `DiracOperator` - Constructs SRT-style operators

**Pattern**:
```python
frequencies → oscillatory_sum → scores
```

### 2. Holographic Domain

**Purpose**: Extract signal from noise via interference

**Key Classes**:
- `PhaseRetrieval` - Extracts amplitude/phase
- `FourPhaseShifting` - Lossless encoding/decoding

**Pattern**:
```python
noisy_signal → phase_retrieval → envelope → refined_signal
```

### 3. Optimization Domain

**Purpose**: Reduce computational complexity

**Key Classes**:
- `SublinearOptimizer` - O(n) → O(√n) conversion
- `SRTCalibrator` - Automated parameter tuning

**Pattern**:
```python
large_set → spectral_score → holographic_refine → top_k
```

### 4. Optimization Toolkit

**Purpose**: Complete optimization pipeline from profiling to deployment

**Key Modules**:
- `performance_profiler.py` - Identify bottlenecks and measure performance
- `error_pattern_visualizer.py` - Discover correction patterns in errors
- `formula_code_generator.py` - Generate production-ready code
- `convergence_analyzer.py` - Decide when to stop optimizing

**Pattern**:
```python
profile → discover_corrections → generate_code → monitor_convergence
```

## Data Flow

### Typical Pipeline

```
Input Data
    ↓
[Spectral Scoring]
    ↓
Raw Scores (noisy)
    ↓
[Holographic Refinement]
    ↓
Refined Scores (clean)
    ↓
[Sublinear Optimization]
    ↓
Top-K Results
```

### Example Code Flow

```python
# 1. Get frequencies
zeros = ZetaFiducials.get_standard(20)

# 2. Spectral scoring
scorer = SpectralScorer(frequencies=zeros)
scores = scorer.compute_scores(candidates)

# 3. Holographic refinement
refined = holographic_refinement(scores, reference)

# 4. Optimization
optimizer = SublinearOptimizer()
top_k, stats = optimizer.optimize(candidates, lambda c: refined[c], top_k=100)
```

## Shared Patterns

### Pattern 1: Frequency-Based Scoring

Used in: `SpectralScorer`, `DiracOperator`, `compute_spectral_scores`

```python
# General form
for frequency in frequencies:
    taper = exp(-damping * frequency^2)
    phase = exp(i * frequency * log(x))
    score += taper * phase / (0.5 + i*frequency)
```

### Pattern 2: Phase Retrieval

Used in: `phase_retrieve_hilbert`, `phase_retrieve_gs`, `holographic_refinement`

```python
# General form
analytic_signal = hilbert_transform(signal)
envelope = abs(analytic_signal)
phase = angle(analytic_signal)
phase_variance = var(diff(phase))
```

### Pattern 3: Holographic Refinement

Used in: `holographic_refinement`, `SublinearOptimizer.optimize`

```python
# General form
theta, aligned = align_phase(object, reference)
envelope, phase_var = phase_retrieve(aligned)
refined = blend_ratio * aligned * envelope + (1-blend_ratio) * reference
if phase_var > threshold:
    refined *= damping_factor
```

### Pattern 4: Normalization

Used everywhere via `utils.normalize_signal`

```python
# General form
normalized = (signal - min) / (max - min + epsilon)
# or
normalized = signal / (max(abs(signal)) + epsilon)
```

## Extension Points

### Adding New Spectral Methods

Add to `spectral.py`:

```python
class MySpectralMethod:
    def __init__(self, frequencies):
        self.frequencies = frequencies
    
    def compute_scores(self, candidates):
        # Your implementation
        return scores
```

### Adding New Phase Retrieval Methods

Add to `holographic.py`:

```python
def phase_retrieve_mymethod(signal, **kwargs):
    # Your implementation
    envelope = ...
    phase_variance = ...
    return envelope, phase_variance
```

Then update `PhaseRetrieval` class to support it.

### Adding New Optimization Strategies

Add to `optimization.py`:

```python
class MyOptimizer:
    def optimize(self, candidates, score_fn, top_k):
        # Your implementation
        return top_candidates, stats
```

## Testing Strategy

### Unit Tests

Each module should have tests:

```python
# test_spectral.py
def test_zeta_fiducials():
    zeros = ZetaFiducials.get_standard(20)
    assert len(zeros) == 20
    assert zeros[0] > 14.0  # First zero ~14.13

# test_holographic.py
def test_phase_retrieval():
    signal = np.sin(np.linspace(0, 10, 100))
    env, pv = phase_retrieve_hilbert(signal)
    assert len(env) == len(signal)
    assert pv >= 0

# test_optimization.py
def test_sublinear_optimizer():
    candidates = np.arange(1000)
    optimizer = SublinearOptimizer()
    top, stats = optimizer.optimize(candidates, lambda c: c, top_k=10)
    assert len(top) == 10
```

### Integration Tests

Test complete workflows:

```python
def test_complete_workflow():
    # Full pipeline
    zeros = ZetaFiducials.get_standard(20)
    scorer = SpectralScorer(frequencies=zeros)
    scores = scorer.compute_scores(candidates)
    refined = holographic_refinement(scores, reference)
    optimizer = SublinearOptimizer()
    top_k, stats = optimizer.optimize(candidates, lambda c: refined[c], top_k=100)
    
    assert len(top_k) == 100
    assert stats.reduction_ratio < 0.1  # 90%+ reduction
```

## Performance Considerations

### Caching

- `ZetaFiducials` caches computed zeros
- `@lru_cache` on expensive functions
- Reuse scorer/optimizer instances

### Memory

- Use `np.float32` instead of `float64` when precision allows
- Process in chunks for large datasets
- Delete intermediate results when done

### Speed

- Vectorize operations with NumPy
- Use FFT for convolutions
- Avoid Python loops over large arrays

## Common Pitfalls

### 1. Division by Zero

Always add epsilon:
```python
# Bad
score = 1 / x

# Good
score = 1 / (x + 1e-12)
```

### 2. Phase Wrapping

Use `arctan2` for phase differences:
```python
# Bad
phase_diff = phase[1:] - phase[:-1]

# Good
phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
```

### 3. Normalization

Check for zero range:
```python
# Bad
normalized = (x - min(x)) / (max(x) - min(x))

# Good
normalized = (x - min(x)) / (max(x) - min(x) + 1e-12)
```

### 4. Array Shapes

Always validate:
```python
# Good practice
obj = np.asarray(object_signal, dtype=float).ravel()
ref = np.asarray(reference_signal, dtype=float).ravel()
n = min(len(obj), len(ref))
obj = obj[:n]
ref = ref[:n]
```

## Complete Module List

### Core Modules (v0.1.0)
1. **spectral.py** - Frequency-domain analysis and zeta-based scoring
2. **holographic.py** - Phase retrieval and interference patterns
3. **optimization.py** - Sublinear algorithms and parameter calibration
4. **fractal_peeling.py** - Recursive lossless compression
5. **holographic_compression.py** - Image compression via harmonics
6. **fast_zetas.py** - High-performance zeta zero computation (26× faster)
7. **time_affinity.py** - Walltime-based parameter discovery
8. **performance_profiler.py** - Performance profiling and bottleneck detection
9. **error_pattern_visualizer.py** - Error pattern discovery and correction
10. **formula_code_generator.py** - Production code generation
11. **convergence_analyzer.py** - Convergence analysis and stopping decisions
12. **utils.py** - Common utilities and helper functions

### Demos
14 interactive Jupyter notebooks demonstrating all features

### Tests
Comprehensive test suite with 9/9 passing tests

## Future Enhancements

### Short Term

- [x] Performance profiling tools (completed)
- [x] Error pattern discovery (completed)
- [x] Code generation (completed)
- [x] Convergence analysis (completed)
- [ ] GPU acceleration for FFT operations
- [ ] Parallel batch processing

### Medium Term

- [ ] Integration with ML frameworks
- [ ] Real-time convergence monitoring
- [ ] Automated test generation
- [ ] Interactive dashboards

### Long Term

- [ ] Quantum-inspired methods
- [ ] Distributed computing support
- [ ] Hardware acceleration
- [ ] Cloud deployment tools

## Versioning

Current: **v0.1.0**

Semantic versioning:
- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes

## Contributing Guidelines

1. **Before adding code**: Check if similar functionality exists
2. **Choose the right module**: Spectral, holographic, or optimization?
3. **Follow patterns**: Use existing code as template
4. **Add tests**: Unit tests for new functions
5. **Update docs**: Add examples to README
6. **Update exports**: Add to `__init__.py`

## Questions?

See:
- `README.md` - Usage guide
- `examples.py` - Working examples
- `WORKBENCH_SUMMARY.md` - High-level overview
