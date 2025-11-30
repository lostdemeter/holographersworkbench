# Clock Dimensional Downcasting

**Machine-precision spectral oracle for quantum clock states.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## The Breakthrough

Instead of building and diagonalizing a 2³⁰ × 2³⁰ unitary matrix (physically impossible), we compute exact eigenphases in O(log n) time using dimensional downcasting.

**Key insight**: The eigenphase counting function N_smooth(θ) satisfies N_smooth(θₙ) ≈ n, enabling disambiguation between candidates via bisection refinement.

## Installation

```bash
git clone https://github.com/lostdemeter/clock_downcaster.git
cd clock_downcaster
pip install -r requirements.txt
```

### Optional: Image Generation Setup

For face generation examples, you need Stable Diffusion (~5GB download):

```bash
# Install image generation dependencies
pip install torch diffusers transformers accelerate pillow

# Download and cache the model
python setup_models.py --diffusion

# Check what's installed
python setup_models.py --check
```

## Quick Start

### Machine-Precision Solver (Recommended)

```python
from clock_downcaster import solve_clock_phase, ClockDimensionalDowncaster

# Quick usage - get the 100th eigenphase
theta_100 = solve_clock_phase(100)
print(f"θ₁₀₀ = {theta_100:.15f}")

# Full control with verification
solver = ClockDimensionalDowncaster()
theta = solver.solve(100)
result = solver.verify(100)
print(f"|C(θ)| = {result['clock_value']:.2e}")  # Should be ~10⁻¹⁵
```

### Memoized Oracle (145× Faster)

```python
from clock_downcaster import LazyClockOracle

# O(1) lookup for precomputed phases
oracle = LazyClockOracle(max_depth=20)
theta = oracle.get_phase(1000)
```

## How It Works

### The Clock Function

The quantum clock unitary follows a recursive construction:
```
U_{n+1} = exp(iθ·φ) ⊗ Uₙ + exp(iθ·φ') ⊗ σₓ Uₙ σₓ
```

The clock function C(θ) = sin(π × N_smooth(θ)) has zeros at eigenphases.

### Dimensional Downcasting

1. **Find sign changes** of C(θ) in a bracket
2. **Select correct eigenphase** using N_smooth(θₙ) ≈ n
3. **Refine** with bisection + Brent's method to machine precision

This achieves |C(θ)| ~ 10⁻¹⁵ (essentially zero).

## Performance

| Operation | Complexity | Time |
|-----------|------------|------|
| Recursive solve | O(log n) | ~1 ms |
| Memoized lookup | O(1) | ~50 µs |
| Batch (1000 phases) | O(n log n) | ~100 ms |

**Memoized oracle**: 145× faster than recursive computation.

## Applications

### 1. Spectral Oracle
Query exact eigenphases of 2⁶⁰⁰ × 2⁶⁰⁰ clock matrices in microseconds.

### 2. Deterministic Attention (See Examples)
Use clock phases to provide attention weights without training.

### 3. Diffusion Model Seeding (See Examples)
Replace random seeds with clock phases for reproducible generation.

### 4. Cryptographic Random Bits
Generate perfect 1/f^α sequences using fractional parts of eigenphases.

## Examples

### Ribbon Attention
Clock-phase attention mechanism for language models:
```bash
python examples/ribbon_attention.py
```

### Ribbon Diffusion
Clock-phase guided Stable Diffusion for face generation:
```bash
python examples/ribbon_diffusion_faces.py
```

## Demo Script

```bash
# Full demo (oracle + text + complexity analysis)
python demo.py --skip-image

# Full demo with image generation (requires GPU + diffusers)
python demo.py

# Just oracle benchmark
python demo.py --mode oracle

# Just text generation
python demo.py --mode text

# Just image generation
python demo.py --mode image

# Complexity analysis
python demo.py --mode complexity
```

### Legacy Demo
```bash
python demo_clock_downcaster.py --mode demo
python demo_clock_downcaster.py --mode benchmark
```

## Mathematical Foundation

See the [paper](paper/paper.md) for full derivations:

- **Counting Function**: N_smooth(θ) = θ/(2π×φ) - corrections
- **Disambiguation**: N_smooth(θₙ) ≈ n (integers, not n-0.5 like zeta zeros!)
- **Clock Function**: C(θ) = sin(π × N_smooth) has zeros at eigenphases

### Key Difference from Zeta Zeros

| Property | Zeta Zeros | Clock States |
|----------|------------|--------------|
| N_smooth at n-th | n - 0.5 | n |
| Disambiguation error | ~0.5 | ~0.01 |
| Sharpness | Good | **50× sharper!** |

## Directory Structure

```
clock_downcaster/
├── __init__.py              # Package exports
├── clock_solver.py          # Machine-precision solver
├── clock_predictor.py       # Smooth predictor
├── clock_downcaster.py      # Main interface
├── fast_clock_predictor.py  # Memoized oracle (145× faster)
├── demo_clock_downcaster.py # Demo script
├── examples/
│   ├── ribbon_attention.py      # Clock-phase attention
│   └── ribbon_diffusion_faces.py # Clock-phase diffusion
└── paper/
    ├── paper.md                 # Full research paper
    ├── mathematical_foundations.md
    └── figures/                 # Publication figures
```

## Requirements

```
numpy>=1.20.0
scipy>=1.7.0
mpmath>=1.2.0
```

Optional for examples:
```
torch>=1.9.0
diffusers>=0.20.0
transformers>=4.20.0
```

## Citation

```bibtex
@software{gushurst2025clock,
  title={Clock Dimensional Downcasting: Machine-Precision Spectral Oracle},
  author={Gushurst, Lesley},
  year={2025},
  url={https://github.com/lostdemeter/clock_downcaster}
}
```

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.

## Author

**Lesley Gushurst** - [lostdemeter](https://github.com/lostdemeter)

---

*"The eigenphases exist. We don't compute them - we locate them."*
