# Ribbon LCM v5: Discovery Engine

A framework for automated mathematical discovery using the Ribbon LCM conceptual language.

## Core Idea

Mathematical truth has a geometry. The Ribbon LCM provides coordinates (6 anchor constants) 
for navigating this space. Discovery is locating truths, not inventing them.

## Major Achievement

**φ-BBP Formula Discovery**: A new class of BBP-type formulas for π:
- **Error**: 7.85×10⁻²² (machine precision)
- **Convergence**: 3.61 digits/term
- **20% faster than Bellard's formula**

## Key Innovations

### 1. Error-as-Signal Paradigm (Validated)
The "error" in approximate solutions contains mathematical structure:
- Deviations follow φ^(-k) patterns
- Total corrections have closed forms involving arctan(1/φ) and log(φ)
- Base changes introduce hidden structure (4096 = 1024 × (φ² + φ⁻² + 1))

### 2. Pattern Detection
- **PhiPatternDetector**: Find (n/d) × φ^k approximations
- **ClosedFormSearcher**: Find arctan(1/φ) + log(φ) combinations
- **Fibonacci/Lucas patterns**: Detect F_n/F_m × φ^k structures

### 3. Five-Layer Architecture
1. **Concept Layer**: Natural language → anchor weights
2. **N_smooth Layer**: Continuous validity measure
3. **Structure Layer**: Valid transformations + LCM pruning
4. **Error Analysis Layer**: Pattern detection in deviations
5. **Verification Layer**: Experimental validation

## Quick Start

```python
from ribbon_lcm_v5 import DiscoveryEngine
from ribbon_lcm_v5.domains import BBPDomain

# Create domain and engine
domain = BBPDomain()
engine = DiscoveryEngine(domain)

# Verify the φ-BBP formula
result = domain.verify_phi_bbp()
print(f"Error: {result['verification']['error']:.2e}")
print(f"Rate: {result['benchmark']['rate']:.2f} digits/term")
print(f"vs Bellard: {result['benchmark']['vs_bellard']}")

# Run discovery
concept = domain.concept.parse_concept("fast converging BBP formula for pi")
discoveries = engine.discover(concept, target=3.14159265358979)
```

## The φ-BBP Formula

```
π = (1/64) × Σ (-1)^k/4096^k × [
    (256 + c₀)/(4k+1) + (-32 + c₁)/(4k+3) + 
    (4 + c₂)/(12k+1) + (1 + c₃)/(12k+3) +
    (-128 + c₄)/(12k+5) + (-64 + c₅)/(12k+7) +
    (-128 + c₆)/(12k+9) + (4 + c₇)/(12k+11)
]
```

Where each correction cᵢ = (nᵢ/dᵢ) × φ^(-kᵢ):

| Slot | Integer | Correction | Pattern |
|------|---------|------------|---------|
| 4 | -128 | +0.0733 | **(13/16) × φ^(-5)** ← cleanest |

Total correction closed form:
```
Total ≈ (13/20)×arctan(1/φ) + (-26/25)×log(φ)
```

## Key Mathematical Identities

1. **φ² + φ⁻² = 3** → 4 = φ² + φ⁻² + 1
2. **arctan(1/φ) + arctan(1/φ³) = π/4** (Fibonacci arctan)
3. **Li₁(1/φ) = 2×log(φ)** (Polylogarithm)
4. **Li₂(1/φ²) = π²/15 - log²(φ)** (Polylog-π connection)

## Directory Structure

```
ribbon_lcm_v5/
├── discovery_engine.py    # Core 5-layer framework
├── fast_search.py         # Fast pattern matching & PSLQ
├── quickstart.py          # Demo script
├── domains/
│   ├── bbp_domain.py      # BBP formula domain
│   └── quadratic_field_domain.py  # Real quadratic fields
├── discoveries/
│   ├── phi_bbp_paper/     # φ-BBP research paper
│   ├── DISCOVERIES_Q29.md # ℚ(√29) search results
│   └── SEARCH_RESULTS.md  # Summary of searches
├── tests/                 # Test suite
└── temp/                  # Temporary search scripts
```

## Evolution from v4

| Version | Key Innovation |
|---------|---------------|
| v3 | N_smooth continuous feedback |
| v4 | Error-as-Signal hypothesis |
| **v5** | **Validated φ-BBP discovery** |

## Citation

```bibtex
@article{tabor2024phibbp,
  title={The φ-BBP Formula: A New Class of Spigot Algorithms for π},
  author={Tabor, Thorin},
  year={2024}
}
```

## License

GPL-3.0-or-later
