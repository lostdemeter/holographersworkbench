# Clock Resonance Compiler

**Automatic upgrade** of any processor to use clock eigenphases!

## Overview

The Clock Resonance Compiler analyzes existing code for random/comb sources and automatically compiles it to a clock-resonant version. This provides deterministic behavior while maintaining (or improving) optimization quality.

## Key Features

- **Automatic Analysis**: Identifies random sources in existing code
- **Drop-in Replacement**: Compiled version has same API
- **Deterministic**: Same input → same output (always)
- **No Seed Management**: Clock phases replace random seeds

## Quick Start

```python
from workbench.core import ClockResonanceCompiler, make_clock_resonant

# Analyze a processor
compiler = ClockResonanceCompiler()
analysis = compiler.analyze(MyProcessor)
print(analysis)

# Compile to clock-resonant version
ClockMyProcessor = compiler.compile(MyProcessor)

# Or use convenience function
ClockMyProcessor = make_clock_resonant(MyProcessor)

# Use the compiled processor
processor = ClockMyProcessor()
result = processor.process(data)  # Deterministic!
```

## ClockOracleMixin

Create clock-native processors directly:

```python
from workbench.core import ClockOracleMixin

class MyClockProcessor(ClockOracleMixin):
    def process(self, data):
        # Uses clock eigenphases instead of random
        phase = self.get_clock_phase()
        noise = self.clock_randn(len(data))  # Deterministic "random"
        return data + noise * phase
```

### Mixin Methods

| Method | Replaces | Description |
|--------|----------|-------------|
| `get_clock_phase(n)` | - | Get n-th clock phase |
| `clock_random(size)` | `np.random.random()` | Uniform [0,1) |
| `clock_randn(size)` | `np.random.randn()` | Normal distribution |
| `clock_choice(a, size)` | `np.random.choice()` | Random selection |

## Analyzed Processors

| Processor | Difficulty | Random Sources |
|-----------|------------|----------------|
| SpectralScorer | none | 0 |
| AdaptiveNonlocalityOptimizer | easy | 2 |
| QuantumFolder | medium | 5 |
| ChaosSeeder | none | 0 |

## TSPLIB Benchmark Results

Using v2 optimizer with clock resonance:

| Instance | Optimal | Original Gap | v2 Gap | Improvement |
|----------|---------|--------------|--------|-------------|
| eil51 | 426 | 35.25% | 6.12% | 82.6% |
| berlin52 | 7542 | 38.48% | 5.53% | 85.6% |
| st70 | 675 | 33.24% | 5.65% | 83.0% |
| eil76 | 538 | 36.54% | 8.03% | 78.0% |
| kroA100 | 21282 | 42.66% | 2.76% | 93.5% |
| **Average** | | 37.23% | 5.62% | **84.9%** |

## Run Demo

```bash
python demo_clock_compiler.py
```

## How It Works

1. **Analysis**: AST inspection finds `np.random.*` calls
2. **Classification**: Determines difficulty (none/easy/medium/hard)
3. **Compilation**: Replaces random calls with clock phase equivalents
4. **Validation**: Verifies compiled version produces correct results

## Files

- `demo_clock_compiler.py` - Interactive demo

## Related

- `workbench/core/clock_compiler.py` - Main implementation
- `workbench/processors/sublinear_clock_v2.py` - Clock-resonant optimizer
