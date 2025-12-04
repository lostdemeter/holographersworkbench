# Dimensional Navigation: Quick Start Guide

## Installation

No installation required - just copy the `dimensional_navigation` directory.

Requirements: `numpy` only.

## 30-Second Example

```python
from dimensional_navigation import PhiLens

# Compress any array
lens = PhiLens()
encoded = lens.encode(my_data)
print(f"Compression: {encoded.compression_ratio:.1f}x")

# Reconstruct
reconstructed = lens.decode(encoded)
```

## When to Use What

| Method | Best For | Compression | Accuracy |
|--------|----------|-------------|----------|
| `PhiLens` | Neural network weights, general data | ~3x | High |
| `VectorMesh` | Data with known clustering points | Variable | Exact at anchors |
| `HierarchicalMesh` | When you need arbitrary precision | Variable | Configurable |

## Key Concepts

### The φ-Level

Every value can be written as:
```
value = φ^(-N_smooth) × sign × scale
```

Where `N_smooth` is the "φ-level" - a single number that encodes the magnitude.

### The Mesh

A precomputed lookup table of φ-values:
```python
mesh[index] = φ^(-N_smooth)
```

With 512 entries (2KB), this fits in L1 cache, making lookups essentially free.

### Encoding

```
value → N_smooth → (n_int, n_frac) → index
```

- `n_int`: 5 bits (0-31) - integer part
- `n_frac`: 4 bits (0-15) - fractional part  
- `sign`: 1 bit
- **Total: 10 bits per value** (vs 32 for float)

### Decoding

```
index → mesh[index] → value
```

Just a lookup and multiply - no exponentials at runtime.

## Examples

### Compress a Matrix

```python
from dimensional_navigation import PhiLens, measure_error

A = np.random.randn(100, 100) * 0.1

lens = PhiLens()
encoded = lens.encode(A)
reconstructed = lens.decode(encoded)

errors = measure_error(A, reconstructed)
print(f"Max error: {errors['max_error']:.6f}")
print(f"Compression: {encoded.compression_ratio:.1f}x")
```

### Analyze Clustering

```python
from dimensional_navigation import VectorMesh

mesh = VectorMesh()
clustering = mesh.analyze_clustering(my_data)

for name, pct in sorted(clustering.items(), key=lambda x: -x[1]):
    print(f"{name}: {pct:.1f}%")
```

### Binary Packing

```python
from dimensional_navigation import PhiLens, pack_10bit, unpack_10bit

lens = PhiLens()
encoded = lens.encode(data)

# Pack 3 values per int32
packed = pack_10bit(encoded.indices, encoded.n_frac, encoded.signs)

# Save to file
np.save('compressed.npy', packed)

# Load and unpack
packed = np.load('compressed.npy')
n_int, n_frac, signs = unpack_10bit(packed, n_values)
```

## The Math (Optional)

The golden ratio φ ≈ 1.618 has the property:
```
φ² = φ + 1
```

This makes φ-powers self-similar:
```
φ^(-k) = φ^(-1) × φ^(-(k-1))
```

Neural network weights cluster at φ^(-k) levels because:
1. Training optimizes toward structured representations
2. φ-structure is mathematically optimal for many problems
3. The clustering is statistically significant (Z-score > 3)

## Files

- `core.py` - Main implementation
- `examples.py` - Runnable examples
- `README.md` - Full documentation
- `QUICKSTART.md` - This file
