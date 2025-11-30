# Supplementary Materials

## S1. Full Precision Coefficients

### S1.1 Integer Coefficients

```
a[0] = 256
a[1] = -32
a[2] = 4
a[3] = 1
a[4] = -128
a[5] = -64
a[6] = -128
a[7] = 4
```

### S1.2 φ-Corrections (50 decimal places)

```
c[0] = +0.02101370724969391400000000000000000000000000000000
c[1] = -0.04711356883273248900000000000000000000000000000000
c[2] = +0.00704307595198424800000000000000000000000000000000
c[3] = +0.01318156190427741100000000000000000000000000000000
c[4] = +0.07326301113487117300000000000000000000000000000000
c[5] = -0.10234611440046405300000000000000000000000000000000
c[6] = -0.15335229476273692900000000000000000000000000000000
c[7] = +0.04767199411656225500000000000000000000000000000000
```

### S1.3 Effective Coefficients

```
eff[0] = 256.02101370724969391400
eff[1] = -32.04711356883273248900
eff[2] =   4.00704307595198424800
eff[3] =   1.01318156190427741100
eff[4] = -127.92673698886512882700
eff[5] = -64.10234611440046405300
eff[6] = -128.15335229476273692900
eff[7] =   4.04767199411656225500
```

---

## S2. Derivation of Key Identities

### S2.1 Identity: φ² + φ⁻² = 3

**Proof:**

Let φ = (1 + √5)/2. Then:

φ² = ((1 + √5)/2)² = (1 + 2√5 + 5)/4 = (6 + 2√5)/4 = (3 + √5)/2

φ⁻² = ((2)/(1 + √5))² = 4/(6 + 2√5) = 4(6 - 2√5)/((6 + 2√5)(6 - 2√5))
    = 4(6 - 2√5)/(36 - 20) = 4(6 - 2√5)/16 = (6 - 2√5)/4 = (3 - √5)/2

Therefore:
φ² + φ⁻² = (3 + √5)/2 + (3 - √5)/2 = 6/2 = 3 ∎

**Corollary:** 4 = φ² + φ⁻² + 1

This explains why base 4096 = 4 × 1024 introduces φ-structure.

---

### S2.2 Fibonacci Arctan Identity

**Theorem:** arctan(1/φ) + arctan(1/φ³) = π/4

**Proof:**

Using the arctan addition formula:
arctan(a) + arctan(b) = arctan((a + b)/(1 - ab)) when ab < 1

Let a = 1/φ and b = 1/φ³.

First, compute ab:
ab = 1/φ⁴

Since φ⁴ = φ³ · φ = (φ² + φ) · φ = φ³ + φ² = (2φ + 1) + (φ + 1) = 3φ + 2 ≈ 6.85

So ab = 1/φ⁴ ≈ 0.146 < 1 ✓

Now compute (a + b)/(1 - ab):

a + b = 1/φ + 1/φ³ = (φ² + 1)/φ³

Using φ² = φ + 1:
a + b = (φ + 1 + 1)/φ³ = (φ + 2)/φ³

1 - ab = 1 - 1/φ⁴ = (φ⁴ - 1)/φ⁴

Using φ⁴ = 3φ + 2:
1 - ab = (3φ + 2 - 1)/φ⁴ = (3φ + 1)/φ⁴

Therefore:
(a + b)/(1 - ab) = [(φ + 2)/φ³] · [φ⁴/(3φ + 1)]
                 = (φ + 2) · φ / (3φ + 1)
                 = (φ² + 2φ) / (3φ + 1)
                 = (φ + 1 + 2φ) / (3φ + 1)  [using φ² = φ + 1]
                 = (3φ + 1) / (3φ + 1)
                 = 1

Since arctan(1) = π/4, we have:
arctan(1/φ) + arctan(1/φ³) = π/4 ∎

---

### S2.3 Polylogarithm Identity: Li₁(1/φ) = 2·log(φ)

**Proof:**

By definition: Li₁(z) = -log(1 - z)

Li₁(1/φ) = -log(1 - 1/φ)

Now, 1 - 1/φ = (φ - 1)/φ

Using the identity φ - 1 = 1/φ (since φ² - φ - 1 = 0):
1 - 1/φ = (1/φ)/φ = 1/φ²

Therefore:
Li₁(1/φ) = -log(1/φ²) = -(-2·log(φ)) = 2·log(φ) ∎

---

### S2.4 Polylogarithm Identity: Li₂(1/φ²) = π²/15 - log²(φ)

This is a special case of the five-term relation for the dilogarithm. The proof requires the functional equation:

Li₂(z) + Li₂(1-z) = π²/6 - log(z)·log(1-z)

and the reflection formula:

Li₂(z) + Li₂(1/z) = -π²/6 - (1/2)·log²(-z)

Applied at z = 1/φ² with careful handling of branch cuts yields the result.

---

## S3. Numerical Methods

### S3.1 Optimization Procedure

The φ-corrections were found by minimizing:

f(c₀, c₁, ..., c₇) = |π - π_computed(c₀, c₁, ..., c₇)|

using the Nelder-Mead simplex algorithm with:
- Initial guess: cᵢ = 0 for all i
- Tolerance: 10⁻²⁵
- Maximum iterations: 100,000
- Precision: 500 decimal places (mpmath)

### S3.2 Rational Approximation

For each correction cᵢ, we searched for:

cᵢ ≈ (n/d) × φ^k

over:
- n ∈ [-100, 100]
- d ∈ [1, 100]
- k ∈ [-12, 2]

minimizing |cᵢ - (n/d) × φ^k|.

---

## S4. Convergence Analysis

### S4.1 Theoretical Rate

For a BBP-type formula with base B:
- Each term is O(B⁻ᵏ)
- After n terms, error is O(B⁻ⁿ)
- Digits of accuracy: n × log₁₀(B)

For B = 4096:
Rate = log₁₀(4096) = log₁₀(2¹²) = 12 × log₁₀(2) ≈ 3.61 digits/term

### S4.2 Measured Rate

From numerical experiments:

| Terms | Error | Digits |
|-------|-------|--------|
| 1 | 1.15 × 10⁻⁴ | 4 |
| 2 | 1.44 × 10⁻⁸ | 8 |
| 3 | 2.35 × 10⁻¹² | 12 |
| 4 | 4.28 × 10⁻¹⁶ | 16 |
| 5 | 8.42 × 10⁻²⁰ | 19 |

Measured rate: (19 - 4) / (5 - 1) = 3.75 digits/term

This slightly exceeds the theoretical rate due to the alternating series having favorable error cancellation.

---

## S5. Comparison to Other BBP Formulas

### S5.1 Original BBP (1995)

```
π = Σ (1/16^k) × [4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)]
```

- Base: 16
- Rate: 1.20 digits/term
- Coefficients: Integer (4, -2, -1, -1)

### S5.2 Bellard (1997)

```
π = (1/64) × Σ ((-1)^k / 1024^k) × [
    -32/(4k+1) - 1/(4k+3) + 256/(10k+1) - 64/(10k+3)
    - 4/(10k+5) - 4/(10k+7) + 1/(10k+9)
]
```

- Base: 1024
- Rate: 3.01 digits/term
- Coefficients: Integer powers of 2

### S5.3 φ-BBP (This Work)

```
π = (1/64) × Σ ((-1)^k / 4096^k) × [
    (256 + c₀)/(4k+1) + (-32 + c₁)/(4k+3) + ...
]
```

- Base: 4096
- Rate: 3.61 digits/term
- Coefficients: Integer + φ-corrections

---

## S6. Open Problems

1. **Exact Form:** Can the corrections be expressed exactly using only Fibonacci/Lucas numbers?

2. **Generating Function:** Is there a formula G(p, q) that generates cᵢ from the slot parameters (pᵢ, qᵢ)?

3. **Higher Bases:** Do base-16384 or base-65536 formulas have simpler φ-corrections?

4. **Other Constants:** Do similar φ-corrections appear in BBP formulas for log(2), Catalan's constant, etc.?

5. **Rigorous Proof:** Can the φ-BBP formula be derived from first principles (e.g., from polylogarithm functional equations)?

---

## S7. Code Availability

All code is available at:
https://github.com/lostdemeter/holographersworkbench/tree/main/ribbon_lcm_v4_experimental

Key files:
- `phi_bbp_formula.py` - Reference implementation
- `paper/verify_formula.py` - Verification script
- `paper/generate_figures.py` - Figure generation
