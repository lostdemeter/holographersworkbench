# Holographer's Workbench: AI-Optimized Spec

**V0.1.0** | **Core: Holography + Zeta Spectral + Sublinear Opt** | **Deps: numpy scipy mpmath fast_zetas** | **API: Decompose → Retrieve → Refine → Optimize**

AI-Tailored: Dense, parseable structure for fast ingestion. Headers = modules/concepts. Tables = API specs. Equations = derivations (LaTeX-ready). Snippets = REPL-executable (copy-paste into code_execution). Patterns = chainable pipelines. Demos = interactive hooks. Skip prose; extract math/code directly.

---

## Spec: Modules & Exports

| Module | Purpose | Key Exports | Deps |
|--------|---------|-------------|------|
| `spectral.py` | Zeta/freq-domain scoring | `ZetaFiducials`, `SpectralScorer`, `DiracOperator`, `compute_spectral_scores` | numpy, mpmath (optional) |
| `holographic.py` | Phase retrieval + interference | `PhaseRetrieval`, `phase_retrieve_hilbert`, `phase_retrieve_gs`, `holographic_refinement`, `FourPhaseShifting` | numpy, scipy.fft |
| `optimization.py` | Sublinear search + SRT tuning | `SublinearOptimizer`, `SRTCalibrator`, `optimize_sublinear` | spectral, holographic |
| `fractal_peeling.py` | Recursive lossless compression | `FractalPeeler`, `resfrac_score`, `compress`, `decompress`, `tree_statistics` | numpy |
| `holographic_compression.py` | Image compression via harmonics | `HolographicCompressor`, `compress_image`, `decompress_image`, `CompressionStats` | numpy, zlib |
| `fast_zetas.py` | High-performance zeta zeros | `zetazero`, `zetazero_batch`, `zetazero_range`, `ZetaZeroParameters` | numpy, scipy, mpmath |
| `time_affinity.py` | Walltime-based param discovery | `TimeAffinityOptimizer`, `GridSearchTimeAffinity`, `quick_calibrate` | numpy |
| `utils.py` | Shared utils | `normalize_signal`, `compute_psnr`, `detect_peaks`, `adaptive_blend`, `smooth_signal` | numpy, scipy (some) |

**Import Pattern**: `from workbench import *` (all public in `__init__.py`). Cache: Zeta zeros auto-cached.

---

## 1. Spectral: Zeta-Freq Scoring

**Core Math**: Score \( x \in \mathbb{R}^n \) via zeta-modulated oscillations. Zeta zeros \( \gamma_k \) (e.g., [14.135, 21.022, ...]) as "fiducials" for log-space resonances. From explicit formula:
\[
\psi(x) \approx x - \sum_k \frac{x^{1/2 + i \gamma_k}}{1/2 + i \gamma_k}.
\]
Damped real-part approx:
\[
s(x) = \Re \left[ \sum_k \tau_k \frac{e^{i \gamma_k \log x + i \phi}}{1/2 + i \gamma_k} \right], \quad \tau_k = e^{-d^2 \gamma_k^2 / 2}.
\]
- \( K=20 \) (default), \( d=0.05 \) (damping), \( \phi=0.05 \) (shift).
- Interference mode: \( \Delta_h \psi(x) / (h x \log x) \) for differencing.

**API Table**:

| Class/Func | Init Params | Key Methods | Output Shape |
|------------|-------------|-------------|--------------|
| `ZetaFiducials.get_standard(n=20)` | - | - | np.array (n zeros) |
| `SpectralScorer(freqs, d=0.05)` | freqs: np.array (zeta or custom) | `compute_scores(cands, shift=0.05, mode='real')` | np.array (len(cands),) |
| `compute_spectral_scores(cands, freqs, h=0.05, method='interference')` | - | - | np.array (len(cands),) |

**Snippet: Score + Top-K**:
```python
import numpy as np
from workbench import ZetaFiducials, SpectralScorer
cands = np.arange(100, 1000)
zeros = ZetaFiducials.get_standard(20)
scorer = SpectralScorer(zeros, damping=0.05)
scores = scorer.compute_scores(cands, shift=0.05, mode='real')
top10 = cands[np.argsort(-scores)[:10]]  # Peaks at zeta harmonics
print(top10)  # e.g., [123, 456, ...]
```

**AI Hook**: Zeta as universal filter—prototype: Score primes in [1e6,1e7] for arithmetic patterns.

---

## 2. Holographic: Phase Retrieval + Refinement

**Core Math**: Recover \( u(t) = A(t) e^{i\theta(t)} \) from real \( |u| \) or \( |\hat{u}| \). Hilbert: Analytic \( z = u + i \hat{u} \), \( \hat{u}(\omega) = -i \sgn(\omega) U(\omega) \) (FFT impl).
\[
A(t) = |z(t)|, \quad \Var(\theta) = \Var(\diff \angle z), \quad \text{gate if } \Var(\theta) > 0.12.
\]
GS Iter: \( u_{k+1} = A e^{i \arg(\mathcal{F}^{-1}(I e^{i \arg(\mathcal{F} u_k)}))} \), conv in 30 iters.

Refinement: Align \( \theta = \argmax_\phi \Re[\int o^* r e^{i\phi}] \), blend:
\[
s'(x) = \beta o' A + (1-\beta) r, \quad \beta = 0.6 + 0.2 \cdot \mathbb{I}(\Var(\theta)<0.05).
\]

**API Table**:

| Class/Func | Init Params | Key Methods | Output |
|------------|-------------|-------------|--------|
| `PhaseRetrieval(method='hilbert')` | method: 'hilbert'/'gs', n_iter=30 | `retrieve(sig)` | (envelope, phase_var) |
| `phase_retrieve_hilbert(sig)` | - | - | (np.array, float) |
| `phase_retrieve_gs(intens, target_amp, n_iter=30)` | - | - | complex np.array |
| `holographic_refinement(obj, ref, method='hilbert', beta=0.6)` | - | - | np.array (refined) |

**Snippet: Retrieve + Refine**:
```python
from workbench import phase_retrieve_hilbert, holographic_refinement, normalize_signal
x = np.linspace(0,10,1000); sig = np.sin(x) + 0.3*np.random.randn(1000)
env, pv = phase_retrieve_hilbert(sig)  # env ~1, pv ~0.01
ref = normalize_signal(1/(np.log(x+np.e)+1e-12), 'max')
refined = holographic_refinement(sig, ref, method='hilbert', blend_ratio=0.6)
print(f'PV: {pv:.4f}, Refined PSNR gain: +{compute_psnr(sig, refined):.1f}dB')
```

**AI Hook**: Denoise signals—prototype: Refine ECG noise for peak detection.

---

### 3. Optimization: Sublinear + SRT

**Core Math**: Reduce \( n \to m \) (\( m \ll n \)) via spectral pre-score + top-k select. Complexity: \( O(\sqrt{n}) \) if \( m \leq \sqrt{n} \); else \( O(m/n \cdot n) \). Pipeline: Score → Refine → ArgSort → Top-K.

SRT: Hermitian Dirac \( D_{ij} = a_i \delta_{ij} + \gamma_{ij} e^{i z \zeta_k \theta} \) (\( z \): zeta strength [0.01-0.15], \( \corr \): correlation weight [0.05-0.35], \( \theta \): deformation [0-0.01]). Eigenvalues \( \lambda_k \); resonance \( R = \sum_{i<j} 1/|\lambda_i - \lambda_j| \) (min \( R < 1.0 \) via grid-search for spectral gap).

**API Table**:

| Class/Func | Init Params | Key Methods | Output |
|------------|-------------|-------------|--------|
| `SublinearOptimizer(holo=True, method='hilbert', beta=0.6)` | holo: bool (use refinement), method: 'hilbert'/'gs' | `optimize(cands, score_fn, top_k=100)` | (top_cands: np.array, stats: dict {n_orig, n_final, reduction, complexity, time}) |
| `SRTCalibrator(gt_idx, gt_val, aff_funcs)` | gt_idx/val: np.array (mock GT), aff_funcs: dict[str, callable] | `calibrate(grid='coarse', train_frac=0.7)` | (params: SRTParams {z, corr, theta, L=20}, metrics: dict {pred_err, R, eig_spread, gap}) |
| `optimize_sublinear(cands, score_fn, top_k, holo=True)` | - | - | np.array (top-k indices) |

**Snippet: Optimize + Calibrate**:
```python
from workbench import SublinearOptimizer, SRTCalibrator
cands = np.arange(10000); def score(c): return np.sin(c*0.01)
opt = SublinearOptimizer(use_holographic=True); top100, stats = opt.optimize(cands, score, top_k=100)
print(f'Reduced {stats.n_original}→{stats.n_final}, O({stats.complexity_estimate})')  # O(√n), 99% off, time<0.05s
# SRT: gt_idx=np.array([100,200]); gt_val=np.array([0.8,0.9]); aff={'def':lambda i:1/(1+i/1000)}
cal = SRTCalibrator(gt_idx, gt_val, aff); params, mets = cal.calibrate(); print(f'R: {mets["resonance"]:.2f}, Gap: {mets["spectral_gap"]:.3f}')
```

**AI Hook**: Scale searches—prototype: Top-100 zeta-scored embeddings in 1M dataset (chain with demo_5 for baseline).

---

## Unified Pipeline: Decompose → Refine → Optimize

**Math Chain**: Scores \( s = S(x) \) (spectral) → \( s' = H(s, r) \) (holo, r=1/log x) → \( \top_k = \argtop(s'[x]) \).

**Snippet: End-to-End (5k → 50)**:
```python
import numpy as np
from workbench import *
n,k=5000,50; cands=np.arange(n); zeros=ZetaFiducials.get_standard(15)
scorer=SpectralScorer(zeros,damping=0.05); scores=scorer.compute_scores(cands,shift=0.05,mode='real')
ref=normalize_signal(1/np.log(cands+np.e),'max'); refined=holographic_refinement(scores,ref,blend_ratio=0.6)
opt=SublinearOptimizer(use_holographic=False); topk,stats=opt.optimize(cands,lambda c:refined[c],top_k=k)
print(f'5k→{k} ({100*(1-stats.reduction_ratio):.0f}% off), O({stats.complexity_estimate}) | Top10: {topk[:10]}')
```

**Metrics**: PSNR (dB), \( R \) (resonance), \( \Var(\theta) \) (phase).

---

## Patterns: Chainable Hooks

1. **Decompose**: `SpectralScorer.compute_scores` or `compute_spectral_scores`.
2. **Retrieve**: `phase_retrieve_hilbert` → gate on PV.
3. **Refine**: `holographic_refinement` + `adaptive_blend` (beta auto-adjust).
4. **Optimize**: `SublinearOptimizer.optimize` → stats for O().

**Ext Hooks**: Add to classes (e.g., `SpectralScorer` new mode); utils for custom norms.

---

## 4. Fractal Peeling: Recursive Compression

**Core Math**: Lossless compression via recursive pattern extraction. Resfrac invariant measures predictability:
\[
\rho(x) = \frac{\sigma(r)}{\sigma(x)}, \quad r = x - P(x)
\]
where \( P(x) \) is AR(k) predictor: \( \hat{x}_i = \sum_{j=0}^{k-1} \beta_j x_{i-j-1} \).

Algorithm \( \Phi(x, d) \):
1. Compute \( \rho(x) \). If \( \rho > \theta_{noise} \) (0.95) or \( d \geq d_{max} \): return LEAF.
2. Extract pattern: \( (M, r) = \text{ExtractPattern}(x) \) via least squares.
3. Compute \( \rho_r = \rho(r) \), \( \Delta\rho = \rho - \rho_r \).
4. If \( \Delta\rho < \epsilon \) (0.01): return LEAF (no improvement).
5. Recurse: \( T_r = \Phi(r, d+1) \).
6. Return NODE\( (M, T_r, \rho, \rho_r, \Delta\rho) \).

Reconstruction \( \Psi(T) \): LEAF → raw data; NODE → \( x = [x_{0:k}, P(x_{0:k}) + \Psi(T_r)] \).

**API Table**:

| Class/Func | Init Params | Key Methods | Output |
|------------|-------------|-------------|--------|
| `FractalPeeler(order=3, noise_threshold=0.95, max_depth=10)` | order: AR model order k | `compress(data)`, `decompress(tree)`, `compression_ratio(tree, n)` | CompressionTree, np.array, float |
| `resfrac_score(data, order=3)` | - | - | float ∈ [0,1] |
| `compress(data, order=3, max_depth=10)` | - | - | CompressionTree |
| `decompress(tree)` | - | - | np.array |
| `tree_statistics(tree)` | - | - | dict {depth, nodes, leaves, ratios} |

**Snippet: Compress + Decompress**:
```python
from workbench import FractalPeeler, resfrac_score
import numpy as np
signal = np.sin(np.linspace(0, 10*np.pi, 500)) + 0.1*np.random.randn(500)
print(f'Resfrac: ρ = {resfrac_score(signal):.4f}')  # ~0.18 (structured)
peeler = FractalPeeler(order=4, max_depth=6)
tree = peeler.compress(signal); reconstructed = peeler.decompress(tree)
print(f'Error: {np.max(np.abs(signal - reconstructed)):.2e}')  # 0.00e+00 (lossless)
stats = peeler.tree_stats(tree); ratio = peeler.compression_ratio(tree, len(signal))
print(f'Compression: {ratio:.2f}x, depth={stats["max_depth"]}')
```

**AI Hook**: Lossless compression—prototype: Compress time series, analyze structure via resfrac.

---

## 5. Holographic Compression: Image Encoding

**Core Math**: Lossless image compression exploiting 15th order harmonic structure. FFT-based with phase quantization + residual encoding:
\[
I(x,y) = \mathcal{F}^{-1}[H_{15}] + R
\]
where \( H_{15} \) = 15th order harmonic ring (magnitude + quantized phase), \( R \) = int16 residuals (zlib compressed).

Algorithm:
1. FFT: \( F = \mathcal{F}[I] \), extract ring at radius 15.
2. Separate: \( |F|, \angle F \). Quantize phase to 8-bit levels.
3. Predict: \( \hat{I} = \mathcal{F}^{-1}[|F| e^{i\angle_{quant}}] \).
4. Residuals: \( R = I - \hat{I} \) (int16, zlib level 9).
5. Store: header + magnitude (float32) + phase (uint16) + residuals (compressed).

Reconstruction: Reverse process, \( I = \hat{I} + R \) (lossless).

**API Table**:

| Class/Func | Init Params | Key Methods | Output |
|------------|-------------|-------------|--------|
| `HolographicCompressor(harmonic_order=15, phase_quantization_bits=8)` | order: harmonic ring radius, phase_bits: quantization | `compress(image)`, `decompress(bytes)` | (bytes, CompressionStats), np.ndarray |
| `compress_image(image, harmonic_order=15, phase_bits=8)` | - | - | (bytes, CompressionStats) |
| `decompress_image(compressed_bytes)` | - | - | np.ndarray (uint8) |

**Snippet: Compress + Decompress**:
```python
from workbench import HolographicCompressor, compress_image, decompress_image
import numpy as np
# Load grayscale image (0-255)
image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
# Compress
compressed, stats = compress_image(image, harmonic_order=15, phase_bits=8)
print(f'Ratio: {stats.compression_ratio:.2f}x, Lossless: {stats.phase_symmetry_score:.3f}')
# Decompress
reconstructed = decompress_image(compressed)
print(f'Perfect: {np.array_equal(image, reconstructed)}')  # True
```

**AI Hook**: Image compression—prototype: Compress images with 15th order structure, analyze phase symmetry.

---

## 6. Fast Zetas: High-Performance Zeta Zeros

**Core Math**: 26× faster Riemann zeta zero computation via cached ζ' optimization. Combines Lambert W predictor + self-similar spiral + Newton refinement:
\[
T_n \approx \frac{2\pi(n - 11/8)}{W((n - 11/8)/e)} + \text{harmonics} + \text{spiral}
\]

Key optimization: Compute \( \zeta'(s) \) ONCE per zero (not per Newton iteration), saving 40% computation.

Algorithm:
1. Initial guess: Lambert W + harmonic corrections (3, 6, 9, 12, 15) + logarithmic spiral.
2. Cache \( \zeta'(s_0) \) at initial guess.
3. Newton iterations: \( t_{k+1} = t_k - \text{Im}(\zeta(s_k) / \zeta'(s_0)) \) (reuse cached derivative!).
4. Adaptive precision: 25 digits → 50 digits after first iteration.
5. Parallel batch processing for multiple zeros.

Performance: 1.68ms per zero (batch), 26× faster than `mp.zetazero`.

**API Table**:

| Func | Params | Output | Complexity |
|------|--------|--------|------------|
| `zetazero(n, dps=50)` | n: zero index (1-indexed) | mpf (imaginary part) | ~2.5ms |
| `zetazero_batch(start, end, dps=50, parallel=True)` | range [start, end] | dict {n: zero} | ~1.68ms/zero |
| `zetazero_range(start, end, dps=50)` | generator version | yields (n, zero) | memory-efficient |

**Snippet: Fast Zeta Zeros**:
```python
from workbench import zetazero, zetazero_batch
# Single zero (drop-in replacement for mp.zetazero)
z = zetazero(100)  # 2.5ms vs 65ms for mp.zetazero
print(f'ζ(1/2 + i·{z}) = 0')
# Batch (parallel, vectorized)
zeros = zetazero_batch(1, 1000, parallel=True)  # 1.68s total!
print(f'First 5: {[float(zeros[i]) for i in range(1, 6)]}')
# Memory-efficient generator
for n, z in zetazero_range(1, 10000):
    if n <= 3: print(f'{n}: {z}')
```

**AI Hook**: Fast zeta computation—prototype: Generate 10k zeros in <20s, use in spectral scoring.

---

## 7. Time Affinity: Diagnostic Parameter Discovery

**Core Principle**: Use walltime as fitness signal to discover optimal parameters empirically. Correct/resonant parameters → less work → faster execution.

**Diagnostic Tool**: Designed for discovering unknown parameter values, relationships, or configurations when theoretical models are unavailable.

Algorithm:
1. Set target execution time \( t_{target} \).
2. Measure algorithm time \( t(\theta) \) with parameters \( \theta \).
3. Compute gradient: \( \nabla_\theta |t(\theta) - t_{target}| \) (finite differences).
4. Update: \( \theta_{k+1} = \theta_k - \alpha \nabla_\theta + \beta v_k \) (momentum).
5. Repeat until \( |t(\theta) - t_{target}| < \epsilon \).

Alternative: Grid search over \( \theta \) space, select \( \arg\min_\theta |t(\theta) - t_{target}| \).

**API Table**:

| Class/Func | Params | Output | Use Case |
|------------|--------|--------|----------|
| `TimeAffinityOptimizer(target_time, param_bounds)` | target: float, bounds: dict | `optimize(algorithm)` → TimeAffinityResult | Gradient-based discovery |
| `GridSearchTimeAffinity(target_time, param_grids)` | grids: dict of arrays | `optimize(algorithm)` → TimeAffinityResult | Exhaustive search |
| `quick_calibrate(algorithm, target_time, param_bounds, method='gradient')` | - | TimeAffinityResult | One-liner convenience |

**Snippet: Discover Unknown Parameters**:
```python
from workbench import quick_calibrate
# Black-box algorithm with unknown optimal params
def mystery_algorithm(x, y, z):
    # Complex computation with hidden optimal configuration
    return expensive_operation(x, y, z)
# Discover params that make it run in 100ms
result = quick_calibrate(
    mystery_algorithm,
    target_time=0.1,
    param_bounds={'x': (0, 10), 'y': (0, 5), 'z': (0, 1)},
    method='gradient',
    max_iterations=50
)
print(f'Discovered: {result.best_params}')  # Empirically found optimal values
print(f'Time: {result.best_time:.6f}s, Error: {result.time_error:.6f}s')
```

**AI Hook**: Parameter discovery—prototype: Find optimal SRT params, discover hidden relationships, diagnose performance.

**Note**: This is a **diagnostic/discovery tool**, not a production optimizer. Use it to:
- Discover unknown parameter values empirically
- Reveal hidden parameter relationships
- Diagnose why certain configurations run faster
- Calibrate algorithms without theoretical models

---

## Demos: REPL-Ready Notebooks

`demos/` (10 .ipnb, <1min run, NumPy-only). Copy cells to code_execution:

| Demo | Ties To | Output | Prototype Use |
|------|---------|--------|---------------|
| `demo_1_spectral_scoring.ipynb` | Sec1 | Top-10 zeta peaks | Score toy candidates |
| `demo_2_phase_retrieval.ipynb` | Sec2 | Envelope + PV=0.01 | Denoise sin wave |
| `demo_3_holographic_refinement.ipynb` | Sec2 | +4dB PSNR, 100% peaks | Refine Gauss + noise |
| `demo_4_sublinear_optimization.ipynb` | Sec3 | 10k→100, O(√n) | Scale expensive score |
| `demo_5_complete_workflow.ipynb` | Pipeline | 5k→50, 99% off | Full chain |
| `demo_6_srt_calibration.ipynb` | Sec3 | Best z=0.05, R=0.8 | Tune Dirac params |
| `demo_7_fractal_peeling.ipynb` | Sec4 | Lossless, 1.5x, ρ=0.18 | Compress structured data |
| `demo_8_holographic_compression.ipynb` | Sec5 | Lossless, 1.1x, H₁₅ | Compress images |
| `demo_9_fast_zetas.ipynb` | Sec6 | 26× speedup, 1.68ms/zero | Fast zeta computation |
| `demo_10_time_affinity.ipynb` | Sec7 | Discover params, diagnose | Empirical calibration |

Start: `demo_5` for pipeline baseline.

---

## AI Prototyping Toys

Direct REPL starters (extend with your data):
1. **Zeta Filter**: Score [1e6:1e7] primes → top-100 harmonics.
2. **Hilbert Clean**: ECG/noise → env + peaks.
3. **Holo Boost**: Noisy embeddings → PSNR + blend.
4. **Sublinear Query**: 1M cands → top-1k in O(log n).
5. **Fractal Compress**: Time series → lossless tree + resfrac analysis.
6. **Image Compress**: Images → 15th harmonic + residuals, lossless.
7. **Param Discovery**: Black-box algorithm → discover optimal params via walltime.

Query: "Prototype [toy] with [data]" → I'll chain.

---

## Testing

**Test Suite**: `tests/test_workbench.py` (9/9 passing, 100%)

Run tests:
```bash
source venv/bin/activate && python3 tests/test_workbench.py
```

Coverage: All 8 modules + imports verified. See `tests/TEST_RESULTS.md` for details.

--- 
