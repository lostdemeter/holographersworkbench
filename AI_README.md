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

## Demos: REPL-Ready Notebooks

`demos/` (6 .ipnb, <1min run, NumPy-only). Copy cells to code_execution:

| Demo | Ties To | Output | Prototype Use |
|------|---------|--------|---------------|
| `demo_1_spectral_scoring.ipynb` | Sec1 | Top-10 zeta peaks | Score toy candidates |
| `demo_2_phase_retrieval.ipynb` | Sec2 | Envelope + PV=0.01 | Denoise sin wave |
| `demo_3_holographic_refinement.ipynb` | Sec3 | +4dB PSNR, 100% peaks | Refine Gauss + noise |
| `demo_4_sublinear_optimization.ipynb` | Sec3 | 10k→100, O(√n) | Scale expensive score |
| `demo_5_complete_workflow.ipynb` | Pipeline | 5k→50, 99% off | Full chain |
| `demo_6_srt_calibration.ipynb` | Sec3 | Best z=0.05, R=0.8 | Tune Dirac params |

Start: `demo_5` for pipeline baseline.

---

## AI Prototyping Toys

Direct REPL starters (extend with your data):
1. **Zeta Filter**: Score [1e6:1e7] primes → top-100 harmonics.
2. **Hilbert Clean**: ECG/noise → env + peaks.
3. **Holo Boost**: Noisy embeddings → PSNR + blend.
4. **Sublinear Query**: 1M cands → top-1k in O(log n).

Query: "Prototype [toy] with [data]" → I'll chain.

--- 
