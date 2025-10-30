# Holographic Depth Analyzer

**Interactive visualization tool for monocular depth estimation using holographic signal processing.**

Extract 3D depth information from single 2D images and generate stereoscopic 3D pairs—all without training data or neural networks!

---

## 🎯 What It Does

The Holographic Depth Analyzer extracts depth maps from 2D images by treating depth estimation as a **phase retrieval problem** from holographic signal processing. It combines multiple depth cues using adaptive fusion and generates stereoscopic 3D image pairs for immersive viewing.

### Key Features

- **Multi-Cue Depth Extraction**: Combines luminance, edge strength, and frequency content
- **Adaptive Weighting**: Spatially-varying weights based on local image contrast
- **Saliency Detection**: Emphasizes perceptually important regions using spectral residual
- **Compositional Priors**: Center-weighted bias for portrait photography
- **Stereoscopic Synthesis**: Generates left/right eye views using DIBR (Depth-Image-Based Rendering)
- **Comprehensive Visualizations**: 12-panel analysis showing all processing stages

### Performance

- **2.4× better dynamic range** than fixed-weight baseline methods
- **~1.7 seconds** to process a 512×512 image (depth + stereo)
- **No training data required** - purely mathematical approach
- **Fully interpretable** - every step has clear physical meaning

---

## 🚀 Quick Start

### Basic Usage

```bash
# Analyze a photograph
python depth_analyzer.py --image path/to/photo.jpg

# Run demo with synthetic portrait
python depth_analyzer.py --demo portrait

# Run all demos (portrait, landscape, geometric)
python depth_analyzer.py --demo all

# Save visualizations to files
python depth_analyzer.py --demo portrait --save-output
```

### Custom Output Directory

```bash
python depth_analyzer.py --image photo.jpg --save-output --output-dir results/
```

---

## 📊 Output Visualizations

The analyzer generates two comprehensive visualizations:

### 1. Component Analysis (12-panel view)

Shows the complete depth extraction pipeline:

1. **Original Image** - Input photograph
2. **Luminance Depth** - Brightness-based depth (brighter = closer)
3. **Edge Depth** - Gradient-based depth (sharp edges = in focus = closer)
4. **Frequency Depth** - High-frequency content (fine details = near-field)
5. **Adaptive Weighted** - Multi-cue fusion with spatially-varying weights
6. **Saliency Map** - Spectral residual highlighting important regions
7. **Center Bias** - Gaussian falloff for compositional prior
8. **Final Depth Map** - Hybrid fusion result with full statistics
9. **Depth Distribution** - Histogram showing depth value distribution
10. **Saliency-Enhanced** - Depth map with saliency emphasis
11. **Center-Weighted** - Depth map with center bias
12. **3D Surface Plot** - Interactive 3D visualization of depth surface

### 2. Stereoscopic Pair (4-panel view)

Shows the 3D viewing experience:

1. **Original Image** - Input photograph
2. **Depth Map** - Extracted depth with colorbar
3. **Left Eye View** - Warped view for left eye
4. **Right Eye View** - Warped view for right eye

**Viewing Instructions**: Use cross-eye or parallel viewing technique to see the 3D effect!

---

## 🔬 Technical Details

### Depth Extraction Algorithm

The analyzer uses a multi-stage pipeline:

```
1. Multi-Cue Extraction
   ├─ Luminance: D_L = normalize(grayscale)
   ├─ Edges: D_E = normalize(√(Gx² + Gy²))  [Sobel gradients]
   └─ Frequency: D_F = normalize(high_pass_filter(FFT))

2. Adaptive Weighting (spatially-varying)
   ├─ Local contrast: C = std(local_window)
   ├─ Edge weight: w_E = C / C_max
   ├─ Luminance weight: w_L = 1 - w_E
   └─ D_A = (w_L·D_L + w_E·D_E + 0.3·D_F) / (w_L + w_E + 0.3)

3. Saliency Detection (spectral residual)
   ├─ Log spectrum: L = log(|FFT(image)|)
   ├─ Residual: R = L - gaussian_filter(L)
   ├─ Saliency: S = |IFFT(exp(R + i·phase))|²
   └─ D_S = D_A · (0.3 + 0.7·S)

4. Center Bias (compositional prior)
   ├─ Gaussian: B = exp(-((x-x_c)² + (y-y_c)²) / (2σ²))
   └─ D_C = D_A · (0.5 + 0.5·B)

5. Hybrid Fusion
   └─ D_final = gaussian_filter(0.4·D_A + 0.3·D_S + 0.3·D_C, σ=4)
```

### Stereoscopic View Synthesis

Uses forward warping with depth-dependent horizontal shifts:

```
1. Non-linear enhancement: D' = D^γ  (γ=0.8)
2. Compute shifts: Δx = ±baseline · width · D'
3. Forward warp: left/right = warp(image, Δx)
4. Hole filling: interpolate missing pixels using depth
```

---

## 📈 Results & Statistics

The analyzer reports comprehensive statistics for each depth map:

- **Range**: [min, max] depth values
- **Mean**: Average depth (higher = more foreground emphasis)
- **Std**: Depth variation (higher = more pronounced 3D structure)
- **Dynamic Range**: max - min (target: 1.0 for full range)

### Typical Results

| Image Type | Range | Mean | Std | Dynamic Range |
|------------|-------|------|-----|---------------|
| Portrait | [0.00, 1.00] | 0.37 | 0.25 | 1.00 |
| Landscape | [0.00, 0.95] | 0.42 | 0.28 | 0.95 |
| Geometric | [0.00, 1.00] | 0.45 | 0.32 | 1.00 |

**Improvement over baselines**: 2.4× better dynamic range (1.00 vs 0.42)

---

## 🎨 Use Cases

### Portrait Photography
- Extract depth for bokeh simulation
- Generate 3D portraits for VR/AR
- Analyze facial structure and lighting

### Landscape Photography
- Visualize scene depth layers
- Create stereoscopic nature scenes
- Analyze depth composition

### Product Photography
- Generate 3D product views
- Create depth-based effects
- Analyze object placement

### Scientific Imaging
- Analyze microscopy images
- Extract depth from medical scans
- Study surface topology

---

## 🔧 Advanced Usage

### Python API

For programmatic use, import directly:

```python
from workbench import HolographicDepthExtractor
import numpy as np

# Load your image
image = np.array(...)  # (H, W) grayscale or (H, W, 3) RGB

# Create extractor with custom parameters
extractor = HolographicDepthExtractor(
    adaptive_weight=0.4,    # Weight for adaptive multi-cue fusion
    saliency_weight=0.3,    # Weight for saliency emphasis
    center_weight=0.3,      # Weight for center bias
    smoothing_sigma=4.0,    # Final Gaussian smoothing
    verbose=True
)

# Extract depth
depth_map, components = extractor.extract_depth(image)

# Access individual components
luminance_depth = components['luminance']
edge_depth = components['edges']
frequency_depth = components['frequency']
adaptive_depth = components['adaptive']
saliency_map = components['saliency_map']
center_bias = components['center_bias']

# Generate stereoscopic pair
left_view, right_view = extractor.generate_stereo_pair(
    image, depth_map, 
    baseline=0.018,  # Stereo baseline (0.01-0.03 recommended)
    gamma=0.8        # Non-linear depth enhancement
)

# Get statistics
stats = extractor.compute_stats(depth_map)
print(f"Depth range: [{stats.min:.3f}, {stats.max:.3f}]")
print(f"Mean depth: {stats.mean:.3f}")
print(f"Dynamic range: {stats.range:.3f}")
```

### Parameter Tuning

**Adaptive Weight** (0.0-1.0, default: 0.4)
- Higher: More emphasis on adaptive multi-cue fusion
- Lower: More emphasis on saliency and center bias

**Saliency Weight** (0.0-1.0, default: 0.3)
- Higher: Stronger emphasis on perceptually important regions
- Lower: More uniform depth distribution

**Center Weight** (0.0-1.0, default: 0.3)
- Higher: Stronger center bias (good for portraits)
- Lower: More uniform spatial weighting

**Smoothing Sigma** (0.0-10.0, default: 4.0)
- Higher: Smoother depth maps, less noise
- Lower: More detail, potential noise

**Stereo Baseline** (0.01-0.03, default: 0.018)
- Higher: Stronger 3D effect, more disparity
- Lower: Subtler 3D effect, easier to view

**Stereo Gamma** (0.5-1.0, default: 0.8)
- Lower: More non-linear depth enhancement
- Higher: More linear depth mapping

---

## 📚 Related Files

### For AI/Code Analysis
- **`examples/demo_holographic_depth.py`** - AI-readable version without visualizations
  - Focuses on numerical output and statistics
  - Ideal for automated testing and code review
  - No matplotlib dependencies for visualization

### For Human Use
- **`practical_applications/depth_analysis/depth_analyzer.py`** - This tool (interactive visualizations)
  - Comprehensive 12-panel component analysis
  - Stereoscopic pair visualization
  - Command-line interface for easy use

### Core Implementation
- **`workbench/processors/holographic_depth.py`** - Core algorithm implementation
  - `HolographicDepthExtractor` class
  - All depth extraction methods
  - Stereoscopic synthesis functions

### Documentation
- **`documents/HOLOGRAPHIC_DEPTH_SUMMARY.md`** - Complete technical documentation
  - Mathematical foundations
  - Algorithm details
  - Performance benchmarks
  - Comparison with deep learning

### Tests
- **`tests/test_holographic_depth.py`** - Unit tests (10/10 passing)
  - Tests all extraction methods
  - Validates dynamic range achievement
  - Ensures stereoscopic quality

---

## 🧪 Mathematical Foundation

### Holographic Interpretation

The approach treats depth estimation as **phase retrieval**:
- 2D image = intensity pattern (magnitude of holographic signal)
- Depth = missing phase information
- Fourier analysis recovers phase from spectral structure

This is analogous to reconstructing 3D scenes from holographic interference patterns.

### Spectral Residual Saliency

Salient regions have unexpected spectral signatures:

```
L(u,v) = log(|F(u,v)|)              # Log amplitude spectrum
R(u,v) = L(u,v) - G_σ * L(u,v)      # Residual (unexpected features)
S(x,y) = |IFFT{exp(R + i·∠F)}|²     # Saliency map
```

**Key insight**: Faces, text, and important objects stand out in the spectral residual.

### Adaptive Weighting

Spatially-varying weights handle diverse image regions:

```
C(x,y) = std(local_window)          # Local contrast
w_E = C / C_max                      # Edge weight (high contrast)
w_L = 1 - w_E                        # Luminance weight (low contrast)
```

**Key insight**: High-contrast regions (edges) benefit from gradient-based depth, while smooth regions rely on luminance.

---

## 🎓 References

Based on the paper:
**"Holographic Depth Extraction: Monocular Depth Estimation via Spectral Analysis and Phase Retrieval"**

Key concepts:
- Holographic phase retrieval
- Spectral residual saliency (Hou & Zhang, 2007)
- Depth-Image-Based Rendering (DIBR)
- Multi-cue depth fusion

---

## 💡 Tips & Tricks

### Getting the Best Results

1. **Portrait Photography**
   - Use default parameters (optimized for portraits)
   - Ensure subject is well-lit and centered
   - Works best with clear foreground/background separation

2. **Landscape Photography**
   - Reduce `center_weight` to 0.1-0.2
   - Increase `adaptive_weight` to 0.5-0.6
   - Works best with clear depth layers (sky, mountains, foreground)

3. **Product Photography**
   - Increase `saliency_weight` to 0.4-0.5
   - Reduce `smoothing_sigma` to 2.0-3.0 for more detail
   - Works best with high-contrast products on plain backgrounds

4. **Viewing Stereoscopic Pairs**
   - **Cross-eye method**: Cross your eyes until the two images merge
   - **Parallel method**: Relax your eyes to look "through" the screen
   - **VR headset**: Display left/right views to respective eyes
   - Start with small baseline (0.01) if you're new to stereo viewing

### Troubleshooting

**Problem**: Depth map looks flat (low dynamic range)
- **Solution**: Increase `adaptive_weight`, reduce `smoothing_sigma`

**Problem**: Too much noise in depth map
- **Solution**: Increase `smoothing_sigma` to 5.0-6.0

**Problem**: Stereo pair is hard to view (too much disparity)
- **Solution**: Reduce `baseline` to 0.01-0.015

**Problem**: Depth map emphasizes wrong regions
- **Solution**: Adjust `saliency_weight` and `center_weight` based on image type

---

## 🚧 Future Enhancements

Potential improvements:
- [ ] Real-time video depth extraction
- [ ] GPU acceleration for faster processing
- [ ] Semantic segmentation integration (face detection)
- [ ] Multi-scale analysis for varying object sizes
- [ ] Temporal consistency for video sequences
- [ ] Metric depth estimation (absolute distances)
- [ ] Adaptive baseline for stereo generation
- [ ] Integration with VR/AR frameworks

---

## 📝 License

Part of the Holographer's Workbench project.

---

## 🤝 Contributing

Found a bug or have a feature request? Please open an issue!

Want to improve the algorithm? Check out the core implementation in `workbench/processors/holographic_depth.py`.

---

**Happy depth analyzing! 🎨📊🔬**
