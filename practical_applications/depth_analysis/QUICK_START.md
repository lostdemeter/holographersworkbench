# Quick Start Guide - Holographic Depth Analyzer

## 🚀 Get Started in 30 Seconds

### Run a Demo

```bash
cd practical_applications/depth_analysis
python depth_analyzer.py --demo portrait
```

This will:
1. Generate a synthetic portrait image
2. Extract depth map using holographic signal processing
3. Show 12-panel component analysis
4. Generate stereoscopic 3D pair
5. Display all results interactively

### Analyze Your Own Image

```bash
python depth_analyzer.py --image /path/to/your/photo.jpg
```

### Save Results to Files

```bash
python depth_analyzer.py --demo portrait --save-output
```

Output files will be saved to `output/` directory:
- `demo_portrait_components.png` - 12-panel analysis
- `demo_portrait_stereo.png` - Stereoscopic pair

## 📊 What You'll See

### Component Analysis (12 panels)
1. Original image
2. Luminance depth (brightness-based)
3. Edge depth (gradient-based)
4. Frequency depth (high-pass filter)
5. Adaptive weighted fusion
6. Saliency map (spectral residual)
7. Center bias (compositional prior)
8. Final depth map with statistics
9. Depth histogram
10. Saliency-enhanced depth
11. Center-weighted depth
12. 3D surface plot

### Stereoscopic Pair (4 panels)
1. Original image
2. Depth map with colorbar
3. Left eye view
4. Right eye view

**Tip**: Use cross-eye viewing to see the 3D effect!

## 🎯 Common Use Cases

### Portrait Photography
```bash
python depth_analyzer.py --demo portrait --save-output
# Uses default parameters optimized for portraits
```

### Landscape Photography
```bash
python depth_analyzer.py --demo landscape --save-output
# Shows depth layers: sky, mountains, foreground
```

### Geometric Shapes
```bash
python depth_analyzer.py --demo geometric --save-output
# Tests on simple shapes at different depths
```

### All Demos at Once
```bash
python depth_analyzer.py --demo all --save-output
# Runs all three demos and saves results
```

## 🔧 Custom Output Directory

```bash
python depth_analyzer.py --image photo.jpg --save-output --output-dir my_results/
```

## 📖 Need More Help?

- **Full documentation**: See `README.md` in this directory
- **AI-readable code**: See `examples/demo_holographic_depth.py`
- **Technical details**: See `documents/HOLOGRAPHIC_DEPTH_SUMMARY.md`
- **Core implementation**: See `workbench/processors/holographic_depth.py`

## 💡 Pro Tips

1. **For portraits**: Use default parameters (already optimized)
2. **For landscapes**: Reduce center weight, increase adaptive weight
3. **For products**: Increase saliency weight, reduce smoothing
4. **Viewing stereo**: Start with cross-eye method, practice makes perfect!

## ⚡ Performance

- **512×512 image**: ~1.7 seconds (depth + stereo)
- **No GPU required**: Pure NumPy/SciPy implementation
- **No training data**: Mathematical approach, works immediately

---

**That's it! You're ready to analyze depth in 2D images. Happy analyzing! 🎨**
