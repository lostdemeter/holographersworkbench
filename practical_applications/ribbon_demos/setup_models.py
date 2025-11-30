#!/usr/bin/env python3
"""
Setup Script for Clock Downcaster
=================================

Downloads and caches required models for the examples.

Models:
1. Stable Diffusion v1.5 - For face generation examples

Usage:
    python setup_models.py              # Download all models
    python setup_models.py --diffusion  # Just Stable Diffusion
    python setup_models.py --check      # Check what's installed

Author: Lesley Gushurst
License: GPLv3
"""

import argparse
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check which dependencies are available."""
    print("Checking dependencies...")
    print("-" * 50)
    
    deps = {
        'numpy': False,
        'scipy': False,
        'mpmath': False,
        'torch': False,
        'diffusers': False,
        'transformers': False,
        'accelerate': False,
        'PIL': False,
    }
    
    for name in deps:
        try:
            if name == 'PIL':
                import PIL
            else:
                __import__(name)
            deps[name] = True
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}")
    
    print("-" * 50)
    
    # Core deps
    core_ok = all(deps[d] for d in ['numpy', 'scipy', 'mpmath'])
    print(f"\nCore (oracle, solver): {'✓ Ready' if core_ok else '✗ Missing deps'}")
    
    # Text generation
    print(f"Text generation: ✓ Ready (uses core only)")
    
    # Image generation
    image_ok = all(deps[d] for d in ['torch', 'diffusers', 'transformers', 'PIL'])
    print(f"Image generation: {'✓ Ready' if image_ok else '✗ Missing deps'}")
    
    if not image_ok:
        print("\n  To enable image generation:")
        print("  pip install torch diffusers transformers accelerate pillow")
    
    return deps


def download_stable_diffusion(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """Download and cache Stable Diffusion model."""
    print(f"\nDownloading Stable Diffusion: {model_id}")
    print("=" * 50)
    print("This will download ~5GB on first run.")
    print("The model will be cached in ~/.cache/huggingface/")
    print()
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall with: pip install torch diffusers transformers accelerate")
        return False
    
    print("Loading pipeline (this downloads the model)...")
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"  Device: {device}")
        print(f"  Dtype: {dtype}")
        
        # Download model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        
        # Use fast scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        
        print("\n✓ Model downloaded and cached!")
        print(f"  Location: ~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}")
        
        # Quick test
        print("\nRunning quick test...")
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        # Generate a tiny test image
        with torch.no_grad():
            result = pipe(
                prompt="test",
                num_inference_steps=1,
                height=64,
                width=64,
            )
        
        print("✓ Model works!")
        
        # Clean up
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def show_cache_info():
    """Show information about cached models."""
    print("\nCached Models")
    print("=" * 50)
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not cache_dir.exists():
        print("No HuggingFace cache found.")
        return
    
    # Find model directories
    models = list(cache_dir.glob("models--*"))
    
    if not models:
        print("No models cached.")
        return
    
    total_size = 0
    for model_dir in sorted(models):
        # Calculate size
        size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        total_size += size
        
        name = model_dir.name.replace("models--", "").replace("--", "/")
        print(f"  {name}: {size / 1e9:.2f} GB")
    
    print(f"\nTotal cache size: {total_size / 1e9:.2f} GB")
    print(f"Cache location: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Setup models for Clock Downcaster")
    parser.add_argument('--check', action='store_true', 
                       help='Check dependencies and cached models')
    parser.add_argument('--diffusion', action='store_true',
                       help='Download Stable Diffusion model')
    parser.add_argument('--model', type=str, default="runwayml/stable-diffusion-v1-5",
                       help='Model ID to download')
    parser.add_argument('--cache-info', action='store_true',
                       help='Show cached model information')
    args = parser.parse_args()
    
    print("=" * 50)
    print("CLOCK DOWNCASTER - Model Setup")
    print("=" * 50)
    
    if args.check or (not args.diffusion and not args.cache_info):
        deps = check_dependencies()
    
    if args.cache_info:
        show_cache_info()
    
    if args.diffusion:
        success = download_stable_diffusion(args.model)
        if success:
            print("\n" + "=" * 50)
            print("Setup complete! You can now run:")
            print("  python demo.py --mode image")
            print("=" * 50)
    
    if not args.diffusion and not args.cache_info:
        print("\n" + "-" * 50)
        print("To download Stable Diffusion model:")
        print("  python setup_models.py --diffusion")
        print("-" * 50)


if __name__ == "__main__":
    main()
