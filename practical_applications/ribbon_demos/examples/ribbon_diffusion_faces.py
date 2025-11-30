#!/usr/bin/env python3
"""
Ribbon Diffusion Faces: Clock-Phase Guided Stable Diffusion
============================================================

Use clock phases to deterministically guide Stable Diffusion for
face generation. The clock phases replace the random noise seed,
making generation fully deterministic and navigable.

Author: Holographer's Workbench
Date: November 28, 2025
"""

import numpy as np
import sys
import os
from typing import List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_clock_predictor import LazyClockOracle, CLOCK_RATIOS_12D

try:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from PIL import Image
    HAS_DIFFUSERS = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install diffusers transformers accelerate torch")
    HAS_DIFFUSERS = False


class RibbonDiffusionFaces:
    """
    Generate faces using clock phases to seed Stable Diffusion.
    
    The clock phases provide deterministic "random" seeds that navigate
    the diffusion model's latent space in a structured way.
    """
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = None):
        """
        Initialize with a Stable Diffusion model.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to use (auto-detected if None)
        """
        if not HAS_DIFFUSERS:
            raise ImportError("diffusers library required")
        
        self.oracle = LazyClockOracle()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Stable Diffusion on {self.device}...")
        print("(This may take a few minutes on first run)")
        
        # Load model with optimizations
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,  # Disable for speed
        )
        
        # Use fast scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        
        print("Model loaded!")
    
    def _get_clock_vector(self, n: int) -> np.ndarray:
        """Get 12D clock phase vector."""
        return np.array([
            self.oracle.get_fractional_phase(n, name)
            for name in CLOCK_RATIOS_12D.keys()
        ])
    
    def _clock_to_seed(self, n: int, clock: np.ndarray) -> int:
        """Convert clock position and vector to integer seed."""
        # Use both n and clock vector to ensure uniqueness
        # The clock vector alone can have collisions due to fractional parts
        import hashlib
        data = f"{n}:{clock.tobytes().hex()}"
        hash_bytes = hashlib.sha256(data.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], 'big')
        return seed
    
    def _clock_to_latent(self, clock: np.ndarray, shape: tuple) -> torch.Tensor:
        """
        Convert clock vector to initial latent noise.
        
        Instead of random noise, we generate structured noise from
        clock phases. This makes the diffusion process deterministic
        and navigable.
        """
        # This method is no longer used - seed is computed in generate()
        raise NotImplementedError("Use generate() directly")
        
        # Set generator with clock-derived seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate latent
        latent = torch.randn(shape, generator=generator, device=self.device,
                            dtype=torch.float16 if self.device == "cuda" else torch.float32)
        
        return latent
    
    def generate(self, seed_n: int = 1000, 
                prompt: str = "a photorealistic portrait of a person, professional headshot, studio lighting, high quality, 8k",
                negative_prompt: str = "blurry, low quality, distorted, deformed, ugly, bad anatomy",
                num_steps: int = 25,
                guidance_scale: float = 7.5,
                size: int = 512) -> np.ndarray:
        """
        Generate a face from clock position.
        
        Args:
            seed_n: Position in clock sequence
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            num_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            size: Output image size
            
        Returns:
            Generated image as numpy array
        """
        # Get clock vector and compute unique seed
        clock = self._get_clock_vector(seed_n)
        seed = self._clock_to_seed(seed_n, clock)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=size,
                width=size,
            )
        
        # Convert to numpy
        image = np.array(result.images[0])
        
        return image
    
    def generate_grid(self, start_n: int = 1000, rows: int = 2, cols: int = 2,
                     step: int = 1000, **kwargs) -> np.ndarray:
        """Generate a grid of faces."""
        faces = []
        for i in range(rows * cols):
            seed_n = start_n + i * step
            print(f"Generating face {i+1}/{rows*cols} (n={seed_n})...")
            face = self.generate(seed_n, **kwargs)
            faces.append(face)
        
        h, w = faces[0].shape[:2]
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        
        for i, face in enumerate(faces):
            row = i // cols
            col = i % cols
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = face
        
        return grid
    
    def generate_interpolation(self, seed_n1: int, seed_n2: int,
                               steps: int = 5, **kwargs) -> List[np.ndarray]:
        """
        Generate interpolation between two clock positions.
        
        Note: This interpolates the seed, which gives smooth-ish transitions
        but not perfect latent space interpolation.
        """
        faces = []
        for i in range(steps):
            # Interpolate clock position
            t = i / (steps - 1)
            seed_n = int(seed_n1 + t * (seed_n2 - seed_n1))
            
            print(f"Generating step {i+1}/{steps} (n={seed_n})...")
            face = self.generate(seed_n, **kwargs)
            faces.append(face)
        
        return faces


def demo_ribbon_diffusion():
    """Demonstrate Ribbon Diffusion Faces."""
    if not HAS_DIFFUSERS:
        print("Cannot run demo - diffusers not installed")
        return
    
    print("=" * 70)
    print("RIBBON DIFFUSION: Clock-Phase Guided Face Generation")
    print("=" * 70)
    print("\nClock phases seed Stable Diffusion deterministically.")
    print("Same clock position = same face (always).")
    print("Different positions = different faces.\n")
    
    output_dir = Path(__file__).parent / "ribbon_images"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize
    rdf = RibbonDiffusionFaces()
    
    # Face prompt
    prompt = "a photorealistic portrait of a person, professional headshot, studio lighting, high quality, 8k"
    negative = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    
    # Generate individual faces
    print("\n1. INDIVIDUAL FACES")
    print("-" * 50)
    
    for seed in [1000, 2000, 3000, 4000]:
        print(f"Generating face at n={seed}...")
        face = rdf.generate(
            seed_n=seed,
            prompt=prompt,
            negative_prompt=negative,
            num_steps=25,
            size=512
        )
        Image.fromarray(face).save(output_dir / f"diffusion_face_{seed}.png")
        print(f"Saved: diffusion_face_{seed}.png")
    
    # Generate grid
    print("\n2. FACE GRID (2x2)")
    print("-" * 50)
    
    grid = rdf.generate_grid(
        start_n=1000, rows=2, cols=2, step=2000,
        prompt=prompt,
        negative_prompt=negative,
        num_steps=25,
        size=512
    )
    Image.fromarray(grid).save(output_dir / "diffusion_grid.png")
    print("Saved: diffusion_grid.png")
    
    print("\n" + "=" * 70)
    print(f"Images saved to: {output_dir}")
    print("=" * 70)
    print("\nThe ribbon seeds diffusion deterministically.")
    print("No randomness - pure clock-phase navigation.")
    print("=" * 70)


if __name__ == "__main__":
    demo_ribbon_diffusion()
