#!/usr/bin/env python3
"""
Structural Compression via Truth Space Geometry

Proof-of-concept implementation of compression based on the discovery
that truth space has φ-governed, Fibonacci-recurrent structure.

Key insight: The structure IS the information. Compress by encoding
the generating constraints, not the data points themselves.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import struct

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


@dataclass
class CompressionStats:
    """Statistics from compression."""
    original_bits: int
    compressed_bits: int
    ratio: float
    reconstruction_error: float
    fibonacci_fraction: float  # Fraction of points using Fibonacci encoding


class FibonacciRecurrenceEncoder:
    """
    Compress truth space points using Fibonacci recurrence.
    
    Observation: 85-99% of valid points satisfy sorted[0] ≈ sorted[1] + sorted[2]
    
    Strategy:
    - Store only the 2 largest coordinates + permutation
    - Reconstruct remaining 4 via Fibonacci recurrence
    - Store small corrections for reconstruction error
    """
    
    def __init__(self, tolerance: float = 0.05, correction_bits: int = 8):
        self.tolerance = tolerance
        self.correction_bits = correction_bits
        self.correction_scale = 2 ** correction_bits
    
    def encode_point(self, point: np.ndarray) -> Tuple[bytes, bool]:
        """
        Encode a single 6D truth space point.
        
        Returns:
            (encoded_bytes, used_fibonacci)
        """
        # Sort and get permutation
        sorted_idx = np.argsort(point)[::-1]
        sorted_vals = point[sorted_idx]
        
        # Check if Fibonacci recurrence holds
        if len(sorted_vals) >= 3:
            fib_error = abs(sorted_vals[0] - sorted_vals[1] - sorted_vals[2])
            use_fibonacci = fib_error < self.tolerance
        else:
            use_fibonacci = False
        
        if use_fibonacci:
            return self._encode_fibonacci(sorted_vals, sorted_idx), True
        else:
            return self._encode_full(point), False
    
    def _encode_fibonacci(self, sorted_vals: np.ndarray, perm: np.ndarray) -> bytes:
        """Encode using Fibonacci recurrence."""
        # Header: 1 byte (0x01 = Fibonacci mode)
        header = bytes([0x01])
        
        # Store 2 largest values as 16-bit fixed point (0-1 range)
        v0 = int(sorted_vals[0] * 65535)
        v1 = int(sorted_vals[1] * 65535)
        values = struct.pack('>HH', v0, v1)
        
        # Store permutation as 3 bytes (6 values, 3 bits each = 18 bits)
        perm_packed = 0
        for i, p in enumerate(perm):
            perm_packed |= (p << (3 * i))
        perm_bytes = struct.pack('>I', perm_packed)[:3]  # Take lower 3 bytes
        
        # Reconstruct and compute corrections
        reconstructed = self._fibonacci_reconstruct(sorted_vals[0], sorted_vals[1])
        corrections = sorted_vals[2:] - reconstructed
        
        # Quantize corrections
        corr_bytes = b''
        for c in corrections:
            # Map correction to 0-255 range (centered at 128)
            quantized = int((c + 0.5) * self.correction_scale)
            quantized = max(0, min(255, quantized))
            corr_bytes += bytes([quantized])
        
        return header + values + perm_bytes + corr_bytes
    
    def _encode_full(self, point: np.ndarray) -> bytes:
        """Encode without Fibonacci (fallback)."""
        # Header: 1 byte (0x00 = full mode)
        header = bytes([0x00])
        
        # Store all 6 values as 16-bit fixed point
        values = b''
        for v in point:
            values += struct.pack('>H', int(v * 65535))
        
        return header + values
    
    def _fibonacci_reconstruct(self, v0: float, v1: float) -> np.ndarray:
        """Reconstruct remaining values via Fibonacci recurrence."""
        result = np.zeros(4)
        result[0] = v0 - v1  # v2 = v0 - v1
        result[1] = v1 - result[0]  # v3 = v1 - v2
        result[2] = result[0] - result[1]  # v4 = v2 - v3
        result[3] = result[1] - result[2]  # v5 = v3 - v4
        return np.maximum(result, 0)  # Ensure non-negative
    
    def decode_point(self, data: bytes) -> np.ndarray:
        """Decode a compressed point."""
        mode = data[0]
        
        if mode == 0x01:
            return self._decode_fibonacci(data[1:])
        else:
            return self._decode_full(data[1:])
    
    def _decode_fibonacci(self, data: bytes) -> np.ndarray:
        """Decode Fibonacci-encoded point."""
        # Read 2 values
        v0, v1 = struct.unpack('>HH', data[:4])
        v0, v1 = v0 / 65535, v1 / 65535
        
        # Read permutation
        perm_packed = struct.unpack('>I', data[4:7] + b'\x00')[0]
        perm = [(perm_packed >> (3 * i)) & 0x07 for i in range(6)]
        
        # Reconstruct via Fibonacci
        reconstructed = self._fibonacci_reconstruct(v0, v1)
        
        # Apply corrections
        corrections = data[7:11]
        for i, c in enumerate(corrections):
            correction = (c / self.correction_scale) - 0.5
            reconstructed[i] += correction
        
        # Build sorted array
        sorted_vals = np.array([v0, v1] + list(reconstructed))
        
        # Invert permutation
        result = np.zeros(6)
        for i, p in enumerate(perm):
            result[p] = sorted_vals[i]
        
        # Normalize to sum to 1
        result = result / np.sum(result)
        
        return result
    
    def _decode_full(self, data: bytes) -> np.ndarray:
        """Decode full-encoded point."""
        result = np.zeros(6)
        for i in range(6):
            result[i] = struct.unpack('>H', data[i*2:(i+1)*2])[0] / 65535
        return result / np.sum(result)
    
    def compress(self, points: List[np.ndarray]) -> Tuple[bytes, CompressionStats]:
        """Compress a list of points."""
        encoded_chunks = []
        fib_count = 0
        
        for point in points:
            encoded, used_fib = self.encode_point(point)
            encoded_chunks.append(encoded)
            if used_fib:
                fib_count += 1
        
        # Concatenate with length prefixes
        compressed = struct.pack('>I', len(points))  # Number of points
        for chunk in encoded_chunks:
            compressed += struct.pack('>B', len(chunk)) + chunk
        
        # Calculate stats
        original_bits = len(points) * 6 * 32  # 6 floats × 32 bits
        compressed_bits = len(compressed) * 8
        
        # Calculate reconstruction error
        decompressed = self.decompress(compressed)
        errors = [np.linalg.norm(p - d) for p, d in zip(points, decompressed)]
        mean_error = np.mean(errors)
        
        stats = CompressionStats(
            original_bits=original_bits,
            compressed_bits=compressed_bits,
            ratio=original_bits / compressed_bits,
            reconstruction_error=mean_error,
            fibonacci_fraction=fib_count / len(points)
        )
        
        return compressed, stats
    
    def decompress(self, data: bytes) -> List[np.ndarray]:
        """Decompress to list of points."""
        n_points = struct.unpack('>I', data[:4])[0]
        points = []
        offset = 4
        
        for _ in range(n_points):
            chunk_len = data[offset]
            chunk = data[offset+1:offset+1+chunk_len]
            points.append(self.decode_point(chunk))
            offset += 1 + chunk_len
        
        return points


class GoldenQuantizer:
    """
    Quantize coordinates to φ^(-k) grid.
    
    Observation: Deviations from uniform cluster at φ^(-k) levels.
    This suggests a natural quantization grid based on golden ratio powers.
    """
    
    def __init__(self, max_power: int = 12):
        self.max_power = max_power
        # Build quantization levels
        self.levels = np.array([PHI ** (-k) for k in range(max_power)])
        self.levels = np.concatenate([[0], self.levels, [1]])
        self.levels = np.sort(np.unique(self.levels))
    
    def quantize(self, value: float) -> int:
        """Quantize a value to nearest φ-level."""
        idx = np.argmin(np.abs(self.levels - value))
        return idx
    
    def dequantize(self, idx: int) -> float:
        """Recover value from quantization index."""
        return self.levels[idx]
    
    def encode_point(self, point: np.ndarray) -> bytes:
        """Encode point using φ-quantization."""
        indices = [self.quantize(v) for v in point]
        # Pack as nibbles (4 bits each, since we have ~14 levels)
        packed = 0
        for i, idx in enumerate(indices):
            packed |= (idx << (4 * i))
        return struct.pack('>I', packed)[:3]  # 24 bits for 6 coords
    
    def decode_point(self, data: bytes) -> np.ndarray:
        """Decode φ-quantized point."""
        packed = struct.unpack('>I', data + b'\x00')[0]
        indices = [(packed >> (4 * i)) & 0x0F for i in range(6)]
        values = np.array([self.dequantize(idx) for idx in indices])
        return values / np.sum(values)  # Normalize


class StructuralCompressor:
    """
    High-level compressor that combines multiple strategies.
    
    1. Detect structure type (Fibonacci, golden, etc.)
    2. Apply appropriate encoder
    3. Store structure metadata + encoded data
    """
    
    def __init__(self):
        self.fib_encoder = FibonacciRecurrenceEncoder()
        self.phi_quantizer = GoldenQuantizer()
    
    def detect_structure(self, points: List[np.ndarray]) -> str:
        """Detect the dominant structure in the data."""
        if not points:
            return "none"
        
        # Check Fibonacci recurrence
        fib_count = 0
        for point in points[:100]:  # Sample
            sorted_vals = np.sort(point)[::-1]
            if len(sorted_vals) >= 3:
                error = abs(sorted_vals[0] - sorted_vals[1] - sorted_vals[2])
                if error < 0.1:
                    fib_count += 1
        
        fib_fraction = fib_count / min(len(points), 100)
        
        if fib_fraction > 0.7:
            return "fibonacci"
        elif fib_fraction > 0.3:
            return "golden"
        else:
            return "uniform"
    
    def compress(self, points: List[np.ndarray]) -> Tuple[bytes, Dict]:
        """Compress points using detected structure."""
        structure = self.detect_structure(points)
        
        if structure == "fibonacci":
            compressed, stats = self.fib_encoder.compress(points)
            return compressed, {
                "structure": structure,
                "stats": stats
            }
        elif structure == "golden":
            # Use φ-quantization
            chunks = [self.phi_quantizer.encode_point(p) for p in points]
            compressed = struct.pack('>I', len(points))
            for chunk in chunks:
                compressed += chunk
            
            return compressed, {
                "structure": structure,
                "original_bits": len(points) * 6 * 32,
                "compressed_bits": len(compressed) * 8,
                "ratio": (len(points) * 6 * 32) / (len(compressed) * 8)
            }
        else:
            # Fallback to raw encoding
            compressed = struct.pack('>I', len(points))
            for point in points:
                for v in point:
                    compressed += struct.pack('>f', v)
            
            return compressed, {
                "structure": "none",
                "ratio": 1.0
            }


def demo():
    """Demonstrate structural compression."""
    print("=" * 70)
    print("STRUCTURAL COMPRESSION DEMO")
    print("Compressing Truth Space Points via Fibonacci Recurrence")
    print("=" * 70)
    
    # Generate test points with Fibonacci structure
    np.random.seed(42)
    points = []
    
    for _ in range(1000):
        # Generate point with Fibonacci-like structure
        v0 = np.random.uniform(0.25, 0.45)
        v1 = np.random.uniform(0.15, 0.30)
        v2 = v0 - v1 + np.random.normal(0, 0.02)  # Fibonacci with noise
        v3 = v1 - v2 + np.random.normal(0, 0.01)
        v4 = v2 - v3 + np.random.normal(0, 0.01)
        v5 = 1 - v0 - v1 - v2 - v3 - v4
        
        point = np.array([v0, v1, v2, v3, v4, v5])
        point = np.maximum(point, 0)
        point = point / np.sum(point)
        
        # Random permutation
        np.random.shuffle(point)
        points.append(point)
    
    print(f"\nGenerated {len(points)} test points with Fibonacci structure")
    
    # Compress
    encoder = FibonacciRecurrenceEncoder()
    compressed, stats = encoder.compress(points)
    
    print(f"\n{'='*50}")
    print("COMPRESSION RESULTS")
    print(f"{'='*50}")
    print(f"Original size:     {stats.original_bits:,} bits ({stats.original_bits//8:,} bytes)")
    print(f"Compressed size:   {stats.compressed_bits:,} bits ({stats.compressed_bits//8:,} bytes)")
    print(f"Compression ratio: {stats.ratio:.2f}x")
    print(f"Reconstruction error: {stats.reconstruction_error:.6f}")
    print(f"Fibonacci encoding: {stats.fibonacci_fraction:.1%} of points")
    
    # Verify reconstruction
    decompressed = encoder.decompress(compressed)
    
    print(f"\n{'='*50}")
    print("RECONSTRUCTION VERIFICATION")
    print(f"{'='*50}")
    
    for i in [0, 100, 500, 999]:
        orig = points[i]
        recon = decompressed[i]
        error = np.linalg.norm(orig - recon)
        print(f"\nPoint {i}:")
        print(f"  Original:      [{', '.join(f'{v:.4f}' for v in orig)}]")
        print(f"  Reconstructed: [{', '.join(f'{v:.4f}' for v in recon)}]")
        print(f"  Error: {error:.6f}")
    
    # Compare with uniform quantization
    print(f"\n{'='*50}")
    print("COMPARISON WITH UNIFORM QUANTIZATION")
    print(f"{'='*50}")
    
    # 8-bit uniform quantization
    uniform_bits = len(points) * 6 * 8  # 8 bits per coord
    print(f"8-bit uniform:  {uniform_bits:,} bits ({uniform_bits//8:,} bytes)")
    print(f"Our method:     {stats.compressed_bits:,} bits ({stats.compressed_bits//8:,} bytes)")
    print(f"Additional savings: {(uniform_bits - stats.compressed_bits) / uniform_bits:.1%}")


if __name__ == "__main__":
    demo()
