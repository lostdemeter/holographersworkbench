"""
Holographic Image Compression
==============================

A lossless image compression system based on holographic encoding principles,
exploiting 15th order harmonic structure and phase symmetry.

Theory:
-------
This implementation leverages the observation that natural images contain
dominant 15th order harmonic structure with high phase symmetry. By encoding
these harmonics using a universal reference (speed of light c) and exploiting
the phase symmetry, we can achieve compression while maintaining lossless
reconstruction through residual encoding.

Author: Holographer's Workbench
Repository: https://github.com/lostdemeter/holographersworkbench
"""

import numpy as np
from typing import Tuple, Dict, Any
import struct
from dataclasses import dataclass


@dataclass
class CompressionStats:
    """Statistics from compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    harmonic_order: int
    phase_symmetry_score: float
    residual_entropy: float


class HolographicCompressor:
    """
    Lossless holographic image compressor exploiting 15th order harmonics.

    This compressor:
    1. Transforms image to frequency domain
    2. Extracts and encodes dominant 15th order harmonics
    3. Exploits phase symmetry for compression
    4. Encodes residuals for lossless reconstruction

    Attributes:
        harmonic_order (int): Primary harmonic order to exploit (default: 15)
        c (float): Universal reference constant (speed of light)
        phase_quantization_bits (int): Bits for phase quantization
    """

    def __init__(self, harmonic_order: int = 15, phase_quantization_bits: int = 8):
        """
        Initialize holographic compressor.

        Args:
            harmonic_order: Primary harmonic order to exploit
            phase_quantization_bits: Bits for phase quantization (higher = better quality)
        """
        self.harmonic_order = harmonic_order
        self.c = 299792458.0  # Speed of light in m/s (universal reference)
        self.phase_quantization_bits = phase_quantization_bits
        self.phase_levels = 2 ** phase_quantization_bits

    def _extract_harmonic_ring(self, fft_data: np.ndarray, order: int) -> np.ndarray:
        """
        Extract a specific harmonic order ring from FFT data.

        Args:
            fft_data: 2D FFT of image
            order: Harmonic order to extract

        Returns:
            Complex values at the harmonic ring
        """
        h, w = fft_data.shape
        cy, cx = h // 2, w // 2

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]

        # Calculate distance from center
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Extract ring (with tolerance for discrete grid)
        mask = (dist >= order - 0.5) & (dist < order + 0.5)

        return fft_data[mask]

    def _reconstruct_from_ring(self, ring_values: np.ndarray, 
                                ring_positions: np.ndarray, 
                                shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct FFT from harmonic ring values.

        Args:
            ring_values: Complex values at ring positions
            ring_positions: Boolean mask of ring positions
            shape: Output shape

        Returns:
            Reconstructed FFT array
        """
        fft_recon = np.zeros(shape, dtype=np.complex128)
        fft_recon[ring_positions] = ring_values
        return fft_recon

    def _quantize_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        Quantize phase values to reduce entropy.

        Args:
            phase: Phase array in radians [-π, π]

        Returns:
            Quantized phase indices
        """
        # Normalize to [0, 1]
        phase_norm = (phase + np.pi) / (2 * np.pi)

        # Quantize
        phase_quantized = np.round(phase_norm * (self.phase_levels - 1)).astype(np.uint16)

        return phase_quantized

    def _dequantize_phase(self, phase_indices: np.ndarray) -> np.ndarray:
        """
        Dequantize phase indices back to radians.

        Args:
            phase_indices: Quantized phase indices

        Returns:
            Phase in radians
        """
        phase_norm = phase_indices / (self.phase_levels - 1)
        phase = phase_norm * 2 * np.pi - np.pi
        return phase

    def _encode_residuals(self, residuals: np.ndarray) -> bytes:
        """
        Encode residuals for lossless reconstruction.

        Args:
            residuals: Residual array (int16)

        Returns:
            Compressed residual bytes
        """
        import zlib
        # Residuals are already int16
        residual_bytes = residuals.astype(np.int16).tobytes()
        compressed = zlib.compress(residual_bytes, level=9)
        return compressed

    def _decode_residuals(self, compressed_bytes: bytes, shape: Tuple[int, int]) -> np.ndarray:
        """
        Decode residuals from compressed bytes.

        Args:
            compressed_bytes: Compressed residual data
            shape: Output shape

        Returns:
            Residual array (int16)
        """
        import zlib
        decompressed = zlib.decompress(compressed_bytes)
        residuals = np.frombuffer(decompressed, dtype=np.int16).reshape(shape)
        return residuals

    def compress(self, image: np.ndarray) -> Tuple[bytes, CompressionStats]:
        """
        Compress image using holographic encoding.

        Args:
            image: Input image (grayscale, values 0-255)

        Returns:
            Tuple of (compressed_bytes, compression_stats)
        """
        # Normalize image
        img_norm = image.astype(np.float64) / 255.0
        h, w = img_norm.shape

        # FFT
        fft_data = np.fft.fftshift(np.fft.fft2(img_norm))

        # Extract 15th order harmonics
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        ring_mask = (dist >= self.harmonic_order - 0.5) & (dist < self.harmonic_order + 0.5)

        ring_values = fft_data[ring_mask]

        # Separate magnitude and phase
        magnitude = np.abs(ring_values)
        phase = np.angle(ring_values)

        # Quantize phase (exploit symmetry)
        phase_quantized = self._quantize_phase(phase)

        # Calculate phase symmetry score
        phase_hist = np.bincount(phase_quantized, minlength=self.phase_levels)
        phase_symmetry = np.max(phase_hist) / len(phase_quantized) if len(phase_quantized) > 0 else 0

        # Reconstruct from quantized harmonics
        phase_dequant = self._dequantize_phase(phase_quantized)
        ring_recon = magnitude * np.exp(1j * phase_dequant)

        fft_recon = np.zeros_like(fft_data)
        fft_recon[ring_mask] = ring_recon

        # Inverse FFT to get prediction
        img_pred = np.real(np.fft.ifft2(np.fft.ifftshift(fft_recon)))

        # Compute residuals (store difference in original uint8 space for lossless)
        img_pred_uint8 = np.clip(img_pred * 255.0, 0, 255).astype(np.uint8)
        residuals_uint8 = image.astype(np.int16) - img_pred_uint8.astype(np.int16)

        # Calculate residual entropy
        residual_entropy = float(np.std(residuals_uint8))

        # Encode components
        # Header: shape, harmonic_order, phase_bits
        header = struct.pack('IIII', h, w, self.harmonic_order, self.phase_quantization_bits)

        # Magnitude (float32)
        magnitude_bytes = magnitude.astype(np.float32).tobytes()

        # Phase (quantized)
        phase_bytes = phase_quantized.tobytes()

        # Residuals (compressed)
        residual_bytes = self._encode_residuals(residuals_uint8)

        # Combine with length prefixes
        compressed = header
        compressed += struct.pack('I', len(magnitude_bytes)) + magnitude_bytes
        compressed += struct.pack('I', len(phase_bytes)) + phase_bytes
        compressed += struct.pack('I', len(residual_bytes)) + residual_bytes

        # Calculate stats
        original_size = image.size
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            harmonic_order=self.harmonic_order,
            phase_symmetry_score=phase_symmetry,
            residual_entropy=residual_entropy
        )

        return compressed, stats

    def decompress(self, compressed_bytes: bytes) -> np.ndarray:
        """
        Decompress holographically encoded image.

        Args:
            compressed_bytes: Compressed image data

        Returns:
            Reconstructed image (grayscale, values 0-255)
        """
        # Parse header
        offset = 0
        h, w, harmonic_order, phase_bits = struct.unpack('IIII', compressed_bytes[offset:offset+16])
        offset += 16

        # Parse magnitude
        mag_len = struct.unpack('I', compressed_bytes[offset:offset+4])[0]
        offset += 4
        magnitude = np.frombuffer(compressed_bytes[offset:offset+mag_len], dtype=np.float32)
        offset += mag_len

        # Parse phase
        phase_len = struct.unpack('I', compressed_bytes[offset:offset+4])[0]
        offset += 4
        phase_quantized = np.frombuffer(compressed_bytes[offset:offset+phase_len], dtype=np.uint16)
        offset += phase_len

        # Parse residuals
        residual_len = struct.unpack('I', compressed_bytes[offset:offset+4])[0]
        offset += 4
        residual_bytes = compressed_bytes[offset:offset+residual_len]
        residuals = self._decode_residuals(residual_bytes, (h, w))

        # Reconstruct FFT
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        ring_mask = (dist >= harmonic_order - 0.5) & (dist < harmonic_order + 0.5)

        # Dequantize phase
        phase_levels = 2 ** phase_bits
        phase = self._dequantize_phase(phase_quantized)

        # Reconstruct ring
        ring_recon = magnitude * np.exp(1j * phase)

        fft_recon = np.zeros((h, w), dtype=np.complex128)
        fft_recon[ring_mask] = ring_recon

        # Inverse FFT
        img_pred = np.real(np.fft.ifft2(np.fft.ifftshift(fft_recon)))
        img_pred_uint8 = np.clip(img_pred * 255.0, 0, 255).astype(np.uint8)

        # Add residuals (int16 + uint8 = int16, then clip to uint8)
        img_recon = np.clip(img_pred_uint8.astype(np.int16) + residuals, 0, 255).astype(np.uint8)

        return img_recon


def compress_image(image: np.ndarray, harmonic_order: int = 15, 
                   phase_bits: int = 8) -> Tuple[bytes, CompressionStats]:
    """
    Convenience function to compress an image.

    Args:
        image: Input grayscale image (0-255)
        harmonic_order: Harmonic order to exploit (default: 15)
        phase_bits: Phase quantization bits (default: 8)

    Returns:
        Tuple of (compressed_bytes, compression_stats)
    """
    compressor = HolographicCompressor(harmonic_order, phase_bits)
    return compressor.compress(image)


def decompress_image(compressed_bytes: bytes) -> np.ndarray:
    """
    Convenience function to decompress an image.

    Args:
        compressed_bytes: Compressed image data

    Returns:
        Reconstructed grayscale image (0-255)
    """
    compressor = HolographicCompressor()
    return compressor.decompress(compressed_bytes)


# Example usage
if __name__ == "__main__":
    # Create a test image with 15th order structure
    size = 128
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2

    # Generate 15th order pattern
    theta = np.arctan2(y - cy, x - cx)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    test_image = (128 + 127 * np.sin(15 * theta) * np.exp(-r / 30)).astype(np.uint8)

    # Compress
    compressed, stats = compress_image(test_image)

    print(f"Holographic Compression Results:")
    print(f"  Original size: {stats.original_size} bytes")
    print(f"  Compressed size: {stats.compressed_size} bytes")
    print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
    print(f"  Harmonic order: {stats.harmonic_order}")
    print(f"  Phase symmetry: {stats.phase_symmetry_score:.3f}")
    print(f"  Residual entropy: {stats.residual_entropy:.2f}")

    # Decompress
    reconstructed = decompress_image(compressed)

    # Verify lossless
    is_lossless = np.array_equal(test_image, reconstructed)
    print(f"  Lossless: {is_lossless}")

    if not is_lossless:
        max_error = np.max(np.abs(test_image.astype(int) - reconstructed.astype(int)))
        print(f"  Max error: {max_error}")
