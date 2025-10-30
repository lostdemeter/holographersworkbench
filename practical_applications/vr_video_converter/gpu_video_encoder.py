#!/usr/bin/env python3
"""
GPU Video Encoder using PyNvVideoCodec
======================================

Encodes video directly from GPU memory using NVIDIA NVENC hardware encoder.
Eliminates GPU→CPU transfer bottleneck.

Phase 3C: Real-time VR encoding
"""

import numpy as np
import cupy as cp
import PyNvVideoCodec as nvc

class GPUVideoEncoder:
    """Encode video directly from GPU memory using NVENC."""
    
    def __init__(self, output_path, width, height, fps, gpu_id=0, bitrate='10M'):
        """
        Initialize GPU video encoder.
        
        Args:
            output_path: Output video file path
            width: Frame width
            height: Frame height
            fps: Frames per second
            gpu_id: GPU device ID
            bitrate: Target bitrate (e.g., '10M' for 10 Mbps)
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.gpu_id = gpu_id
        
        # Parse bitrate
        if isinstance(bitrate, str):
            if bitrate.endswith('M'):
                self.bitrate_bps = int(bitrate[:-1]) * 1000000
            elif bitrate.endswith('K'):
                self.bitrate_bps = int(bitrate[:-1]) * 1000
            else:
                self.bitrate_bps = int(bitrate)
        else:
            self.bitrate_bps = bitrate
        
        # Create encoder
        self._create_encoder()
        
        # Write to temporary H.264 file, then remux to MP4
        self.temp_h264_path = output_path + '.h264'
        self.h264_file = open(self.temp_h264_path, 'wb')
        
        print(f"GPUVideoEncoder initialized:")
        print(f"  Resolution: {width}×{height}")
        print(f"  FPS: {fps}")
        print(f"  Bitrate: {self.bitrate_bps / 1000000:.1f} Mbps")
        print(f"  GPU: {gpu_id}")
        print(f"  Output: {output_path}")
    
    def _create_encoder(self):
        """Create NVENC encoder instance."""
        # PyNvVideoCodec uses NV12 format
        fmt = 'NV12'
        
        # WORKAROUND: Use CPU input buffer, we'll transfer from GPU to CPU
        # The library doesn't seem to support direct GPU buffer input properly
        use_cpu_input = True
        
        # Encoder settings as kwargs
        # Disable B-frames for smooth playback (no frame reordering)
        encode_settings = {
            'codec': 'h264',
            'fps': str(self.fps),
            'bitrate': str(self.bitrate_bps),
            'gop': str(int(self.fps * 2)),  # GOP size = 2 seconds
            'bf': '0',  # No B-frames (prevents stuttering)
            'gpu': str(self.gpu_id)
        }
        
        try:
            self.encoder = nvc.CreateEncoder(
                self.width,
                self.height,
                fmt,
                use_cpu_input,
                **encode_settings
            )
            print(f"✅ NVENC encoder created successfully (using CPU input buffer)")
            print(f"   Note: Frames will be transferred GPU→CPU before encoding")
        except Exception as e:
            print(f"❌ Failed to create encoder: {e}")
            raise
    
    def encode_frame_gpu(self, frame_gpu):
        """
        Encode a frame from GPU memory.
        
        Args:
            frame_gpu: CuPy array (H, W, 3) in BGR format, uint8
                       Must be on GPU already
        
        Returns:
            bool: True if encoding succeeded
        """
        # Ensure frame is on GPU
        if not isinstance(frame_gpu, cp.ndarray):
            raise ValueError("Frame must be a CuPy array (on GPU)")
        
        # Ensure correct shape and type
        if frame_gpu.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape mismatch: expected ({self.height}, {self.width}, 3), got {frame_gpu.shape}")
        
        if frame_gpu.dtype != cp.uint8:
            frame_gpu = frame_gpu.astype(cp.uint8)
        
        # Convert BGR to NV12 (required by NVENC)
        nv12_frame = self._bgr_to_nv12_gpu(frame_gpu)
        
        # Encode frame
        try:
            # Transfer NV12 frame from GPU to CPU
            nv12_cpu = cp.asnumpy(nv12_frame)
            
            # Encode the frame (encoder will copy to GPU internally)
            packets = self.encoder.Encode(nv12_cpu)
            
            if packets:
                # packets is a list of bytes (integers)
                # Convert to bytes and write
                packet_bytes = bytes(packets)
                self.h264_file.write(packet_bytes)
                return True
            
            # No packets yet (waiting for GOP)
            return True
            
        except Exception as e:
            print(f"Encoding error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _bgr_to_nv12_gpu(self, bgr_gpu):
        """
        Convert BGR to NV12 format on GPU.
        
        NV12 is a YUV 4:2:0 format required by NVENC:
        - Y plane: full resolution
        - UV plane: half resolution, interleaved
        
        Args:
            bgr_gpu: CuPy array (H, W, 3) in BGR format
        
        Returns:
            CuPy array in NV12 format
        """
        # Extract BGR channels
        b = bgr_gpu[:, :, 0].astype(cp.float32)
        g = bgr_gpu[:, :, 1].astype(cp.float32)
        r = bgr_gpu[:, :, 2].astype(cp.float32)
        
        # Convert BGR to YUV (BT.709 color space for HD)
        # Y = 0.2126*R + 0.7152*G + 0.0722*B
        # U = -0.1146*R - 0.3854*G + 0.5*B + 128
        # V = 0.5*R - 0.4542*G - 0.0458*B + 128
        
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        u = -0.1146 * r - 0.3854 * g + 0.5 * b + 128
        v = 0.5 * r - 0.4542 * g - 0.0458 * b + 128
        
        # Clamp to [0, 255]
        y = cp.clip(y, 0, 255).astype(cp.uint8)
        u = cp.clip(u, 0, 255).astype(cp.uint8)
        v = cp.clip(v, 0, 255).astype(cp.uint8)
        
        # Downsample U and V to half resolution (4:2:0)
        u_half = u[::2, ::2]
        v_half = v[::2, ::2]
        
        # Interleave U and V
        uv = cp.stack([u_half, v_half], axis=-1).reshape(-1)
        
        # Pack into NV12 format: Y plane followed by interleaved UV plane
        nv12_size = self.height * self.width + (self.height // 2) * (self.width // 2) * 2
        nv12 = cp.zeros(nv12_size, dtype=cp.uint8)
        
        # Y plane
        nv12[:self.height * self.width] = y.ravel()
        
        # UV plane
        nv12[self.height * self.width:] = uv
        
        return nv12
    
    def finalize(self):
        """Flush encoder, close H.264 file, and remux to MP4."""
        print("Finalizing encoder...")
        
        # Flush remaining frames from encoder
        try:
            # EndEncode flushes remaining packets
            packets = self.encoder.EndEncode()
            if packets:
                packet_bytes = bytes(packets)
                self.h264_file.write(packet_bytes)
        except Exception as e:
            print(f"Flush error: {e}")
        
        # Close H.264 file
        self.h264_file.close()
        
        # Remux H.264 to MP4 with FFmpeg
        print(f"Remuxing H.264 to MP4...")
        import subprocess
        result = subprocess.run([
            'ffmpeg',
            '-y',  # Overwrite
            '-r', str(self.fps),  # Set framerate
            '-i', self.temp_h264_path,  # Input H.264 file
            '-c:v', 'copy',  # Copy video stream
            '-movflags', '+faststart',  # Enable streaming
            self.output_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg remux error:")
            print(result.stderr[-500:])
        else:
            # Remove temp H.264 file
            import os
            os.remove(self.temp_h264_path)
            print(f"✅ Video saved: {self.output_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


# Test function
def test_gpu_encoder():
    """Test GPU video encoder with synthetic frames."""
    import time
    
    print("="*70)
    print("Testing GPU Video Encoder")
    print("="*70)
    
    # Create test video
    width, height = 1920, 1080
    fps = 24
    num_frames = 120  # 5 seconds
    
    output_path = "/tmp/test_gpu_encode.mp4"
    
    print(f"\nCreating {num_frames} test frames...")
    
    with GPUVideoEncoder(output_path, width, height, fps) as encoder:
        start_time = time.time()
        
        for i in range(num_frames):
            # Create test frame on GPU
            frame_gpu = cp.random.randint(0, 255, (height, width, 3), dtype=cp.uint8)
            
            # Add frame number text effect
            frame_gpu[100:200, 100:500] = 255
            
            # Encode
            success = encoder.encode_frame_gpu(frame_gpu)
            
            if not success:
                print(f"Failed to encode frame {i}")
                break
            
            if (i + 1) % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = (i + 1) / elapsed
                print(f"  Encoded {i+1}/{num_frames} frames ({fps_actual:.1f} FPS)")
        
        total_time = time.time() - start_time
        avg_fps = num_frames / total_time
    
    print(f"\n✅ Encoding complete!")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    test_gpu_encoder()
