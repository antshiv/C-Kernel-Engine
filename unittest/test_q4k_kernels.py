"""
Q4_K Kernel Unit Tests

Tests dequantization and GEMM accuracy for Q4_K (GGML k-quant format).
Compares C kernel output against reference Python implementation.
"""
import ctypes
import sys
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "unittest"
for path in (ROOT, UNITS):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)

# Load the library
try:
    lib = load_lib("libckernel_quant.so", "libckernel_engine.so")
except Exception as e:
    print(f"Warning: Could not load quantization library: {e}")
    print("Run 'make libckernel_quant.so' first")
    sys.exit(0)

# Q4_K constants
QK_K = 256  # Weights per super-block
BLOCK_Q4_K_SIZE = 144  # Bytes per block


# ============================================================================
# Reference Python Implementation
# ============================================================================

def fp16_to_fp32(h: int) -> float:
    """Convert FP16 (uint16) to FP32."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF

    if exp == 0:
        if mant == 0:
            return (-1.0 if sign else 1.0) * 0.0
        # Denormalized
        return (-1.0 if sign else 1.0) * (mant / 1024.0) * (2.0 ** -14)
    elif exp == 31:
        if mant == 0:
            return float('-inf') if sign else float('inf')
        return float('nan')
    else:
        return (-1.0 if sign else 1.0) * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))


def unpack_q4_k_scales(scales: bytes) -> tuple:
    """Unpack 6-bit scales and mins from 12-byte packed array."""
    sc = [0] * 8
    m = [0] * 8

    # Simplified unpacking (matches ggml_quants.h)
    sc[0] = scales[0] & 0x3F
    sc[1] = (scales[0] >> 6) | ((scales[1] & 0x0F) << 2)
    sc[2] = (scales[1] >> 4) | ((scales[2] & 0x03) << 4)
    sc[3] = scales[2] >> 2
    sc[4] = scales[3] & 0x3F
    sc[5] = (scales[3] >> 6) | ((scales[4] & 0x0F) << 2)
    sc[6] = (scales[4] >> 4) | ((scales[5] & 0x03) << 4)
    sc[7] = scales[5] >> 2

    m[0] = scales[6] & 0x3F
    m[1] = (scales[6] >> 6) | ((scales[7] & 0x0F) << 2)
    m[2] = (scales[7] >> 4) | ((scales[8] & 0x03) << 4)
    m[3] = scales[8] >> 2
    m[4] = scales[9] & 0x3F
    m[5] = (scales[9] >> 6) | ((scales[10] & 0x0F) << 2)
    m[6] = (scales[10] >> 4) | ((scales[11] & 0x03) << 4)
    m[7] = scales[11] >> 2

    return sc, m


def dequant_q4_k_block_ref(block_data: bytes) -> np.ndarray:
    """Reference dequantization of a Q4_K block (256 floats)."""
    # Parse block header
    d_bits = struct.unpack('<H', block_data[0:2])[0]
    dmin_bits = struct.unpack('<H', block_data[2:4])[0]
    scales = block_data[4:16]
    qs = block_data[16:144]

    d = fp16_to_fp32(d_bits)
    dmin = fp16_to_fp32(dmin_bits)

    sc, m = unpack_q4_k_scales(scales)

    output = np.zeros(256, dtype=np.float32)

    # Process 8 sub-blocks
    for sub in range(8):
        scale = d * sc[sub]
        min_val = dmin * m[sub]

        for i in range(16):
            packed = qs[sub * 16 + i]
            q0 = (packed & 0x0F) - 8
            q1 = (packed >> 4) - 8

            output[sub * 32 + 2*i + 0] = scale * q0 + min_val
            output[sub * 32 + 2*i + 1] = scale * q1 + min_val

    return output


def create_random_q4_k_block() -> bytes:
    """Create a random Q4_K block for testing."""
    # Random scale and min (FP16 format)
    d = np.random.uniform(0.01, 0.5)
    dmin = np.random.uniform(0.0, 0.1)

    # Convert to FP16 bits (simplified)
    d_bits = np.float16(d).view(np.uint16)
    dmin_bits = np.float16(dmin).view(np.uint16)

    # Random sub-block scales (6-bit, 0-63)
    sc = np.random.randint(0, 64, 8, dtype=np.uint8)
    m = np.random.randint(0, 64, 8, dtype=np.uint8)

    # Pack 6-bit scales into 12 bytes (each scale uses 6 bits, packed across byte boundaries)
    # sc[0]: 6 bits in byte[0] bits 0-5
    # sc[1]: 2 bits in byte[0] bits 6-7, 4 bits in byte[1] bits 0-3
    # sc[2]: 4 bits in byte[1] bits 4-7, 2 bits in byte[2] bits 0-1
    # sc[3]: 6 bits in byte[2] bits 2-7
    # (same pattern repeats for sc[4-7] in bytes 3-5, and m[0-7] in bytes 6-11)
    scales = bytes([
        (sc[0] & 0x3F) | ((sc[1] & 0x03) << 6),
        ((sc[1] >> 2) & 0x0F) | ((sc[2] & 0x0F) << 4),
        ((sc[2] >> 4) & 0x03) | ((sc[3] & 0x3F) << 2),
        (sc[4] & 0x3F) | ((sc[5] & 0x03) << 6),
        ((sc[5] >> 2) & 0x0F) | ((sc[6] & 0x0F) << 4),
        ((sc[6] >> 4) & 0x03) | ((sc[7] & 0x3F) << 2),
        (m[0] & 0x3F) | ((m[1] & 0x03) << 6),
        ((m[1] >> 2) & 0x0F) | ((m[2] & 0x0F) << 4),
        ((m[2] >> 4) & 0x03) | ((m[3] & 0x3F) << 2),
        (m[4] & 0x3F) | ((m[5] & 0x03) << 6),
        ((m[5] >> 2) & 0x0F) | ((m[6] & 0x0F) << 4),
        ((m[6] >> 4) & 0x03) | ((m[7] & 0x3F) << 2),
    ])

    # Random 4-bit weights
    qs = np.random.randint(0, 256, 128, dtype=np.uint8).tobytes()

    # Pack block
    block = struct.pack('<H', d_bits) + struct.pack('<H', dmin_bits) + scales + qs
    return block


# ============================================================================
# Test Functions
# ============================================================================

def test_dequant_q4_k():
    """Test Q4_K dequantization accuracy."""
    np.random.seed(42)

    # Create test block
    block_data = create_random_q4_k_block()

    # Reference implementation
    ref_output = dequant_q4_k_block_ref(block_data)

    # C kernel
    try:
        lib.dequant_q4_k_row.argtypes = [
            ctypes.c_void_p,   # src
            ctypes.POINTER(ctypes.c_float),  # dst
            ctypes.c_size_t    # n_elements
        ]
        lib.dequant_q4_k_row.restype = None

        c_output = np.zeros(256, dtype=np.float32)
        block_arr = np.frombuffer(block_data, dtype=np.uint8)

        lib.dequant_q4_k_row(
            block_arr.ctypes.data_as(ctypes.c_void_p),
            c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(256)
        )

        diff = np.max(np.abs(c_output - ref_output))
        return diff <= 1e-5, diff
    except Exception as e:
        return False, str(e)


def test_gemv_q4_k():
    """Test Q4_K GEMV accuracy."""
    np.random.seed(42)

    M = 64   # Output size
    K = 256  # Input size (one Q4_K block per row)

    # Create random weights (M x K in Q4_K format)
    blocks = b''.join([create_random_q4_k_block() for _ in range(M)])

    # Random input vector
    x = np.random.randn(K).astype(np.float32)

    # Reference: dequantize all weights, then matmul
    W_fp32 = np.zeros((M, K), dtype=np.float32)
    for row in range(M):
        block_data = blocks[row * BLOCK_Q4_K_SIZE : (row + 1) * BLOCK_Q4_K_SIZE]
        W_fp32[row, :] = dequant_q4_k_block_ref(block_data)

    ref_y = W_fp32 @ x

    # C kernel
    try:
        lib.gemv_q4_k.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # y
            ctypes.c_void_p,                 # W
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.c_int,                    # M
            ctypes.c_int                     # K
        ]
        lib.gemv_q4_k.restype = None

        c_y = np.zeros(M, dtype=np.float32)
        blocks_arr = np.frombuffer(blocks, dtype=np.uint8)

        lib.gemv_q4_k(
            c_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            blocks_arr.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M),
            ctypes.c_int(K)
        )

        diff = np.max(np.abs(c_y - ref_y))
        # For Q4_K (4-bit quantized weights), tolerance of 1e-3 is reasonable
        # due to FP16 scale precision and accumulated rounding
        return diff <= 1e-3, diff
    except Exception as e:
        return False, str(e)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print_system_info()

    print("\n" + "=" * 70)
    print("  Q4_K Kernel Unit Tests")
    print("=" * 70)

    # Test dequantization
    print("\nTest 1: Q4_K Dequantization")
    passed, result = test_dequant_q4_k()
    if passed:
        print(f"  [PASS] max_diff = {result:.2e}")
    else:
        print(f"  [FAIL] {result}")

    # Test GEMV
    print("\nTest 2: Q4_K GEMV")
    passed, result = test_gemv_q4_k()
    if passed:
        print(f"  [PASS] max_diff = {result:.2e}")
    else:
        print(f"  [FAIL] {result}")

    print("\n" + "=" * 70)
