"""
Q4_K Quantizer Sanity Test

Validates that the offline Python Q4_K encoder produces bytes that the C
dequantizer (`dequant_q4_k_row`) can consume, and that the round-trip error is
within a reasonable bound (this is 4-bit quantization; it is not expected to be
exact).
"""

import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "unittest"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, UNITS, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from lib_loader import load_lib
from q4_k_quantize import quantize_q4_k_row
from test_utils import print_system_info


def dequant_q4_k_row_c(lib, src_bytes: bytes, n_elements: int) -> np.ndarray:
    out = np.zeros(n_elements, dtype=np.float32)
    buf = ctypes.create_string_buffer(src_bytes, len(src_bytes))
    lib.dequant_q4_k_row(
        ctypes.cast(buf, ctypes.c_void_p),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(n_elements),
    )
    return out


def main():
    print_system_info()

    try:
        lib = load_lib("libckernel_quant.so", "libckernel_engine.so")
    except Exception as exc:
        print(f"Warning: Could not load quantization library: {exc}")
        print("Run 'make libckernel_quant.so' first")
        sys.exit(0)

    lib.dequant_q4_k_row.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.dequant_q4_k_row.restype = None

    np.random.seed(0)

    # 1) Exact-zero should remain exact-zero.
    z = np.zeros(256, dtype=np.float32)
    qz = quantize_q4_k_row(z)
    dz = dequant_q4_k_row_c(lib, qz, 256)
    assert float(np.max(np.abs(dz))) == 0.0

    # 2) A structured ramp catches packing/order issues.
    for n in (256, 512, 1024):
        x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        q = quantize_q4_k_row(x)
        assert len(q) == (n // 256) * 144
        y = dequant_q4_k_row_c(lib, q, n)
        max_diff = float(np.max(np.abs(y - x)))
        # Loose bound: we only want to catch "clearly wrong" packing/dequant.
        assert max_diff < 0.25, f"max_diff too large for n={n}: {max_diff}"

    # 3) Random values in a small range should quantize with modest error.
    x = (np.random.randn(1024).astype(np.float32) * 0.1).astype(np.float32)
    q = quantize_q4_k_row(x)
    y = dequant_q4_k_row_c(lib, q, x.size)
    max_diff = float(np.max(np.abs(y - x)))
    assert max_diff < 0.1, f"unexpectedly large max_diff: {max_diff}"

    print("Q4_K quantizer sanity: PASS")


if __name__ == "__main__":
    main()

