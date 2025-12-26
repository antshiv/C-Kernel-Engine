import ctypes
import os
import argparse
import time

import torch
import torch.nn.functional as F

from lib_loader import load_lib


lib = load_lib("libckernel_vision.so", "libckernel_engine.so")

# void im2patch(const float *image, float *patches, int C, int H, int W, int P)
lib.im2patch.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, # C
    ctypes.c_int, # H
    ctypes.c_int, # W
    ctypes.c_int  # P
]
lib.im2patch.restype = None

# void patch2im(const float *d_patches, float *d_image, int C, int H, int W, int P)
lib.patch2im.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, # C
    ctypes.c_int, # H
    ctypes.c_int, # W
    ctypes.c_int  # P
]
lib.patch2im.restype = None


def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_c_im2patch(image: torch.Tensor, P: int) -> torch.Tensor:
    C, H, W = image.shape
    num_patches = (H // P) * (W // P)
    patch_dim = C * P * P
    
    patches_out = torch.empty((num_patches, patch_dim), dtype=torch.float32)
    
    lib.im2patch(
        tensor_to_ptr(image),
        tensor_to_ptr(patches_out),
        C, H, W, P
    )
    return patches_out


def run_c_patch2im(d_patches: torch.Tensor, C, H, W, P) -> torch.Tensor:
    d_image_out = torch.empty((C, H, W), dtype=torch.float32)
    
    lib.patch2im(
        tensor_to_ptr(d_patches),
        tensor_to_ptr(d_image_out),
        C, H, W, P
    )
    return d_image_out


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()

def test_im2patch(C=3, H=224, W=224, P=16):
    print(f"\n--- Testing im2patch (Image {H}x{W}, Patch {P}) ---")
    torch.manual_seed(42)
    image = torch.randn(C, H, W, dtype=torch.float32)

    # PyTorch reference using unfold
    # unfold(dimension, size, step)
    # To get non-overlapping patches:
    # 1. Unfold H: [C, H/P, W, P]
    # 2. Unfold W: [C, H/P, W/P, P, P]
    # 3. Permute and reshape
    t0 = time.time()
    unfold = torch.nn.Unfold(kernel_size=(P, P), stride=(P, P))
    # input must be [B, C, H, W]
    ref = unfold(image.unsqueeze(0)) # [1, C*P*P, num_patches]
    ref = ref.squeeze(0).transpose(0, 1) # [num_patches, C*P*P]
    t_torch = time.time() - t0

    t0 = time.time()
    out_c = run_c_im2patch(image, P)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"PyTorch time: {t_torch*1000:.3f} ms")
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("im2patch forward mismatch!")


def test_patch2im(C=3, H=224, W=224, P=16):
    print(f"\n--- Testing patch2im (Image {H}x{W}, Patch {P}) ---")
    torch.manual_seed(43)
    num_patches = (H // P) * (W // P)
    patch_dim = C * P * P
    d_patches = torch.randn(num_patches, patch_dim, dtype=torch.float32)

    # PyTorch reference using fold
    # fold(output_size, kernel_size, stride)
    t0 = time.time()
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=(P, P), stride=(P, P))
    # input must be [B, C*P*P, num_patches]
    ref = fold(d_patches.transpose(0, 1).unsqueeze(0)) # [1, C, H, W]
    ref = ref.squeeze(0)
    t_torch = time.time() - t0

    t0 = time.time()
    out_c = run_c_patch2im(d_patches, C, H, W, P)
    t_c = time.time() - t0

    diff = max_diff(out_c, ref)
    print(f"PyTorch time: {t_torch*1000:.3f} ms")
    print(f"C Kernel time: {t_c*1000:.3f} ms")
    print(f"Max diff: {diff:.2e}")

    if diff > 1e-7:
        raise AssertionError("patch2im backward mismatch!")


if __name__ == "__main__":
    test_im2patch()
    test_patch2im()
    
    # Test a non-multiple size just in case (though ViT usually uses multiples)
    test_im2patch(C=3, H=32, W=32, P=8)
    test_patch2im(C=3, H=32, W=32, P=8)
