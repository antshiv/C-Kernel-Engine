import ctypes
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from lib_loader import load_lib

# Load the vision library
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

def tensor_to_ptr(t: torch.Tensor):
    return t.contiguous().view(-1).numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def run_c_im2patch(image: torch.Tensor, P: int) -> torch.Tensor:
    C, H, W = image.shape
    num_patches = (H // P) * (W // P)
    patch_dim = C * P * P
    patches_out = torch.empty((num_patches, patch_dim), dtype=torch.float32)
    lib.im2patch(tensor_to_ptr(image), tensor_to_ptr(patches_out), C, H, W, P)
    return patches_out

# 1. Load the image
img_path = "Screenshots/Screenshot 2025-12-23 at 06-30-56 .png"
img = Image.open(img_path).convert('RGB')
W_orig, H_orig = img.size
print(f"Original size: {W_orig}x{H_orig}")

# 2. Pad to multiple of P=16
P = 16
H_new = ((H_orig + P - 1) // P) * P
W_new = ((W_orig + P - 1) // P) * P
print(f"New size: {W_new}x{H_new}")

# Simple padding with black pixels
img_new = Image.new('RGB', (W_new, H_new), (0, 0, 0))
img_new.paste(img, (0, 0))

# 3. Convert to tensor [C, H, W]
img_np = np.array(img_new).astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
C, H, W = img_tensor.shape

# 4. Run PyTorch Reference (Unfold)
unfold = torch.nn.Unfold(kernel_size=(P, P), stride=(P, P))
ref = unfold(img_tensor.unsqueeze(0)) # [1, C*P*P, num_patches]
ref = ref.squeeze(0).transpose(0, 1) # [num_patches, C*P*P]

# 5. Run C Kernel
out_c = run_c_im2patch(img_tensor, P)

# 6. Compare
diff = (out_c - ref).abs().max().item()
print(f"Comparison Result: Max Diff = {diff:.2e}")

if diff < 1e-7:
    print("SUCCESS: C im2patch exactly matches PyTorch unfold on real image data!")
else:
    print("FAILURE: Mismatch detected.")
    exit(1)

# 7. Bonus: Reconstruct a few patches back to images to see if they look right
# Patch 0 should be the top-left corner
patch_0 = out_c[0].view(C, P, P).permute(1, 2, 0).numpy() * 255.0
patch_0 = Image.fromarray(patch_0.astype(np.uint8))
patch_0.save("build/patch_top_left.png")
print("Saved build/patch_top_left.png for visual inspection.")
