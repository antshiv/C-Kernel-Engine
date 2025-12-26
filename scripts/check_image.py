from PIL import Image
import torch
import numpy as np
import os

img_path = "Screenshots/Screenshot 2025-12-23 at 06-30-56 .png"
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found")
    exit(1)

img = Image.open(img_path).convert('RGB')
print(f"Image mode: {img.mode}, size: {img.size}") # (W, H)

# Convert to numpy then to torch tensor [C, H, W]
img_np = np.array(img).astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
print(f"Tensor shape: {img_tensor.shape}")