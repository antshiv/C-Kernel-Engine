import torch
import torch.nn as nn
import os
import struct

# ==============================================================================
# 1. PyTorch Model Definition
# ==============================================================================
# This section defines a PyTorch model that mirrors the architecture implied by
# your C orchestration code (ck_layer_forward_rmsnorm_swiglu).

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        rstd = (var + self.eps).rsqrt()
        return x * rstd * self.weight

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        
        # Q, K, V, and Output projections
        self.wq = nn.Linear(config["hidden_size"], self.num_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(config["hidden_size"], self.num_kv_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(config["hidden_size"], self.num_kv_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.num_heads * self.head_dim, config["hidden_size"], bias=True)

    def forward(self, x):
        B, T, D = x.shape  # Batch, Tokens, Dim

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, v)

        # Combine heads and project out
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=True) # Combined gate and up projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(torch.nn.functional.silu(gate) * up)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SwiGLU(config["hidden_size"], config["intermediate_size"])
        self.attention_norm = RMSNorm(config["hidden_size"])
        self.ffn_norm = RMSNorm(config["hidden_size"])

    def forward(self, x):
        # Attention block
        h = x + self.attention(self.attention_norm(x))
        # FFN block
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config["num_hidden_layers"])
])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==============================================================================
# 2. Generate Reference Data
# ==============================================================================
def generate_and_export():
    # Ensure build directory exists
    if not os.path.exists("build"):
        os.makedirs("build")

    # --- Config from default.config.json ---
    config = {
        "num_hidden_layers": 2,
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_attention_heads": 8,
        "num_key_value_heads": 8, # MHA, not GQA
    }
    T = 16 # Context length for the test

    print("PyTorch Model Config:")
    print(f"  Layers: {config['num_hidden_layers']}")
    print(f"  Hidden Size (D): {config['hidden_size']}")
    print(f"  Intermediate Size (MLP): {config['intermediate_size']}")
    print(f"  Attention Heads: {config['num_attention_heads']}")
    print(f"  KV Heads: {config['num_key_value_heads']}")
    print(f"  Sequence Length (T): {T}")
    
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)

    # --- Instantiate Model and Input ---
    model = Transformer(config)
    model.eval() # Set to evaluation mode
    input_tensor = torch.randn(1, T, config["hidden_size"])

    # --- Run Forward Pass for Reference Output ---
    with torch.no_grad():
        output_ref = model(input_tensor)

    # --- Export Tensors to Binary Files ---
    print("\nExporting reference data to 'build/' directory...")

    # Export input and reference output
    input_tensor.numpy().tofile("build/input.bin")
    output_ref.numpy().tofile("build/output_ref.bin")
    print("  - Saved input.bin")
    print("  - Saved output_ref.bin")

    # Export weights in the precise order expected by the C test harness
    with open("build/weights.bin", "wb") as f:
        for i, layer in enumerate(model.layers):
            print(f"  - Exporting weights for Layer {i}...")
            
            # Note on memory layout:
            # PyTorch Linear layer weight is [out_features, in_features].
            # Your C GEMM kernel expects B matrix as [N, K], which is also [out, in].
            # So we can use the weights directly without transposing.
            
            # Layer {i}: RMSNorm before attention
            f.write(layer.attention_norm.weight.detach().numpy().tobytes())
            
            # Layer {i}: Attention weights and biases
            f.write(layer.attention.wq.weight.detach().numpy().tobytes())
            f.write(layer.attention.wq.bias.detach().numpy().tobytes())
            f.write(layer.attention.wk.weight.detach().numpy().tobytes())
            f.write(layer.attention.wk.bias.detach().numpy().tobytes())
            f.write(layer.attention.wv.weight.detach().numpy().tobytes())
            f.write(layer.attention.wv.bias.detach().numpy().tobytes())
            f.write(layer.attention.wo.weight.detach().numpy().tobytes())
            f.write(layer.attention.wo.bias.detach().numpy().tobytes())

            # Layer {i}: RMSNorm before FFN
            f.write(layer.ffn_norm.weight.detach().numpy().tobytes())

            # Layer {i}: SwiGLU MLP weights and biases
            f.write(layer.feed_forward.w1.weight.detach().numpy().tobytes())
            f.write(layer.feed_forward.w1.bias.detach().numpy().tobytes())
            f.write(layer.feed_forward.w2.weight.detach().numpy().tobytes())
            f.write(layer.feed_forward.w2.bias.detach().numpy().tobytes())

    print("  - Saved weights.bin")
    print("\nReference data generation complete.")

if __name__ == "__main__":
    generate_and_export()
