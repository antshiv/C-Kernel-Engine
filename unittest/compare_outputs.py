import numpy as np
import os

def compare_outputs():
    # Ensure build directory exists
    if not os.path.exists("build"):
        print("Error: 'build/' directory not found. Please run the C test first.")
        return

    # Load reference output from PyTorch
    try:
        output_ref = np.fromfile("build/output_ref.bin", dtype=np.float32)
        print(f"Loaded PyTorch reference output: {output_ref.shape} floats")
    except FileNotFoundError:
        print("Error: 'build/output_ref.bin' not found. Please run generate_reference_data.py.")
        return

    # Load C model output
    try:
        output_c = np.fromfile("build/c_output.bin", dtype=np.float32)
        print(f"Loaded C model output: {output_c.shape} floats")
    except FileNotFoundError:
        print("Error: 'build/c_output.bin' not found. Please run ./build/test_forward_pass.")
        return

    # Ensure shapes match
    if output_ref.shape != output_c.shape:
        print(f"Error: Output shapes mismatch! PyTorch: {output_ref.shape}, C: {output_c.shape}")
        return

    # Compute difference
    diff = np.abs(output_ref - output_c)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nComparison Results:")
    print(f"  Maximum absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference:    {mean_diff:.6e}")

    # Set a tolerance for floating-point comparisons
    tolerance = 1e-4 # Adjust as needed, based on precision requirements and algorithm differences

    if max_diff < tolerance:
        print(f"\n✅ PASSED: Outputs are within the tolerance of {tolerance:.1e}.")
    else:
        print(f"\n❌ FAILED: Outputs exceed the tolerance of {tolerance:.1e}.")
        # Optionally print a few differing values for debugging
        if max_diff > 1e-3: # Only print if difference is significant
            print("\nFirst 10 differing values:")
            for i in range(min(10, len(output_ref))):
                if diff[i] > tolerance:
                    print(f"  Index {i}: Ref={output_ref[i]:.6f}, C={output_c[i]:.6f}, Diff={diff[i]:.6e}")


if __name__ == "__main__":
    compare_outputs()
