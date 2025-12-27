#!/usr/bin/env python3
"""
Kernel Coverage Report Tool

Parses src/kernels/*.c to detect which data types are supported for each kernel.
Outputs a formatted table showing forward/backward coverage.

Usage:
    python3 scripts/kernel_coverage.py
    python3 scripts/kernel_coverage.py --json
    python3 scripts/kernel_coverage.py --markdown
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Data types to detect
DTYPES = ["FP32", "BF16", "FP16", "INT8", "INT4", "Q4_0", "Q4_K", "Q8_0"]

# Patterns to detect dtype variants in function names
DTYPE_PATTERNS = {
    "BF16": [r"_bf16", r"bf16_"],
    "FP16": [r"_f16(?![0-9])", r"f16_(?![0-9])"],
    "INT8": [r"_int8", r"int8_"],
    "INT4": [r"_int4", r"int4_"],
    "Q4_0": [r"_q4_0", r"q4_0_", r"q4_0\b"],
    "Q4_K": [r"_q4_k", r"q4_k_", r"q4k_", r"_q4k\b"],
    "Q8_0": [r"_q8_0", r"q8_0_", r"q8_0\b"],
}

# Kernel categories and their detection patterns
KERNEL_CATEGORIES = {
    "GEMM/GEMV": {
        "patterns": [r"\bgemm_", r"\bgemv_", r"gemm\b", r"gemv\b"],
        "files": ["gemm_kernels"],
    },
    "RMSNorm": {
        "patterns": [r"\brmsnorm_"],
        "files": ["rmsnorm_kernels"],
    },
    "LayerNorm": {
        "patterns": [r"\blayernorm_", r"\blayer_norm_"],
        "files": ["layernorm_kernels"],
    },
    "Attention": {
        "patterns": [r"\battention_"],
        "files": ["attention_kernels"],
    },
    "SwiGLU": {
        "patterns": [r"\bswiglu_"],
        "files": ["swiglu_kernels"],
    },
    "RoPE": {
        "patterns": [r"\brope_"],
        "files": ["rope_kernels"],
    },
    "GELU": {
        "patterns": [r"\bgelu_"],
        "files": ["gelu_kernels"],
    },
    "Softmax": {
        "patterns": [r"\bsoftmax_"],
        "files": ["softmax_kernels"],
    },
    "Sigmoid": {
        "patterns": [r"\bsigmoid_"],
        "files": ["sigmoid_kernels"],
    },
    "ReLU": {
        "patterns": [r"\brelu_"],
        "files": ["relu_kernels"],
    },
    "Embedding": {
        "patterns": [r"\bembedding_"],
        "files": ["embedding_kernels"],
    },
    "MLP": {
        "patterns": [r"\bmlp_", r"\bfc1_", r"\bfc2_"],
        "files": ["mlp_kernels"],
    },
    "Dequant": {
        "patterns": [r"\bdequant_"],
        "files": ["dequant_kernels"],
    },
    "Vision": {
        "patterns": [r"\bvision_", r"\bpatch_"],
        "files": ["vision_kernels"],
    },
}


def find_kernel_dir():
    """Find the kernels directory."""
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    kernel_dir = root / "src" / "kernels"
    if kernel_dir.exists():
        return kernel_dir
    # Try from cwd
    kernel_dir = Path("src/kernels")
    if kernel_dir.exists():
        return kernel_dir
    return None


def extract_functions(file_path: Path) -> list:
    """Extract function definitions from a C file."""
    content = file_path.read_text()

    functions = []

    # Pattern 1: Single-line function definitions
    # Looks for: return_type function_name(args) {
    pattern1 = r'^(?:static\s+)?(?:inline\s+)?(?:void|float|int|size_t|double|__m\d+\w*)\s+(\w+)\s*\([^)]*\)\s*\{'
    for match in re.finditer(pattern1, content, re.MULTILINE):
        functions.append(match.group(1).lower())

    # Pattern 2: Multi-line function definitions
    # Match: return_type function_name( at start of line (args may span lines)
    pattern2 = r'^(?:static\s+)?(?:inline\s+)?(?:void|float|int|size_t|double|__m\d+\w*)\s+(\w+)\s*\('
    for match in re.finditer(pattern2, content, re.MULTILINE):
        func_name = match.group(1).lower()
        if func_name not in functions:
            # Look for closing ) followed by { within next 800 chars
            start = match.end()
            snippet = content[start:start + 800]
            # Find ) followed by { (possibly with newlines/spaces between)
            if re.search(r'\)\s*\{', snippet, re.DOTALL):
                functions.append(func_name)

    return list(set(functions))  # Remove duplicates


def detect_dtype(func_name: str, file_name: str) -> str:
    """Detect the data type from function name or file name."""
    combined = func_name + " " + file_name

    for dtype, patterns in DTYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return dtype

    return "FP32"  # Default


def detect_direction(func_name: str, category: str = None) -> tuple:
    """Detect if function is forward, backward, or both.

    Special cases:
    - GEMM: gemm_nn and gemm_tn are backward variants (for dX and dW computation)
    - Functions with 'backward', '_bwd', '_grad' are backward
    """
    has_forward = False
    has_backward = False

    # Check for explicit backward naming
    if "backward" in func_name or "_bwd" in func_name or "_grad" in func_name:
        has_backward = True
    # GEMM special case: gemm_nn (C = A @ B) and gemm_tn (C = A.T @ B) are used for backward
    elif category == "GEMM/GEMV" and ("_nn" in func_name or "_tn" in func_name):
        has_backward = True
    else:
        has_forward = True

    return has_forward, has_backward


def analyze_kernels(kernel_dir: Path) -> dict:
    """Analyze all kernel files and build coverage map."""
    coverage = defaultdict(lambda: defaultdict(lambda: {"forward": False, "backward": False}))

    for c_file in kernel_dir.glob("*.c"):
        file_stem = c_file.stem
        functions = extract_functions(c_file)

        for func_name in functions:
            # Skip internal/helper functions
            if func_name.startswith("_") or "helper" in func_name:
                continue

            # Find which category this function belongs to
            for category, info in KERNEL_CATEGORIES.items():
                # Check if this file is relevant to this category
                file_matches = any(f in file_stem for f in info["files"])

                # Check if function name matches category patterns
                func_matches = any(re.search(p, func_name) for p in info["patterns"])

                if file_matches or func_matches:
                    dtype = detect_dtype(func_name, file_stem)
                    is_forward, is_backward = detect_direction(func_name, category)

                    if is_forward:
                        coverage[category][dtype]["forward"] = True
                    if is_backward:
                        coverage[category][dtype]["backward"] = True

    return coverage


def format_cell(has_forward: bool, has_backward: bool) -> str:
    """Format a cell for the table."""
    if has_forward and has_backward:
        return "F/B"
    elif has_forward:
        return "F"
    elif has_backward:
        return "B"
    else:
        return "-"


def print_table(coverage: dict):
    """Print a formatted ASCII table."""
    # Header
    col_widths = [12] + [6] * len(DTYPES)

    header = "| {:<12} |".format("Kernel")
    for dtype in DTYPES:
        header += " {:^5} |".format(dtype)

    separator = "+" + "-" * 14 + "+" + (("-" * 7 + "+") * len(DTYPES))

    print("\n" + "=" * 70)
    print("  Kernel Data Type Coverage Report")
    print("=" * 70)
    print()
    print("Legend: F = Forward, B = Backward, F/B = Both, - = Not implemented")
    print()
    print(separator)
    print(header)
    print(separator)

    # Sort categories for consistent output
    sorted_categories = sorted(coverage.keys())

    for category in sorted_categories:
        row = "| {:<12} |".format(category[:12])
        for dtype in DTYPES:
            info = coverage[category].get(dtype, {"forward": False, "backward": False})
            cell = format_cell(info["forward"], info["backward"])
            row += " {:^5} |".format(cell)
        print(row)

    print(separator)

    # Summary
    total_possible = len(coverage) * len(DTYPES) * 2  # forward + backward for each
    total_implemented = sum(
        1 for cat in coverage.values()
        for dtype_info in cat.values()
        for direction in ["forward", "backward"]
        if dtype_info.get(direction, False)
    )

    print()
    print(f"Coverage: {total_implemented} / {total_possible} ({100*total_implemented/total_possible:.1f}%)")
    print()


def print_markdown(coverage: dict):
    """Print markdown table."""
    print("\n## Kernel Data Type Coverage\n")
    print("Legend: F = Forward, B = Backward, F/B = Both, - = Not implemented\n")

    # Header
    header = "| Kernel |"
    separator = "|--------|"
    for dtype in DTYPES:
        header += f" {dtype} |"
        separator += "------|"

    print(header)
    print(separator)

    for category in sorted(coverage.keys()):
        row = f"| {category} |"
        for dtype in DTYPES:
            info = coverage[category].get(dtype, {"forward": False, "backward": False})
            cell = format_cell(info["forward"], info["backward"])
            row += f" {cell} |"
        print(row)


def print_json(coverage: dict):
    """Print JSON output."""
    output = {}
    for category, dtypes in coverage.items():
        output[category] = {}
        for dtype, info in dtypes.items():
            output[category][dtype] = info
    print(json.dumps(output, indent=2))


def print_missing(coverage: dict):
    """Print what's missing."""
    print("\n" + "=" * 70)
    print("  Missing Implementations")
    print("=" * 70)

    # Priority targets
    priority_dtypes = ["Q4_K", "BF16", "FP16"]
    priority_kernels = ["GEMM/GEMV", "RMSNorm", "Attention", "SwiGLU", "RoPE"]

    missing = []
    for kernel in priority_kernels:
        if kernel not in coverage:
            for dtype in priority_dtypes:
                missing.append(f"{kernel} ({dtype})")
        else:
            for dtype in priority_dtypes:
                info = coverage[kernel].get(dtype, {"forward": False, "backward": False})
                if not info["forward"]:
                    missing.append(f"{kernel} forward ({dtype})")
                if not info["backward"]:
                    missing.append(f"{kernel} backward ({dtype})")

    if missing:
        print("\nHigh-priority gaps:")
        for item in missing[:10]:  # Top 10
            print(f"  - {item}")
    else:
        print("\nAll priority kernels implemented!")


def main():
    parser = argparse.ArgumentParser(description="Kernel coverage report tool")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", "--md", action="store_true", help="Output as Markdown")
    parser.add_argument("--missing", action="store_true", help="Show missing implementations")
    args = parser.parse_args()

    kernel_dir = find_kernel_dir()
    if not kernel_dir:
        print("Error: Could not find src/kernels directory", file=sys.stderr)
        sys.exit(1)

    coverage = analyze_kernels(kernel_dir)

    if args.json:
        print_json(coverage)
    elif args.markdown:
        print_markdown(coverage)
    else:
        print_table(coverage)
        if args.missing:
            print_missing(coverage)


if __name__ == "__main__":
    main()
