#!/usr/bin/env python3
"""
Test Coverage Report Generator

Scans unittest/ directory and generates a report showing:
- What test files exist
- What kernels/features are tested
- FP32 vs BF16 test coverage
- Test function counts

Usage:
    python3 scripts/test_coverage.py              # Full report
    python3 scripts/test_coverage.py --summary    # Summary only
    python3 scripts/test_coverage.py --markdown   # Markdown format
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
UNITTEST_DIR = ROOT / "unittest"
BF16_DIR = UNITTEST_DIR / "bf16"


def find_test_files():
    """Find all test files in unittest/ and unittest/bf16/."""
    test_files = {
        "fp32": [],
        "bf16": [],
        "other": [],
    }

    # FP32 tests in unittest/
    for f in sorted(UNITTEST_DIR.glob("test_*.py")):
        if f.is_file():
            test_files["fp32"].append(f)

    # BF16 tests in unittest/bf16/
    if BF16_DIR.exists():
        for f in sorted(BF16_DIR.glob("test_*_bf16.py")):
            if f.is_file():
                test_files["bf16"].append(f)

    # Other test files
    for f in sorted(UNITTEST_DIR.glob("*.py")):
        if f.is_file() and not f.name.startswith("test_") and f.name not in ["__init__.py", "lib_loader.py", "test_utils.py", "bf16_utils.py"]:
            test_files["other"].append(f)

    return test_files


def extract_test_info(filepath):
    """Extract test functions and info from a test file."""
    info = {
        "name": filepath.stem,
        "functions": [],
        "imports": [],
        "kernel": None,
        "lines": 0,
    }

    try:
        content = filepath.read_text()
        info["lines"] = len(content.splitlines())

        # Extract kernel name from filename
        name = filepath.stem
        if name.startswith("test_"):
            name = name[5:]
        if name.endswith("_bf16"):
            name = name[:-5]
        info["kernel"] = name

        # Parse AST for test functions
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("test_") or node.name.startswith("run_"):
                        info["functions"].append(node.name)
        except SyntaxError:
            # Fall back to regex for malformed files
            info["functions"] = re.findall(r"def (test_\w+|run_\w+)\(", content)

    except Exception as e:
        info["error"] = str(e)

    return info


def get_kernel_test_matrix():
    """Build a matrix of kernels vs test types."""
    test_files = find_test_files()
    matrix = defaultdict(lambda: {"fp32": None, "bf16": None, "fp32_funcs": 0, "bf16_funcs": 0})

    for dtype, files in test_files.items():
        if dtype == "other":
            continue
        for f in files:
            info = extract_test_info(f)
            kernel = info["kernel"]
            if kernel:
                matrix[kernel][dtype] = f.name
                matrix[kernel][f"{dtype}_funcs"] = len(info["functions"])

    return dict(matrix)


def print_test_table(markdown=False):
    """Print the test coverage table."""
    matrix = get_kernel_test_matrix()

    if markdown:
        print("\n## Test Coverage by Kernel\n")
        print("| Kernel | FP32 Tests | BF16 Tests | FP32 Funcs | BF16 Funcs |")
        print("|--------|:----------:|:----------:|:----------:|:----------:|")
        for kernel in sorted(matrix.keys()):
            info = matrix[kernel]
            fp32 = "Yes" if info["fp32"] else "-"
            bf16 = "Yes" if info["bf16"] else "-"
            fp32_funcs = info["fp32_funcs"] if info["fp32"] else "-"
            bf16_funcs = info["bf16_funcs"] if info["bf16"] else "-"
            print(f"| {kernel} | {fp32} | {bf16} | {fp32_funcs} | {bf16_funcs} |")
    else:
        print("\n" + "=" * 70)
        print("  TEST COVERAGE BY KERNEL")
        print("=" * 70)
        print(f"\n{'Kernel':<20} {'FP32':<10} {'BF16':<10} {'FP32#':<8} {'BF16#':<8}")
        print("-" * 70)

        for kernel in sorted(matrix.keys()):
            info = matrix[kernel]
            fp32 = "Yes" if info["fp32"] else "-"
            bf16 = "Yes" if info["bf16"] else "-"
            fp32_funcs = str(info["fp32_funcs"]) if info["fp32"] else "-"
            bf16_funcs = str(info["bf16_funcs"]) if info["bf16"] else "-"
            print(f"{kernel:<20} {fp32:<10} {bf16:<10} {fp32_funcs:<8} {bf16_funcs:<8}")

        print("-" * 70)


def print_test_files(markdown=False):
    """Print list of all test files."""
    test_files = find_test_files()

    if markdown:
        print("\n## Test Files\n")
        print("### FP32 Tests\n")
        for f in test_files["fp32"]:
            info = extract_test_info(f)
            print(f"- `{f.name}` ({len(info['functions'])} test functions)")

        print("\n### BF16 Tests\n")
        for f in test_files["bf16"]:
            info = extract_test_info(f)
            print(f"- `{f.name}` ({len(info['functions'])} test functions)")
    else:
        print("\n" + "=" * 70)
        print("  TEST FILES")
        print("=" * 70)

        print("\n  FP32 Tests (unittest/):")
        print("  " + "-" * 50)
        for f in test_files["fp32"]:
            info = extract_test_info(f)
            print(f"    {f.name:<35} {len(info['functions']):>3} functions")

        print("\n  BF16 Tests (unittest/bf16/):")
        print("  " + "-" * 50)
        for f in test_files["bf16"]:
            info = extract_test_info(f)
            print(f"    {f.name:<35} {len(info['functions']):>3} functions")


def print_summary(markdown=False):
    """Print summary statistics."""
    test_files = find_test_files()
    matrix = get_kernel_test_matrix()

    fp32_count = len(test_files["fp32"])
    bf16_count = len(test_files["bf16"])

    fp32_funcs = sum(extract_test_info(f)["functions"].__len__() for f in test_files["fp32"])
    bf16_funcs = sum(extract_test_info(f)["functions"].__len__() for f in test_files["bf16"])

    kernels_with_fp32 = sum(1 for k, v in matrix.items() if v["fp32"])
    kernels_with_bf16 = sum(1 for k, v in matrix.items() if v["bf16"])
    kernels_with_both = sum(1 for k, v in matrix.items() if v["fp32"] and v["bf16"])

    if markdown:
        print("\n## Test Summary\n")
        print(f"- **FP32 Test Files**: {fp32_count} ({fp32_funcs} test functions)")
        print(f"- **BF16 Test Files**: {bf16_count} ({bf16_funcs} test functions)")
        print(f"- **Kernels with FP32 tests**: {kernels_with_fp32}")
        print(f"- **Kernels with BF16 tests**: {kernels_with_bf16}")
        print(f"- **Kernels with both**: {kernels_with_both}")
    else:
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)
        print(f"\n  FP32 Test Files:        {fp32_count:>3}  ({fp32_funcs} test functions)")
        print(f"  BF16 Test Files:        {bf16_count:>3}  ({bf16_funcs} test functions)")
        print(f"  Total Test Files:       {fp32_count + bf16_count:>3}")
        print(f"\n  Kernels with FP32:      {kernels_with_fp32:>3}")
        print(f"  Kernels with BF16:      {kernels_with_bf16:>3}")
        print(f"  Kernels with both:      {kernels_with_both:>3}")


def print_missing_tests(markdown=False):
    """Print kernels missing tests."""
    matrix = get_kernel_test_matrix()

    # Kernels that have FP32 but not BF16
    missing_bf16 = [k for k, v in matrix.items() if v["fp32"] and not v["bf16"]]

    if markdown:
        print("\n## Missing BF16 Tests\n")
        if missing_bf16:
            for k in sorted(missing_bf16):
                print(f"- {k}")
        else:
            print("All FP32 kernels have BF16 tests!")
    else:
        print("\n" + "=" * 70)
        print("  MISSING BF16 TESTS")
        print("=" * 70)
        print("\n  FP32 kernels without BF16 tests:")
        print("  " + "-" * 40)
        if missing_bf16:
            for k in sorted(missing_bf16):
                print(f"    - {k}")
        else:
            print("    All FP32 kernels have BF16 tests!")


def main():
    parser = argparse.ArgumentParser(description="Test coverage report generator")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--files", action="store_true", help="Show test file list")
    parser.add_argument("--missing", action="store_true", help="Show missing tests")
    parser.add_argument("--markdown", "--md", action="store_true", help="Output as Markdown")
    args = parser.parse_args()

    show_all = not any([args.summary, args.files, args.missing])

    if not args.markdown:
        print("\n" + "=" * 70)
        print("  C-KERNEL-ENGINE TEST COVERAGE REPORT")
        print("=" * 70)

    if show_all or args.summary:
        print_summary(markdown=args.markdown)

    if show_all:
        print_test_table(markdown=args.markdown)

    if show_all or args.files:
        print_test_files(markdown=args.markdown)

    if show_all or args.missing:
        print_missing_tests(markdown=args.markdown)

    print()


if __name__ == "__main__":
    main()
