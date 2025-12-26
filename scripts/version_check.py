#!/usr/bin/env python3
"""Version checking utilities for C-Kernel-Engine scripts."""

import sys

MIN_TRANSFORMERS_VERSION = "4.35.0"
MIN_TORCH_VERSION = "2.0.0"


def parse_version(version_str):
    """Parse version string to tuple of integers."""
    parts = version_str.split(".")
    return tuple(int(p) for p in parts[:3])


def check_transformers():
    """Check transformers version and provide helpful error if too old."""
    try:
        import transformers
        version = getattr(transformers, "__version__", "0.0.0")
        if parse_version(version) < parse_version(MIN_TRANSFORMERS_VERSION):
            print(f"WARNING: transformers {version} is older than recommended {MIN_TRANSFORMERS_VERSION}")
            print("  Some features may not work correctly.")
            print("  Upgrade with: pip install --upgrade transformers")
        return version
    except ImportError:
        print("ERROR: transformers not installed.")
        print("  Install with: pip install transformers>=4.35.0")
        sys.exit(1)


def check_torch():
    """Check torch version."""
    try:
        import torch
        version = getattr(torch, "__version__", "0.0.0").split("+")[0]
        if parse_version(version) < parse_version(MIN_TORCH_VERSION):
            print(f"WARNING: torch {version} is older than recommended {MIN_TORCH_VERSION}")
        return version
    except ImportError:
        print("ERROR: torch not installed.")
        print("  Install with: pip install torch>=2.0.0")
        sys.exit(1)


def check_all():
    """Check all dependencies and print versions."""
    torch_ver = check_torch()
    transformers_ver = check_transformers()
    print(f"torch: {torch_ver}, transformers: {transformers_ver}")
    return torch_ver, transformers_ver


if __name__ == "__main__":
    check_all()
