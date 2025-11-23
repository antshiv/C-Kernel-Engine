# C Kernel Engine

A C-first kernel library for LLMs and deep-learning workloads.

This project focuses on a **small, high-impact set of kernel shapes** that appear in transformer-style models, instead of trying to cover every BLAS case. The goal is to:

- Implement a clean kernel DSL around a handful of GEMM and elementwise patterns.
- Tune those kernels aggressively for CPU (x86/ARM/RISC-V, AVX2/AVX-512, etc.).
- Keep the implementation understandable and hackable for systems and HPC engineers.

See `docs/01-llm-kernel-shapes.md` for the core math that drives the design.
