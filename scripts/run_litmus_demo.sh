#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-"$ROOT/build"}"

mkdir -p "$BUILD_DIR"

ARGS=(
  --vocab 100
  --ctx 100
  --embed 64
  --intermediate 128
  --heads 4
  --kv-heads 2
  --svg "$BUILD_DIR/litmus_report.svg"
)

python3 "$ROOT/unittest/test_lm_head_litmus.py" "${ARGS[@]}" | tee "$BUILD_DIR/litmus_demo.log"
