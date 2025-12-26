#!/bin/bash
# Test script for training fix
# Run this on a machine with sufficient memory (>8GB recommended)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_CACHE="${HOME}/.cache/ck-engine/models/HuggingFaceTB--SmolLM-135M"

cd "$PROJECT_DIR"

echo "=== Step 1: Clean and build ==="
make clean
make
make ck-cli build/ck_ir_demo

echo ""
echo "=== Step 2: Regenerate model.c with fixed codegen ==="
./build/ck_ir_demo "$MODEL_CACHE/config.json" \
    --emit "$MODEL_CACHE/model.c" --emit-lib

echo ""
echo "=== Step 3: Verify gradient allocation is conditional ==="
if grep -q "d_token_emb_offset = m->training_enabled" "$MODEL_CACHE/model.c"; then
    echo "OK: Gradient allocation is conditional on training_enabled"
else
    echo "ERROR: Gradient allocation not properly conditional"
    exit 1
fi

if grep -q 'getenv("CK_ENABLE_TRAINING")' "$MODEL_CACHE/model.c"; then
    echo "OK: Environment variable check is present"
else
    echo "ERROR: Environment variable check missing"
    exit 1
fi

echo ""
echo "=== Step 4: Recompile libmodel.so ==="
cd "$MODEL_CACHE"
icx -O3 -fPIC -shared -I"$PROJECT_DIR/include" -o libmodel.so model.c \
    -L"$PROJECT_DIR/build" -lckernel_engine -lm
cd "$PROJECT_DIR"

echo ""
echo "=== Step 5: Run training test WITH CK_ENABLE_TRAINING=1 ==="
echo "This allocates gradient buffers at init time"
echo ""

export CK_ENABLE_TRAINING=1
export LD_LIBRARY_PATH="$PROJECT_DIR/build:$LD_LIBRARY_PATH"

python3 scripts/test_training.py --model-dir "$MODEL_CACHE" --steps 5

echo ""
echo "=== Done ==="
echo ""
echo "If loss decreased across steps, the fix is working!"
echo "If loss stayed constant, there may be another issue."
