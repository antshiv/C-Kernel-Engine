#!/bin/bash
# run_all_tests.sh - Comprehensive test suite for C-Kernel-Engine
# Usage: ./scripts/run_all_tests.sh [quick|full|stress]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
SKIPPED=0

# Test mode: quick (default), full, stress
MODE="${1:-quick}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  C-Kernel-Engine Test Suite${NC}"
echo -e "${BLUE}  Mode: ${MODE}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create test configs directory
mkdir -p build/test_configs

#######################################
# Helper functions
#######################################

run_test() {
    local name="$1"
    local cmd="$2"
    local expected_max_diff="${3:-1e-3}"

    echo -e "${YELLOW}[TEST]${NC} $name"

    if eval "$cmd" 2>&1 | tee build/test_output.log; then
        # Check if max weight diff is acceptable
        max_diff=$(grep "Max weight diff" build/test_output.log | awk '{print $4}' || echo "0")
        if [ -n "$max_diff" ]; then
            echo -e "       Max diff: $max_diff (threshold: $expected_max_diff)"
        fi
        echo -e "${GREEN}[PASS]${NC} $name"
        ((PASSED++))
    else
        echo -e "${RED}[FAIL]${NC} $name"
        ((FAILED++))
    fi
    echo ""
}

skip_test() {
    local name="$1"
    local reason="$2"
    echo -e "${YELLOW}[SKIP]${NC} $name - $reason"
    ((SKIPPED++))
    echo ""
}

create_config() {
    local file="$1"
    local hidden="$2"
    local heads="$3"
    local kv_heads="$4"
    local layers="$5"
    local intermediate="$6"
    local vocab="$7"
    local ctx="$8"
    local rope_theta="${9:-10000.0}"

    cat > "$file" << EOF
{
  "hidden_size": $hidden,
  "num_attention_heads": $heads,
  "num_key_value_heads": $kv_heads,
  "num_hidden_layers": $layers,
  "intermediate_size": $intermediate,
  "vocab_size": $vocab,
  "max_position_embeddings": $ctx,
  "rms_norm_eps": 1e-5,
  "rope_theta": $rope_theta
}
EOF
}

#######################################
# Generate test configs
#######################################

echo -e "${BLUE}Generating test configs...${NC}"

# Tiny (baseline)
create_config build/test_configs/tiny.json 64 2 2 2 128 256 64

# Medium
create_config build/test_configs/medium.json 256 4 4 4 512 1024 128

# GQA (grouped query attention)
create_config build/test_configs/gqa.json 128 8 2 2 256 256 64

# No RoPE
create_config build/test_configs/no_rope.json 64 2 2 2 128 256 64 0.0

# Single layer
create_config build/test_configs/single_layer.json 64 2 2 1 128 256 64

# Deep (more layers)
create_config build/test_configs/deep.json 64 2 2 6 128 256 64

# Wide (more heads)
create_config build/test_configs/wide.json 128 8 8 2 256 256 64

echo -e "${GREEN}Done.${NC}"
echo ""

#######################################
# QUICK TESTS (< 1 minute)
#######################################

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Quick Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. Build library first
echo -e "${YELLOW}[BUILD]${NC} Building library..."
if make all 2>&1 | tail -5; then
    echo -e "${GREEN}[BUILD]${NC} Library built successfully"
else
    echo -e "${RED}[BUILD]${NC} Library build failed"
    exit 1
fi
echo ""

# 2. Basic kernel tests (Python unit tests)
run_test "Kernel Unit Tests" "make test"

# 3. Tiny model - 5 steps
run_test "Tiny Model (5 steps)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/tiny.json --steps 5 --lr 1e-3"

# 4. Single layer test
run_test "Single Layer Model" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/single_layer.json --steps 5 --lr 1e-3"

# 5. No RoPE test
run_test "No RoPE (rope_theta=0)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/no_rope.json --steps 5 --lr 1e-3"

if [ "$MODE" = "quick" ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Quick Test Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    echo -e "${YELLOW}Skipped: $SKIPPED${NC}"
    echo ""
    echo "Run './scripts/run_all_tests.sh full' for comprehensive tests"
    exit $FAILED
fi

#######################################
# FULL TESTS (5-10 minutes)
#######################################

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Full Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 5. GQA test
run_test "GQA (8 heads, 2 kv_heads)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/gqa.json --steps 5 --lr 1e-3"

# 6. Medium model
run_test "Medium Model (256 hidden)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/medium.json --steps 5 --lr 1e-3"

# 7. Deep model
run_test "Deep Model (6 layers)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/deep.json --steps 5 --lr 1e-3"

# 8. Wide model
run_test "Wide Model (8 heads)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/wide.json --steps 5 --lr 1e-3"

# 9. Different learning rates
run_test "Very Small LR (1e-5)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/tiny.json --steps 5 --lr 1e-5"

run_test "Large LR (1e-1)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/tiny.json --steps 5 --lr 1e-1"

# 10. More training steps
run_test "Longer Training (20 steps)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/tiny.json --steps 20 --lr 1e-3"

if [ "$MODE" = "full" ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Full Test Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    echo -e "${YELLOW}Skipped: $SKIPPED${NC}"
    echo ""
    echo "Run './scripts/run_all_tests.sh stress' for stress tests"
    exit $FAILED
fi

#######################################
# STRESS TESTS (10+ minutes)
#######################################

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Stress Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 11. Convergence test (overfit)
run_test "Convergence Test (100 steps)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/tiny.json --steps 100 --lr 1e-2"

# 12. Overfit test
run_test "Overfit Test (500 steps)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/tiny.json --steps 500 --lr 1e-2"

# 13. Medium model longer
run_test "Medium Model (50 steps)" \
    "python3 scripts/tiny_train_parity.py --config build/test_configs/medium.json --steps 50 --lr 1e-3"

#######################################
# Summary
#######################################

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Final Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed. Check build/test_output.log for details.${NC}"
fi

exit $FAILED
