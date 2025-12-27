#!/bin/bash
# PyTorch Parity Test Runner
# Runs parity tests and logs results to logs/parity_test/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/parity_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/parity_test_${TIMESTAMP}.txt"

# Create log directory
mkdir -p "$LOG_DIR"

echo "=== PyTorch Parity Test Runner ===" | tee "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 1: Build
echo "=== Step 1: Building project ===" | tee -a "$LOG_FILE"
cd "$PROJECT_DIR"
make 2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 2: Run parity tests
echo "=== Step 2: Running PyTorch Parity Tests ===" | tee -a "$LOG_FILE"
cd "$PROJECT_DIR/unittest"
export LD_LIBRARY_PATH="$PROJECT_DIR/build:$LD_LIBRARY_PATH"

# Run with --quick for faster execution
python3 test_pytorch_parity.py --quick 2>&1 | tee -a "$LOG_FILE"
TEST_EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "=== Test completed ===" | tee -a "$LOG_FILE"
echo "Exit code: $TEST_EXIT_CODE" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"

# Create a symlink to the latest log
ln -sf "parity_test_${TIMESTAMP}.txt" "$LOG_DIR/latest.txt"

exit $TEST_EXIT_CODE
