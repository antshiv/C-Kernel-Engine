CC      := gcc
# You can override AVX/arch flags from the environment if needed, e.g.:
#   make AVX_FLAGS="-march=native -mavx512f -mfma"
AVX_FLAGS ?= -march=native -mavx512f -mfma
INCLUDES := -Iinclude
CFLAGS  := -O3 -fPIC -fopenmp -Wall $(AVX_FLAGS) $(INCLUDES)

BUILD_DIR := build

SRCS    := src/backend_native.c \
           src/ckernel_ir.c \
           src/ckernel_codegen.c \
           src/kernels/gemm_kernels.c \
           src/kernels/layernorm_kernels.c \
           src/kernels/gelu_kernels.c \
           src/kernels/softmax_kernels.c \
           src/kernels/attention_kernels.c \
           src/kernels/mlp_kernels.c \
           src/kernels/rmsnorm_kernels.c \
           src/kernels/swiglu_kernels.c \
           src/kernels/sigmoid_kernels.c
LIB          := $(BUILD_DIR)/libckernel_engine.so
LIB_GELU     := $(BUILD_DIR)/libckernel_gelu.so
LIB_RMSNORM  := $(BUILD_DIR)/libckernel_rmsnorm.so
LIB_LN       := $(BUILD_DIR)/libckernel_layernorm.so
LIB_SOFT     := $(BUILD_DIR)/libckernel_softmax.so
LIB_SWIGLU   := $(BUILD_DIR)/libckernel_swiglu.so
LIB_SIGMOID  := $(BUILD_DIR)/libckernel_sigmoid.so
LIB_ATTENTION:= $(BUILD_DIR)/libckernel_attention.so

IR_DEMO := $(BUILD_DIR)/ck_ir_demo
DEFAULT_CONFIG := default.config.json

PYTHON  ?= python3
PY_TESTS := unittest/test_layernorm.py \
            unittest/test_gelu.py \
            unittest/test_softmax.py \
            unittest/test_softmax_backward.py \
            unittest/test_gemm.py \
            unittest/test_mlp.py \
            unittest/test_rmsnorm.py \
            unittest/test_swiglu.py \
            unittest/test_sigmoid.py \
            unittest/test_attention.py

all: $(BUILD_DIR) $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB): $(SRCS)
	$(CC) $(CFLAGS) -shared -o $@ $(SRCS) -lm

$(IR_DEMO): $(BUILD_DIR) src/ckernel_ir.c src/ckernel_ir_demo.c src/ckernel_codegen.c include/ckernel_ir.h include/ckernel_codegen.h
	$(CC) -O2 -Wall -Iinclude -o $@ src/ckernel_ir.c src/ckernel_codegen.c src/ckernel_ir_demo.c

ck: $(IR_DEMO)
	@echo "Running $(IR_DEMO) with $(DEFAULT_CONFIG)..."
	./$(IR_DEMO) $(DEFAULT_CONFIG)

$(LIB_GELU): $(BUILD_DIR) src/kernels/gelu_kernels.c include/ckernel_engine.h
	$(CC) -O3 -fPIC -Wall -Iinclude -shared -o $@ src/kernels/gelu_kernels.c -lm

$(LIB_RMSNORM): $(BUILD_DIR) src/kernels/rmsnorm_kernels.c include/ckernel_engine.h
	$(CC) -O3 -fPIC -Wall -Iinclude -shared -o $@ src/kernels/rmsnorm_kernels.c -lm

$(LIB_LN): $(BUILD_DIR) src/kernels/layernorm_kernels.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/layernorm_kernels.c -lm

$(LIB_SOFT): $(BUILD_DIR) src/kernels/softmax_kernels.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/softmax_kernels.c -lm

$(LIB_SWIGLU): $(BUILD_DIR) src/kernels/swiglu_kernels.c src/kernels/sigmoid_kernels.c include/ckernel_engine.h
	$(CC) -O3 -fPIC -Wall -Iinclude -shared -o $@ src/kernels/swiglu_kernels.c src/kernels/sigmoid_kernels.c -lm

$(LIB_SIGMOID): $(BUILD_DIR) src/kernels/sigmoid_kernels.c include/ckernel_engine.h
	$(CC) -O3 -fPIC -Wall -Iinclude -shared -o $@ src/kernels/sigmoid_kernels.c -lm

$(LIB_ATTENTION): $(BUILD_DIR) src/kernels/attention_kernels.c src/kernels/softmax_kernels.c include/ckernel_engine.h
	$(CC) -O3 -fPIC -Wall -Iinclude -shared -o $@ src/kernels/attention_kernels.c src/kernels/softmax_kernels.c -lm

# Convenience alias targets so existing commands still work.
libckernel_gelu.so: $(LIB_GELU)
	@true

libckernel_rmsnorm.so: $(LIB_RMSNORM)
	@true

libckernel_layernorm.so: $(LIB_LN)
	@true

libckernel_softmax.so: $(LIB_SOFT)
	@true

libckernel_swiglu.so: $(LIB_SWIGLU)
	@true

libckernel_sigmoid.so: $(LIB_SIGMOID)
	@true

libckernel_attention.so: $(LIB_ATTENTION)
	@true

test-libs: $(LIB_GELU) $(LIB_RMSNORM) $(LIB_LN) $(LIB_SOFT) $(LIB_SWIGLU) $(LIB_SIGMOID) $(LIB_ATTENTION)

test: $(LIB) test-libs
	@set -e; \
	for t in $(PY_TESTS); do \
	  echo "Running $$t"; \
	  LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $$t; \
	done; \
	echo "All Python kernel tests completed."

help:
	@echo "C-Kernel-Engine Make targets:"
	@echo "  make                 Build full engine library ($(LIB))"
	@echo "  make test            Build engine + per-kernel libs, run all Python kernel tests"
	@echo "  make $(IR_DEMO)      Build IR + codegen tool (HF config.json -> IR + C skeleton) into $(BUILD_DIR)"
	@echo "  make ck              Build IR tool and run it with $(DEFAULT_CONFIG) (prints forward/backward IR and C skeleton)"
	@echo "  make test-libs       Build per-kernel shared libs in $(BUILD_DIR) ($(LIB_GELU), $(LIB_RMSNORM), $(LIB_LN), $(LIB_SOFT), $(LIB_SWIGLU), $(LIB_SIGMOID))"
	@echo "  make libckernel_gelu.so      Build GELU-only shared library (outputs to $(LIB_GELU))"
	@echo "  make libckernel_rmsnorm.so   Build RMSNorm-only shared library (outputs to $(LIB_RMSNORM))"
	@echo "  make libckernel_layernorm.so Build LayerNorm-only shared library (requires AVX-512 CPU, outputs to $(LIB_LN))"
	@echo "  make libckernel_softmax.so    Build Softmax-only shared library (requires AVX-512 CPU, outputs to $(LIB_SOFT))"
	@echo "  make libckernel_swiglu.so     Build SwiGLU-only shared library (outputs to $(LIB_SWIGLU))"
	@echo "  make libckernel_sigmoid.so    Build Sigmoid-only shared library (outputs to $(LIB_SIGMOID))"
	@echo "  make libckernel_attention.so  Build attention-only shared library (scalar math + softmax kernel, outputs to $(LIB_ATTENTION))"
	@echo "  make clean            Remove all built libraries"
	@echo ""
	@echo "Python unittest scripts (run with: python3 <script>):"
	@echo "  unittest/test_gelu.py        - GELU forward/backward vs PyTorch"
	@echo "  unittest/test_rmsnorm.py     - RMSNorm forward/backward vs PyTorch"
	@echo "  unittest/test_sigmoid.py     - Sigmoid forward/backward vs PyTorch"
	@echo "  unittest/test_layernorm.py   - LayerNorm forward/backward vs PyTorch (AVX-512)"
	@echo "  unittest/test_softmax.py     - Causal softmax forward vs PyTorch (AVX-512)"
	@echo "  unittest/test_softmax_backward.py - Causal softmax backward vs PyTorch (AVX-512)"
	@echo "  unittest/test_gemm.py        - GEMM variants vs PyTorch matmul"
	@echo "  unittest/test_mlp.py         - MLP block forward/backward vs PyTorch"
	@echo "  unittest/test_swiglu.py      - SwiGLU activation forward/backward vs PyTorch"

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean test test-libs help
