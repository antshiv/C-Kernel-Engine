CC      := gcc
# You can override AVX/arch flags from the environment if needed, e.g.:
#   make AVX_FLAGS="-march=native -mavx512f -mfma"
AVX_FLAGS ?= -march=native -mavx512f -mfma
INCLUDES := -Iinclude
CFLAGS  := -O3 -fPIC -fopenmp -Wall $(AVX_FLAGS) $(INCLUDES)
SRCS    := src/backend_native.c \
           src/kernels/gemm_kernels.c \
           src/kernels/layernorm_kernels.c \
           src/kernels/gelu_kernels.c \
           src/kernels/softmax_kernels.c \
           src/kernels/mlp_kernels.c \
           src/kernels/rmsnorm_kernels.c \
           src/kernels/swiglu_kernels.c \
           src/kernels/sigmoid_kernels.c
LIB        := libckernel_engine.so
LIB_GELU   := libckernel_gelu.so
LIB_RMSNORM:= libckernel_rmsnorm.so
LIB_LN     := libckernel_layernorm.so
LIB_SOFT   := libckernel_softmax.so
LIB_SWIGLU := libckernel_swiglu.so
LIB_SIGMOID:= libckernel_sigmoid.so

PYTHON  ?= python3
PY_TESTS := unittest/test_layernorm.py \
            unittest/test_gelu.py \
            unittest/test_softmax.py \
            unittest/test_softmax_backward.py \
            unittest/test_gemm.py \
            unittest/test_mlp.py \
            unittest/test_rmsnorm.py \
            unittest/test_swiglu.py \
            unittest/test_sigmoid.py

all: $(LIB)

$(LIB): $(SRCS)
	$(CC) $(CFLAGS) -shared -o $@ $(SRCS) -lm

$(LIB_GELU): src/kernels/gelu_kernels.c include/ckernel_engine.h
	$(CC) -O3 -fPIC -Wall -Iinclude -shared -o $@ src/kernels/gelu_kernels.c -lm

$(LIB_RMSNORM): src/kernels/rmsnorm_kernels.c include/ckernel_engine.h
	$(CC) -O3 -fPIC -Wall -Iinclude -shared -o $@ src/kernels/rmsnorm_kernels.c -lm

$(LIB_LN): src/kernels/layernorm_kernels.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/layernorm_kernels.c -lm

$(LIB_SOFT): src/kernels/softmax_kernels.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/softmax_kernels.c -lm

$(LIB_SWIGLU): src/kernels/swiglu_kernels.c src/kernels/sigmoid_kernels.c include/ckernel_engine.h
	$(CC) -O3 -fPIC -Wall -Iinclude -shared -o $@ src/kernels/swiglu_kernels.c src/kernels/sigmoid_kernels.c -lm

$(LIB_SIGMOID): src/kernels/sigmoid_kernels.c include/ckernel_engine.h
	$(CC) -O3 -fPIC -Wall -Iinclude -shared -o $@ src/kernels/sigmoid_kernels.c -lm

test-libs: $(LIB_GELU) $(LIB_RMSNORM) $(LIB_LN) $(LIB_SOFT) $(LIB_SWIGLU) $(LIB_SIGMOID)

test: $(LIB) test-libs
	@set -e; \
	for t in $(PY_TESTS); do \
	  echo "Running $$t"; \
	  $(PYTHON) $$t; \
	done; \
	echo "All Python kernel tests completed."

help:
	@echo "C-Kernel-Engine Make targets:"
	@echo "  make                 Build full engine library ($(LIB))"
	@echo "  make test            Build engine + per-kernel libs, run all Python kernel tests"
	@echo "  make test-libs       Build per-kernel shared libs ($(LIB_GELU), $(LIB_RMSNORM), $(LIB_LN), $(LIB_SOFT), $(LIB_SWIGLU), $(LIB_SIGMOID))"
	@echo "  make libckernel_gelu.so      Build GELU-only shared library"
	@echo "  make libckernel_rmsnorm.so   Build RMSNorm-only shared library"
	@echo "  make libckernel_layernorm.so Build LayerNorm-only shared library (requires AVX-512 CPU)"
	@echo "  make libckernel_softmax.so   Build Softmax-only shared library (requires AVX-512 CPU)"
	@echo "  make libckernel_swiglu.so    Build SwiGLU-only shared library"
	@echo "  make libckernel_sigmoid.so   Build Sigmoid-only shared library"
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
	rm -f $(LIB) $(LIB_GELU) $(LIB_RMSNORM) $(LIB_LN) $(LIB_SOFT) $(LIB_SWIGLU) $(LIB_SIGMOID)

.PHONY: all clean test test-libs help
