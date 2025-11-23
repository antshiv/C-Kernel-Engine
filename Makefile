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
           src/kernels/softmax_kernels.c
LIB     := libckernel_engine.so

PYTHON  ?= python3
PY_TESTS := unittest/test_layernorm.py \
            unittest/test_gelu.py \
            unittest/test_softmax.py \
            unittest/test_softmax_backward.py \
            unittest/test_gemm.py

all: $(LIB)

$(LIB): $(SRCS)
	$(CC) $(CFLAGS) -shared -o $@ $(SRCS) -lm

test: $(LIB)
	@set -e; \
	for t in $(PY_TESTS); do \
	  echo "Running $$t"; \
	  $(PYTHON) $$t; \
	done; \
	echo "All Python kernel tests completed."

clean:
	rm -f $(LIB)

.PHONY: all clean test
