CC      := gcc
# You can override AVX/arch flags from the environment if needed, e.g.:
#   make AVX_FLAGS="-mavx2"
#   make AVX_FLAGS=""            # scalar build
CPU_FLAGS := $(shell grep -m1 '^flags' /proc/cpuinfo 2>/dev/null)
ifneq (,$(findstring avx512f,$(CPU_FLAGS)))
AVX_FLAGS ?= -mavx512f
else ifneq (,$(findstring avx2,$(CPU_FLAGS)))
AVX_FLAGS ?= -mavx2
else ifneq (,$(findstring avx,$(CPU_FLAGS)))
AVX_FLAGS ?= -mavx
else
AVX_FLAGS ?=
endif
INCLUDES := -Iinclude
CFLAGS  := -O3 -fPIC -fopenmp -Wall $(AVX_FLAGS) $(INCLUDES)

BUILD_DIR := build

SRCS    := src/backend_native.c \
           src/ckernel_ir.c \
           src/ckernel_codegen.c \
           src/ckernel_kernel_specs.c \
           src/ckernel_alloc.c \
           src/ckernel_registry.c \
           src/ckernel_orchestration.c \
           src/ckernel_model_layout.c \
           src/ckernel_model_load.c \
            src/kernels/gemm_kernels.c \
            src/kernels/layernorm_kernels.c \
           src/kernels/gelu_kernels.c \
           src/kernels/softmax_kernels.c \
           src/kernels/attention_kernels.c \
           src/kernels/embedding_kernels.c \
           src/kernels/loss_kernels.c \
           src/kernels/mlp_kernels.c \
           src/kernels/rmsnorm_kernels.c \
           src/kernels/swiglu_kernels.c \
           src/kernels/sigmoid_kernels.c \
           src/kernels/rope_kernels.c
LIB          := $(BUILD_DIR)/libckernel_engine.so
LIB_GELU     := $(BUILD_DIR)/libckernel_gelu.so
LIB_RMSNORM  := $(BUILD_DIR)/libckernel_rmsnorm.so
LIB_LN       := $(BUILD_DIR)/libckernel_layernorm.so
LIB_SOFT     := $(BUILD_DIR)/libckernel_softmax.so
LIB_SWIGLU   := $(BUILD_DIR)/libckernel_swiglu.so
LIB_SIGMOID  := $(BUILD_DIR)/libckernel_sigmoid.so
LIB_ATTENTION:= $(BUILD_DIR)/libckernel_attention.so
LIB_ROPE     := $(BUILD_DIR)/libckernel_rope.so

IR_DEMO := $(BUILD_DIR)/ck_ir_demo
DEFAULT_CONFIG := default.config.json
CONFIG ?= $(DEFAULT_CONFIG)
OUT ?= $(BUILD_DIR)/generated_model.c
TINY_CONFIG ?= tiny.config.json
SMALL_CONFIG ?= small10mb.config.json
TINY_TRAIN_LR ?= 1e-3
TINY_TRAIN_ARGS ?= --dump

PYTHON  ?= python3
PYTHONFLAGS ?= -B
PY_TESTS := unittest/test_layernorm.py \
            unittest/test_gelu.py \
            unittest/test_softmax.py \
            unittest/test_softmax_backward.py \
            unittest/test_gemm.py \
            unittest/test_mlp.py \
            unittest/test_rmsnorm.py \
            unittest/test_swiglu.py \
            unittest/test_sigmoid.py \
            unittest/test_attention.py \
            unittest/test_attention_backward.py \
            unittest/test_rope.py \
            unittest/test_embedding.py \
            unittest/test_cross_entropy.py \
            unittest/test_orchestration_layer.py \
            unittest/test_lm_head_litmus.py

LITMUS_DEMO_ARGS ?= --vocab 100 --ctx 100 --embed 64 --intermediate 128 --heads 4 --kv-heads 2
LITMUS_DEMO_SVG ?= $(BUILD_DIR)/litmus_report.svg
LITMUS_DEMO_LOG ?= $(BUILD_DIR)/litmus_demo.log

all: $(BUILD_DIR) $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB): $(SRCS)
	$(CC) $(CFLAGS) -shared -o $@ $(SRCS) -lm

$(IR_DEMO): $(BUILD_DIR) src/ckernel_ir.c src/ckernel_ir_demo.c src/ckernel_codegen.c src/ckernel_kernel_specs.c src/ckernel_registry.c include/ckernel_ir.h include/ckernel_codegen.h include/ckernel_registry.h include/ckernel_kernel_specs.h
	$(CC) -O2 -Wall -Iinclude -o $@ src/ckernel_ir.c src/ckernel_codegen.c src/ckernel_kernel_specs.c src/ckernel_registry.c src/ckernel_ir_demo.c

ck: $(IR_DEMO)
	@echo "Running $(IR_DEMO) with $(DEFAULT_CONFIG)..."
	./$(IR_DEMO) $(DEFAULT_CONFIG)

emit: $(IR_DEMO)
	@echo "Generating runtime from $(CONFIG) -> $(OUT)..."
	./$(IR_DEMO) $(CONFIG) --emit $(OUT)

ck-emit: emit
	@true

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

$(LIB_ROPE): $(BUILD_DIR) src/kernels/rope_kernels.c include/ckernel_engine.h
	$(CC) -O3 -fPIC -Wall -Iinclude -shared -o $@ src/kernels/rope_kernels.c -lm

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

libckernel_rope.so: $(LIB_ROPE)
	@true

test-libs: $(LIB_GELU) $(LIB_RMSNORM) $(LIB_LN) $(LIB_SOFT) $(LIB_SWIGLU) $(LIB_SIGMOID) $(LIB_ATTENTION) $(LIB_ROPE)

test: $(LIB) test-libs
	@set -e; \
	for t in $(PY_TESTS); do \
	  echo "Running $$t"; \
	  LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) $$t; \
	done; \
	echo "All Python kernel tests completed."

litmus:
	$(PYTHON) $(PYTHONFLAGS) unittest/test_lm_head_litmus.py $(ARGS)

litmus-demo: $(BUILD_DIR)
	@echo "Running litmus demo: $(LITMUS_DEMO_ARGS)"
	@$(PYTHON) $(PYTHONFLAGS) unittest/test_lm_head_litmus.py $(LITMUS_DEMO_ARGS) --svg $(LITMUS_DEMO_SVG) | tee $(LITMUS_DEMO_LOG)

layer-parity: $(LIB)
	$(PYTHON) $(PYTHONFLAGS) unittest/test_orchestration_layer.py $(ARGS) $(if $(TOL),--tol $(TOL),)

layer-parity-scalar:
	$(MAKE) -B $(LIB) AVX_FLAGS=
	$(PYTHON) $(PYTHONFLAGS) unittest/test_orchestration_layer.py $(ARGS) $(if $(TOL),--tol $(TOL),)

gen-specs:
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_kernel_specs.py

tiny-e2e: $(IR_DEMO) $(LIB)
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_bump.py --config $(TINY_CONFIG) --output $(BUILD_DIR)/tiny_weights.bin
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_tokens.py --config $(TINY_CONFIG) --output $(BUILD_DIR)/tiny_tokens.bin
	./$(IR_DEMO) $(TINY_CONFIG) --emit $(BUILD_DIR)/tiny_generated.c
	$(CC) $(CFLAGS) -Iinclude $(BUILD_DIR)/tiny_generated.c $$(cat $(BUILD_DIR)/tiny_generated.c.kernels) -o $(BUILD_DIR)/tiny_model -lm
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(BUILD_DIR)/tiny_model \
	  --model-weights $(BUILD_DIR)/tiny_weights.bin \
	  --tokens $(BUILD_DIR)/tiny_tokens.bin \
	  --out-logits $(BUILD_DIR)/tiny_logits.bin

tiny-train: $(IR_DEMO) $(LIB)
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_bump.py --config $(TINY_CONFIG) --output $(BUILD_DIR)/tiny_weights.bin
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_tokens.py --config $(TINY_CONFIG) --output $(BUILD_DIR)/tiny_tokens.bin --targets $(BUILD_DIR)/tiny_targets.bin
	./$(IR_DEMO) $(TINY_CONFIG) --emit $(BUILD_DIR)/tiny_generated.c
	$(CC) $(CFLAGS) -Iinclude $(BUILD_DIR)/tiny_generated.c $$(cat $(BUILD_DIR)/tiny_generated.c.kernels) -o $(BUILD_DIR)/tiny_model -lm
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(BUILD_DIR)/tiny_model \
	  --model-weights $(BUILD_DIR)/tiny_weights.bin \
	  --tokens $(BUILD_DIR)/tiny_tokens.bin \
	  --targets $(BUILD_DIR)/tiny_targets.bin \
	  --backward --lr $(TINY_TRAIN_LR) $(TINY_TRAIN_ARGS)

tiny-parity: $(IR_DEMO)
	$(PYTHON) $(PYTHONFLAGS) scripts/tiny_train_parity.py --config $(TINY_CONFIG)

small-e2e: $(IR_DEMO) $(LIB)
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_bump.py --config $(SMALL_CONFIG) --output $(BUILD_DIR)/small_weights.bin
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_tokens.py --config $(SMALL_CONFIG) --output $(BUILD_DIR)/small_tokens.bin
	./$(IR_DEMO) $(SMALL_CONFIG) --emit $(BUILD_DIR)/small_generated.c
	$(CC) $(CFLAGS) -Iinclude $(BUILD_DIR)/small_generated.c $$(cat $(BUILD_DIR)/small_generated.c.kernels) -o $(BUILD_DIR)/small_model -lm
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(BUILD_DIR)/small_model \
	  --model-weights $(BUILD_DIR)/small_weights.bin \
	  --tokens $(BUILD_DIR)/small_tokens.bin \
	  --out-logits $(BUILD_DIR)/small_logits.bin

help:
	@echo "C-Kernel-Engine Make targets:"
	@echo "  make                 Build full engine library ($(LIB))"
	@echo "  make test            Build engine + per-kernel libs, run all Python kernel tests"
	@echo "  make $(IR_DEMO)      Build IR + codegen tool (HF config.json -> IR + C skeleton) into $(BUILD_DIR)"
	@echo "  make ck              Build IR tool and run it with $(DEFAULT_CONFIG) (prints forward/backward IR and C skeleton)"
	@echo "  make emit CONFIG=path OUT=path  Emit stitched runtime C file from config/IR (writes OUT.kernels manifest)"
	@echo "  make ck-emit CONFIG=path OUT=path  Alias for emit"
	@echo "  ./$(IR_DEMO) <config> --emit out.c  Emit a stitched runtime C file from config/IR (writes out.c.kernels)"
	@echo "  make test-libs       Build per-kernel shared libs in $(BUILD_DIR) ($(LIB_GELU), $(LIB_RMSNORM), $(LIB_LN), $(LIB_SOFT), $(LIB_SWIGLU), $(LIB_SIGMOID))"
	@echo "  make litmus ARGS=\"--layers 1 --embed 64 --svg build/litmus.svg\"  Run LM head + CE backward parity litmus (PyTorch)"
	@echo "  make litmus-demo     Run the 100x100 litmus demo and capture output + SVG"
	@echo "  make layer-parity ARGS=\"--tokens 1024 --embed 64 --heads 4 --kv-heads 2 --rope\" TOL=5e-2  Run layer forward parity vs PyTorch"
	@echo "  make layer-parity-scalar ARGS=\"--tokens 1024 --embed 64 --heads 4 --kv-heads 2 --rope --strict-ref\" TOL=1e-3  Run layer parity with scalar build (C ref)"
	@echo "  make tiny-e2e        Generate random weights/tokens and run tiny end-to-end forward"
	@echo "  make tiny-train      Generate random weights/tokens/targets and run tiny forward+backward+SGD"
	@echo "  make tiny-parity     Run tiny end-to-end training parity vs PyTorch (1 step)"
	@echo "  make small-e2e       Same as tiny-e2e but ~10MB weights"
	@echo "  make libckernel_gelu.so      Build GELU-only shared library (outputs to $(LIB_GELU))"
	@echo "  make libckernel_rmsnorm.so   Build RMSNorm-only shared library (outputs to $(LIB_RMSNORM))"
	@echo "  make libckernel_layernorm.so Build LayerNorm-only shared library (AVX-512 if available, outputs to $(LIB_LN))"
	@echo "  make libckernel_softmax.so    Build Softmax-only shared library (outputs to $(LIB_SOFT))"
	@echo "  make libckernel_swiglu.so     Build SwiGLU-only shared library (outputs to $(LIB_SWIGLU))"
	@echo "  make libckernel_sigmoid.so    Build Sigmoid-only shared library (outputs to $(LIB_SIGMOID))"
	@echo "  make libckernel_attention.so  Build attention-only shared library (scalar math + softmax kernel, outputs to $(LIB_ATTENTION))"
	@echo "  make gen-specs        Regenerate C kernel spec registry from kernel_maps/*.json"
	@echo "  make clean            Remove all built libraries"
	@echo ""
	@echo "Python unittest scripts (run with: python3 <script>):"
	@echo "  unittest/test_gelu.py        - GELU forward/backward vs PyTorch"
	@echo "  unittest/test_rmsnorm.py     - RMSNorm forward/backward vs PyTorch"
	@echo "  unittest/test_sigmoid.py     - Sigmoid forward/backward vs PyTorch"
	@echo "  unittest/test_layernorm.py   - LayerNorm forward/backward vs PyTorch"
	@echo "  unittest/test_softmax.py     - Causal softmax forward vs PyTorch"
	@echo "  unittest/test_softmax_backward.py - Causal softmax backward vs PyTorch"
	@echo "  unittest/test_gemm.py        - GEMM variants vs PyTorch matmul"
	@echo "  unittest/test_mlp.py         - MLP block forward/backward vs PyTorch"
	@echo "  unittest/test_swiglu.py      - SwiGLU activation forward/backward vs PyTorch"
	@echo "  unittest/test_orchestration_layer.py - Full layer forward stitch vs PyTorch (GQA/MHA)"

clean:
	rm -rf $(BUILD_DIR)

# Litmus test for full forward pass parity with PyTorch
# ==============================================================================
TEST_HARNESS_SRCS := src/backend_native.c \
	src/ckernel_alloc.c \
	src/ckernel_ir.c \
	src/ckernel_orchestration.c \
	src/ckernel_registry.c \
	src/kernels/attention_kernels.c \
	src/kernels/gelu_kernels.c \
	src/kernels/gemm_kernels.c \
	src/kernels/layernorm_kernels.c \
	src/kernels/mlp_kernels.c \
	src/kernels/rmsnorm_kernels.c \
	src/kernels/sigmoid_kernels.c \
	src/kernels/softmax_kernels.c \
	src/kernels/swiglu_kernels.c \
	src/kernels/rope_kernels.c

.PHONY: litmus-test

litmus-test:
	@echo "--- [Step 1] Generating C Runtime Code ---"
	@make emit OUT=build/generated_model.c
	@echo "\n--- [Step 2] Generating PyTorch Reference Data ---"
	$(PYTHON) $(PYTHONFLAGS) unittest/generate_reference_data.py
	@echo "\n--- [Step 3] Compiling C Test Harness ---"
	$(CC) $(CFLAGS) -Iinclude test_forward_pass.c build/generated_model.c $(TEST_HARNESS_SRCS) -o build/test_forward_pass -lm -lpthread -lrt
	@echo "Compilation complete: build/test_forward_pass"
	@echo "\n--- [Step 4] Running C Test Harness ---"
	./build/test_forward_pass
	@echo "\n--- [Step 5] Comparing C output with PyTorch reference ---"
	$(PYTHON) $(PYTHONFLAGS) unittest/compare_outputs.py

.PHONY: all clean test test-libs help litmus litmus-test
