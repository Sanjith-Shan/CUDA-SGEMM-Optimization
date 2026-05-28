# Plain Makefile fallback (CMake is the primary build). Uses -arch=native so
# the build matches whatever GPU is present rather than hardcoding an arch.
NVCC    ?= nvcc
ARCH    ?= native
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH)
LDLIBS  := -lcublas

BIN := sgemm
FUSED := softmax layernorm

.PHONY: all fused clean
all: $(BIN) fused
fused: $(FUSED)

$(BIN): src/sgemm/sgemm.cu src/sgemm/kernels.cuh
	$(NVCC) $(NVCCFLAGS) src/sgemm/sgemm.cu -o $(BIN) $(LDLIBS)

softmax: src/fused/softmax.cu
	$(NVCC) $(NVCCFLAGS) src/fused/softmax.cu -o softmax

layernorm: src/fused/layernorm.cu
	$(NVCC) $(NVCCFLAGS) src/fused/layernorm.cu -o layernorm

clean:
	rm -f $(BIN) $(FUSED)
