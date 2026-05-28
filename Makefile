# Plain Makefile fallback (CMake is the primary build). Uses -arch=native so
# the build matches whatever GPU is present rather than hardcoding an arch.
NVCC    ?= nvcc
ARCH    ?= native
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH)
LDLIBS  := -lcublas

BIN := sgemm

.PHONY: all clean
all: $(BIN)

$(BIN): src/sgemm/sgemm.cu src/sgemm/kernels.cuh
	$(NVCC) $(NVCCFLAGS) src/sgemm/sgemm.cu -o $(BIN) $(LDLIBS)

clean:
	rm -f $(BIN)
