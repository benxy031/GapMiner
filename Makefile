VERSION   = 0.0.1
SRC       = ./src
BIN       = ./bin
CC        = g++
NVCC      = nvcc
DBFLAGS   = -g
CXXFLAGS  = -Wall -Wextra -c -Winline -Wformat -Wformat-security \
            -pthread --param max-inline-insns-single=1000 -lm \
						-I/usr/include 
CUDA_HOME ?= /usr/local/cuda
CUDA_INC   = $(CUDA_HOME)/include
CUDA_LIB   = $(CUDA_HOME)/lib64
CUDA_ARCH ?= sm_86
NVCCFLAGS ?= -std=c++14 -O3 -arch=$(CUDA_ARCH) -Xcompiler "-fPIC -pthread" -I$(CUDA_INC)
LDFLAGS   = -lm -lcrypto -lmpfr -lgmp -pthread -lcurl -ljansson \
					  -L/usr/lib/x86_64-linux-gnu -lOpenCL
CUDA_LDFLAGS = -L$(CUDA_LIB) -lcudart
OTFLAGS   = -O3 -march=native -mtune=native -mavx -mfma -ffast-math -fPIC -pipe


# Ensure C++ compilation units see the CUDA backend macro when building the
# CUDA binary target so host/result types match the device kernels.
ifneq ($(filter gapminer-cuda,$(MAKECMDGOALS)),)
CXXFLAGS += -DUSE_CUDA_BACKEND -I$(CUDA_INC)
endif



.PHONY: clean test all install gapminer gapminer-cuda cuda

# default target
all: $(BIN)/gapminer

gapminer: $(BIN)/gapminer

gapminer-cuda: $(BIN)/gapminer-cuda

cuda: gapminer-cuda

install: all
	cp $(BIN)/gapminer /usr/bin/


# development
CXXFLAGS += $(DBFLAGS) 

# PoWCore debugging
#CXXFLAGS += -D DEBUG

# GPU-Miner enable fast debugging
# CXXFLAGS += -D DEBUG_BASIC -D DEBUG_FAST

# GPU-Miner enable slow debugging (more tests)
#CXXFLAGS += -D DEBUG_BASIC -D DEBUG_FAST -D DEBUG_SLOW

# ChineseSieve debugging
#CXXFLAGS += -D DEBUG_PREV_PRIME

# optimization
CXXFLAGS  += $(OTFLAGS)
LDFLAGS   += $(OTFLAGS)

# disable GPU support
# CXXFLAGS += -DCPU_ONLY 
# LDFLAGS   = -lm -lcrypto -lmpfr -lgmp -pthread -lcurl -ljansson

EV_SRC  = $(shell find $(SRC)/Evolution -type f -name '*.c'|grep -v -e test -e evolution.c)
EV_OBJ  = $(EV_SRC:%.c=%.o) $(SRC)/Evolution/src/evolution-O3.o
ALL_SRC = $(shell find $(SRC) -type f -name '*.cpp' ! -name 'main_cuda.cpp')
ALL_OBJ = $(ALL_SRC:%.cpp=%.o)

CUDA_CPP_SRCS = $(filter-out $(SRC)/GPUFermat.cpp,$(ALL_SRC))
CUDA_CPP_SRCS := $(filter-out $(SRC)/main.cpp,$(CUDA_CPP_SRCS))
CUDA_CPP_SRCS += $(SRC)/main_cuda.cpp
CUDA_OBJ = $(CUDA_CPP_SRCS:%.cpp=%.cuda.o) $(SRC)/CUDAFermat.cu.o

$(SRC)/GPUFermat.o:
	$(CC) $(CXXFLAGS) -std=c++11  $(SRC)/GPUFermat.cpp -o  $(SRC)/GPUFermat.o

%.o: %.cpp
	$(CC) $(CXXFLAGS) $^ -o $@

%.cuda.o: %.cpp
	$(CC) $(CXXFLAGS) -DUSE_CUDA_BACKEND -I$(CUDA_INC) $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA_BACKEND -c $< -o $@

evolution:
	$(MAKE) -C $(SRC)/Evolution evolution-O3

compile: $(ALL_OBJ) evolution

prepare:
	@mkdir -p bin

link: prepare compile

$(BIN)/gapminer: link
	$(CC) $(ALL_OBJ) $(EV_OBJ) $(LDFLAGS) -o $@

$(BIN)/gapminer-cuda: prepare evolution $(CUDA_OBJ)
	$(CC) $(CUDA_OBJ) $(EV_OBJ) $(LDFLAGS) $(CUDA_LDFLAGS) -o $@


clean:
	rm -rf $(BIN)
	rm -f $(ALL_OBJ) $(CUDA_OBJ)
	$(MAKE) -C $(SRC)/Evolution clean

