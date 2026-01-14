VERSION   = 0.0.1
SRC       = ./src
BIN       = ./bin
CC        = g++
DBFLAGS   = -g
CXXFLAGS  = -Wall -Wextra -c -Winline -Wformat -Wformat-security -Werror=calloc-transposed-args \
            -pthread --param max-inline-insns-single=1000 -lm \
						-I/usr/include 
LDFLAGS   = -lm -lcrypto -lmpfr -lgmp -pthread -lcurl -ljansson \
					  -L/usr/lib/x86_64-linux-gnu -lOpenCL
OTFLAGS   = -O3 -march=skylake -mtune=skylake -mavx2 -mfma -ffast-math -fPIC -pipe



.PHONY: clean test all install

# default target
all: link
	$(CC) $(ALL_OBJ) $(EV_OBJ) $(LDFLAGS) -o $(BIN)/gapminer

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
ALL_SRC = $(shell find $(SRC) -type f -name '*.cpp')
ALL_OBJ = $(ALL_SRC:%.cpp=%.o) 

$(SRC)/GPUFermat.o:
	$(CC) $(CXXFLAGS) -std=c++11  $(SRC)/GPUFermat.cpp -o  $(SRC)/GPUFermat.o

%.o: %.cpp
	$(CC) $(CXXFLAGS) $^ -o $@

evolution:
	$(MAKE) -C $(SRC)/Evolution evolution-O3

compile: $(ALL_OBJ) $(ALL_OBJ) evolution

prepare:
	@mkdir -p bin

link: prepare compile

clean:
	rm -rf $(BIN)
	rm -f $(ALL_OBJ)
	$(MAKE) -C $(SRC)/Evolution clean

