# GapMiner - A standalone Gapcoin (GAP) CPU, GPU, rpc, pool miner
---
<br/>
## So what's the purpose of a standalone rpc miner? 


  * GapMiner is completely open source (GPLv3) and (hopefully)
    well documented, giving developers a good start into the
    pGap algorithm (Gapcoins prime gap based hashing algorithm).

  * simplicity and little dependencies (pure C++ code and only 
    pthread, openssl, curl, jansson, boost_system, gmp and mpfr dependencies).

  * speed (at the moment, GapMiner has no speed improvements in comparison with Gapcoin,
    but this is hopefully going to change.)



## get it running
---

First of all, keep in mind that GapMiner still has alpha qualities and 
doesn't claim to be the fastest CPU miner out there. The focal point
is the readability and understandability of the Gapcoin mining algorithm!

Plus, currently it's only for Linux. Sorry.

### required libraries
  - pthread
  - openssl
  - curl
  - jansson
  - gmp 
  - mpfr
  - boost system library
  - OpenCL runtime/headers (AMD APP SDK, NVIDIA OpenCL, or Mesa/POCL) for GPU mining
  - *(optional)* CUDA toolkit 11.x+ if you plan to build the CUDA miner

### installation
```sh
  git clone https://github.com/gapcoin-project/GapMiner.git or
  for CUDA
  git clone https://github.com/benxy031/GapMiner.git
  cd GapMiner
  git submodule update --init --recursive
  make            # builds the default CPU/OpenCL binary: bin/gapminer

  # Optional: build the CUDA miner once the NVIDIA toolkit is installed.
  # Adjust CUDA_HOME if your toolkit is in a non-default location.
  make gapminer-cuda CUDA_HOME=/usr/local/cuda

  sudo make install   # installs bin/gapminer only
```

### build targets & options

| Target | Command | Notes |
| --- | --- | --- |
| CPU/OpenCL miner (default) | `make` or `make gapminer` | Requires OpenCL headers/libraries (NVIDIA, AMD, Intel, Mesa/POCL, etc.). Builds `bin/gapminer`. |
| CUDA miner | `make gapminer-cuda` *(or `make cuda`)* | Needs a CUDA-capable NVIDIA GPU and the CUDA toolkit. Set `CUDA_HOME` if it is not `/usr/local/cuda`, and optionally override `CUDA_ARCH` (default `sm_70`). Produces `bin/gapminer-cuda`. |
| Clean | `make clean` | Removes both binaries and intermediate objects. |

Additional tips:

1. **CPU-only build** – define `CPU_ONLY=1` on the make command line or add `-DCPU_ONLY` to `CXXFLAGS` if you want to disable all GPU code paths.
2. **Verbosity/diagnostics** – the Makefile inherits standard `make` variables, so you can run `make V=1` for verbose commands or `make -j$(nproc)` to parallelize compilation.
3. **CUDA tuning** – adjust `CUDA_ARCH` (e.g., `sm_86`) to match your GPU for best performance. Ensure the CUDA toolkit `bin/` directory is on your `PATH` so `nvcc` is found.

## Usage
---

Both miners expose the same CLI surface; simply run the binary you built (`bin/gapminer` for CPU/OpenCL or `bin/gapminer-cuda` for CUDA) with the options below.

  `gapminer [--options]`

### basic

 - `-o  --host [ipv4]` host ip address

 - `-p  --port [port]` port to connect to

 - `-u  --user [user]` user for gapcoin rpc authentification

 - `-x  --pwd [pwd]` password for gapcoin rpc authentification

#### example:

`gapminer -o 127.0.0.1 -p 31397 -u rpcuser -x rpcpassword`

### advanced

 - `-q  --quiet` be quiet (only prints shares)

 - `-i  --stats-interval [NUM]` interval (sec) to print mining informations

 - `-t  --threads [NUM]` number of mining threads

 - `-l  --pull-interval [NUM]` seconds to wait between getwork request

 - `-s  --sieve-size [NUM]` the prime sieve size

 - `-r  --sieve-primes [NUM]` number of primes to sieve

 - `-f  --shift [NUM]` the adder shift

 - `-h  --help` print this information

 - `-v  --license` show license of this program
