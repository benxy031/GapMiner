# Copilot Instructions for GapMiner

## Project Overview
- **GapMiner** is a standalone miner for Gapcoin, supporting CPU, OpenCL (GPU), and CUDA (NVIDIA GPU) backends. It implements Gapcoin's prime gap-based Proof-of-Work (PoW) algorithm.
- The codebase is C++ (with some C and CUDA), organized into modular components: mining logic, PoW core, evolutionary algorithms, and GPU acceleration.
- Major directories:
  - `src/` — Main miner sources (CPU, OpenCL, CUDA)
  - `src/PoWCore/` — Prime gap PoW logic and sieving
  - `src/Evolution/` — Evolutionary algorithm library (ANSI C)
  - `src/gpu/` and `bin/gpu/` — OpenCL kernels
  - `bin/` — Built binaries: `gapminer` (CPU/OpenCL), `gapminer-cuda` (CUDA)

## Build & Run
- **Dependencies:** pthread, openssl, curl, jansson, gmp, mpfr, boost_system, OpenCL (for GPU), CUDA toolkit (for CUDA miner)
- **Build commands:**
  - `make` or `make gapminer` — CPU/OpenCL miner (`bin/gapminer`)
  - `make gapminer-cuda` — CUDA miner (`bin/gapminer-cuda`)
  - `make clean` — Remove binaries/objects
  - `sudo make install` — Install CPU miner to `/usr/bin/`
  - Use `CUDA_HOME` and `CUDA_ARCH` to customize CUDA builds
- **Run:**
  - `bin/gapminer [options]` or `bin/gapminer-cuda [options]`
  - See `cli.txt` for example CLI invocations and tuning flags

## Key Architectural Patterns
- **Mining Pipeline:**
  - Block header and work fetched via RPC/Stratum (`Rpc.*`, `Stratum.*`)
  - Sieving for prime gaps: `ChineseSieve`, `HybridSieve`, `PoWCore/src/Sieve.*`
  - Candidate generation and Fermat primality tests: CPU, OpenCL, or CUDA (`GPUFermat`, `CUDAFermat`)
  - Evolutionary algorithms for optimization: `src/Evolution/`
- **GPU Acceleration:**
  - OpenCL kernels in `src/gpu/` and `bin/gpu/`
  - CUDA path uses `--cuda-sieve-proto` (see `sievePrototypeKernel.txt` for pipeline details)
  - GPU/CPU work queues are managed to maximize throughput (see `run-notes.txt`)
- **Options Parsing:**
  - All CLI options handled by `Opts.*` (see `Opts.h` for available flags)

## Project-Specific Conventions
- **Debugging:**
  - Enable debug flags in `Makefile` (e.g., `-D DEBUG`, `-D DEBUG_BASIC`)
  - Extra-verbose GPU logs: use `--extra-verbose` or `-e`
- **Tuning:**
  - Sieve and batch sizes: `--sieve-size`, `--work-items`, `--num-gpu-tests`, etc.
  - See `run-notes.txt` and `gpu-copy-monitoring.txt` for performance tuning and diagnostics
- **Extending:**
  - New PoW logic: extend `PoWCore/`
  - New mining strategies: see `Miner.*`, `HybridSieve.*`, `GPUFermat.*`

## References
- `README.md` — Setup, build, and usage
- `sievePrototypeKernel.txt`, `run-notes.txt`, `gpu-copy-monitoring.txt` — GPU/CUDA pipeline and tuning
- `cli.txt` — Example CLI usage
- `src/PoWCore/README.md` — Prime gap PoW algorithm details

---
For any unclear conventions or missing details, please request clarification or point to relevant files for further documentation.
