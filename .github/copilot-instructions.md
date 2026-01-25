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
  - `crt/` — Precomputed Chinese Remainder Theorem tables for sieve optimization

## Build & Run
- **Dependencies:** pthread, openssl, curl, jansson, gmp, mpfr, boost_system, OpenCL (for GPU), CUDA toolkit (for CUDA miner)
- **Build commands:**
  - `make` or `make gapminer` — CPU/OpenCL miner (`bin/gapminer`)
  - `make gapminer-cuda` — CUDA miner (`bin/gapminer-cuda`)
  - `make clean` — Remove binaries/objects
  - `sudo make install` — Install CPU miner to `/usr/bin/`
  - Use `CUDA_HOME` and `CUDA_ARCH` to customize CUDA builds
  - Enable debug flags in `Makefile` (e.g., `-D DEBUG`, `-D DEBUG_BASIC`) for development
- **Run:**
  - `bin/gapminer [options]` or `bin/gapminer-cuda [options]`
  - See `cli.txt` for example CLI invocations and tuning flags
  - Use `--extra-verbose` or `-e` for detailed GPU diagnostics

## Key Architectural Patterns
- **Mining Pipeline:**
  - Block header and work fetched via RPC/Stratum (`Rpc.*`, `Stratum.*`)
  - Sieving for prime gaps: `ChineseSieve`, `HybridSieve`, `PoWCore/src/Sieve.*`
  - Candidate generation and Fermat primality tests: CPU, OpenCL, or CUDA (`GPUFermat`, `CUDAFermat`)
  - Evolutionary algorithms for CRT optimization: `src/Evolution/`, `ctr-evolution.*`
  - Share submission with difficulty validation: `ShareProcessor.*`
- **GPU Acceleration:**
  - OpenCL kernels in `src/gpu/` and `bin/gpu/` (320-bit and 352-bit Montgomery arithmetic)
  - CUDA path uses `--cuda-sieve-proto` for experimental sieve front-end (see `sievePrototypeKernel.txt`)
  - GPU/CPU work queues managed via `GPUWorkList` in `HybridSieve.*` to maximize throughput
  - Adaptive batching: GPU waits for queue depth before launching kernels (see `run-notes.txt`)
- **Chinese Remainder Theorem (CRT) Optimization:**
  - Precomputed residue tables in `crt/` directory for fast sieve initialization
  - Evolutionary algorithms (`ctr-evolution.*`) optimize CRT parameters for better performance
  - Use `--calc-ctr` options to generate custom CRT tables
- **Options Parsing:**
  - All CLI options handled by `Opts.*` (see `Opts.h` for available flags)
  - Singleton pattern for global options access

## Project-Specific Conventions
- **Debugging:**
  - Enable debug flags in `Makefile` (e.g., `-D DEBUG`, `-D DEBUG_BASIC`, `-D DEBUG_FAST`)
  - Extra-verbose GPU logs: use `--extra-verbose` or `-e` (outputs to `tests` file)
  - GPU diagnostics include candidate dumps, queue depth monitoring, and benchmark kernels
- **Tuning:**
  - Sieve parameters: `--sieve-size`, `--sieve-primes`, `--shift`
  - GPU batching: `--work-items`, `--num-gpu-tests`, `--queue-size`, `--bitmap-pool-buffers`, `--snapshot-pool-buffers`
  - Launch control: `--gpu-launch-divisor`, `--gpu-launch-wait-ms`
  - RTX 3060 optimization: `-w 4096-8192`, `-n 32-64`, `--queue-size 16384-32768`
  - See `run-notes.txt` and `gpu-copy-monitoring.txt` for performance tuning and diagnostics
- **Memory Management:**
  - GPU work queues use pooled buffers to avoid allocation overhead
  - Bitmap pools (`--bitmap-pool-buffers`) cache sieve bitmaps in RAM
  - Snapshot pools (`--snapshot-pool-buffers`) for CUDA sieve residue caching
- **Threading:**
  - Pthread-based threading with mutex protection (`io_mutex` for console output)
  - Separate threads for getwork polling, sieving, and GPU processing
- **Extending:**
  - New PoW logic: extend `PoWCore/` with custom sieving or primality tests
  - New mining strategies: modify `Miner.*`, `HybridSieve.*`, `GPUFermat.*`
  - GPU kernels: add to `src/gpu/` (OpenCL) or extend `CUDAFermat.cu`
  - CRT optimization: use evolutionary algorithms in `src/Evolution/`
  - Recent CUDA optimizations (Jan 2026): `locate_window()` uses binary search for O(log n) performance, `sievePrototypeScanKernel` uses atomicAdd for thread safety and stability

## References
- `README.md` — Setup, build, and usage
- `sievePrototypeKernel.txt`, `run-notes.txt`, `gpu-copy-monitoring.txt` — GPU/CUDA pipeline and tuning
- `cli.txt` — Example CLI usage
- `src/PoWCore/README.md` — Prime gap PoW algorithm details
- `src/Evolution/README.md` — Evolutionary algorithm library

---
For any unclear conventions or missing details, please request clarification or point to relevant files for further documentation.
