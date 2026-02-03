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

 - `-f  --shift [NUM]` the adder shift (default 20 for CPU, 45 for GPU)

 - `-w  --work-items [NUM]` GPU batch size (candidates launched per Fermat
   cycle). Higher values keep the CUDA queue deeper at the cost of more device
   memory per batch. For RTX 3060: start with 4096-8192.

 - `-n  --num-gpu-tests [NUM]` number of sieve offsets drained from the queue
   before each GPU kernel launch. Raise this alongside `-w` when the logs show
   repeated "queue depth ... using 110 tests (partial batch)" messages.
   For RTX 3060: start with 32-64.

 - `-z  --queue-size [NUM]` override for the `GPUWorkList` capacity. Leaving it
   unset keeps the auto-sized depth of `work-items * GPU_group_size /
   num-gpu-tests` (≈16k with the defaults). When you do override it, start around
   `32768` (≈32K) so the GPU still sees the same depth the auto sizing would
   pick, then raise it gradually only if the logs still show “GPU queue depth …
   waiting to batch.” Supplying a value applies it to both the GPU queue and the
   host sieve queue/bitmap pool, so always pair large values with
   `--bitmap-pool-buffers` to avoid ballooning RAM usage.

 - `--bitmap-pool-buffers [NUM]` override the number of reusable sieve bitmaps
   kept in RAM. The fallback is `queue-size + 2`, which becomes enormous if you
   set `--queue-size 32768`, so override this too. As a starting point, try
   `--queue-size 32768 --bitmap-pool-buffers 512`, which consumes about 750 MB
   with a 12 M sieve. Each buffer costs `sieve-size / 8` bytes (~1.9 MB when the
   window is 15 M), so scale the setting according to your available RAM and the
   backlog you want between the CPU sieve and GPU worker.

 - `--snapshot-pool-buffers [NUM]` override how many CUDA residue snapshots can
   be staged concurrently (used only with `--cuda-sieve-proto`). Each snapshot
   is `sieve-primes * 4 bytes` (~12 MB when sieving 3 M primes), so size the
   pool according to your available RAM and how frequently the logs show
   "missing prime configuration" fallbacks.

 - `--gpu-launch-divisor [NUM]` override the divisor used to compute the
   preferred GPU queue depth (`default 6`, meaning `queue-size / 6`). Higher
   values delay launches; lower values force quicker kernels when the queue is
   shallow.

 - `--gpu-launch-wait-ms [NUM]` cap how long (in milliseconds) the GPU worker
   waits for the queue to reach the preferred depth before running a partial
   batch (default `50`). Set it to `0` to disable waiting entirely.

 - `-h  --help` print this information

 - `-v  --license` show license of this program

 - `--cuda-sieve-proto` enable the experimental CUDA sieve front-end. The CPU
   sieve still marks composites, but candidate enumeration and Fermat prefill
   happen on the GPU with adaptive multi-window batching. Requires building
   `gapminer-cuda` and an NVIDIA GPU.

### Experimental CUDA sieve prototype

When `--cuda-sieve-proto` is supplied (on the CUDA binary), the miner routes the
`HybridSieve` output through an experimental CUDA pipeline:

- **Bitmap ingestion** – Each `SieveItem` hands the ready bitmap to
  `GPUFermat::prototype_sieve_batch()`, which can pack up to four consecutive
  sieve windows into a single kernel launch when the GPU candidate queue is
  running low. Windows that were built from residue snapshots continue to use
  the legacy CPU scan so CUDA launches only ever mix bitmap-backed work.
- **Adaptive batching** – The GPU worker inspects `GPUWorkList` fill levels and
  opportunistically dequeues additional windows so the CUDA kernel amortizes
  transfers over more work without stalling the sieve thread. Windows that rely
  on residue snapshots are still processed individually.
- **Per-window slices** – Even when launches are batched, the code preserves a
  window-offset index so downstream Fermat/chain validation logic receives the
  exact offsets that originated from each sieve round.
- **Visibility & tuning** – Extra verbose logging (`-e`) will emit
  `CUDA sieve prototype batched N windows; candidates=...` lines so you can see
  how often batching kicks in. Combine this with `--work-items`/`--queue-size`
  to keep the GPU fed.

### Recent changes (Jan 2026)

- Diagnostics and logging improvements:
  - Device-side diagnostic dumps (candidate limbs, carries) now route to the
    extra-verbose log file instead of printing to the CLI. Enable with `-e`.
  - Submit logging now includes full `mpz_hash` and `mpz_adder` hex dumps plus
    a human-friendly `ratio = share_difficulty / target` to aid correlation
    between backends and the pool/node responses.
  - RPC (`Rpc.cpp`) and Stratum (`Stratum.cpp`) submit payloads and raw
    responses are recorded to the extra-verbose log to make it easy to match
    miner submit attempts with pool/node accept/reject messages.

- Candidate ordering alignment:
  - The CUDA prototype previously returned candidate offsets in a different
    per-window order than the OpenCL path. The host-side pipeline now sorts
    per-window absolute offsets (in `HybridSieve::create_candidates`) so CUDA
    emits deterministic ascending ordering that aligns with OpenCL's ordering
    semantics. This makes side-by-side comparisons and parity testing
    straightforward.

- CRT (Chinese sieve) quality-of-life:
  - Added backpressure to the CRT gap queue to prevent runaway heap growth; Fermat
    consumers wake producers as they drain the queue.
  - Priority scoring now prefers gaps with higher candidate density and higher
    current target, so Fermat threads focus on stronger gaps first.
  - New option `--gap-queue-limit <N>` lets you tune the queue cap at runtime
    (default `8192`). Lower it to conserve RAM/locking when `--sieve-primes` is
    large; raise it if Fermat throughput is high and you want a deeper backlog.

- Other developer tooling:
  - Added `tools/offset_difficulty` — a small deterministic utility that
    computes `PoW::difficulty()` for given `prime_base`, `target` and a list
    of offsets. Use it to compare CPU/OpenCL/CUDA difficulty results for the
    exact same candidate (helpful to prove arithmetic parity).

These changes are primarily diagnostic and ordering fixes to make the CUDA
prototype easier to compare against the OpenCL path and to reduce CLI noise
while keeping full trace logs available in the `tests` extra-verbose file.

- Recent kernel optimizations (Jan 2026):
  - `locate_window()` function optimized with binary search (O(log n)) instead
    of linear search (O(n)) for better performance with large window counts.
  - `sievePrototypeScanKernel` reverted to atomicAdd-based implementation for
    correctness and stability, avoiding shared memory layout bugs that caused
    illegal memory access and reduced batching efficiency.

See [sievePrototypeKernel.txt](sievePrototypeKernel.txt) and
[run-notes.txt](run-notes.txt) for deeper dive notes, troubleshooting tips, and
in-flight limitations.

### Sizing the GPU buffer pools

When the CUDA sieve path is enabled there are two host-resident pools that
govern how smoothly work reaches the GPU:

1. **Bitmap pool** (`--bitmap-pool-buffers`) - each entry holds a full sieve
  bitmap, which costs `sieve-size / 8` bytes of RAM. Keep this at least
  `queue-size + 2`; doubling it can eliminate "GPU queue depth ... waiting to
  batch" stalls if the CPU finishes windows faster than the GPU can consume
  them.
2. **Snapshot pool** (`--snapshot-pool-buffers`) - used only with
  `--cuda-sieve-proto`, each entry stores `sieve-primes` residues (4 bytes
  apiece). Multiply the setting by `sieve-primes * 4` to estimate its memory
  footprint. Maintaining a pool similar in size to the bitmap pool ensures the
  GPU always receives a residue snapshot instead of falling back to CPU
  scanning.

Rule of thumb: add up `(bitmap buffers x sieve-size/8)` and
`(snapshot buffers x sieve-primes x 4)`, then make sure the total fits
comfortably in system RAM. If you need to reclaim memory, shrink the snapshot
pool first - the miner will continue to run, albeit with more CPU-side fallbacks.

If the GPU queue still idles despite ample buffers, tune launch behavior with
`--gpu-launch-divisor` (lower values trigger kernels sooner) and
`--gpu-launch-wait-ms` (how long to wait before running a partial batch).

### GPU Capacity & Packing

- **What changed:** The miner now queries the GPU's actual per-launch capacity (the internal `elementsNum`) and uses that value when packing candidate offsets for Fermat tests instead of a hardcoded 512 limit. This lets the miner fill the device more effectively and avoid unnecessary throughput loss.

- **Why it matters:** Using the real device capacity preserves the number of Fermat tests run per kernel launch which directly impacts throughput. A hardcoded small cap can dramatically reduce work per launch and increase CPU/GPU imbalance.

- **Tuning tips:**
  - Raise `--work-items` and `--num-gpu-tests` together to increase queue depth and amortize host↔device transfers.
  - If you observe many "partial batch" launches, try lowering `--gpu-launch-wait-ms` or increasing `--gpu-launch-divisor` to allow more time or depth for batching.
  - 
For CUDA users, aligning the total candidates per launch to the kernel's block size (the CUDA backend uses a `kCudaBlockSize` constant, commonly 512) can improve efficiency; an optimization to enforce this alignment may be applied in future updates.

### CUDA Backend Details

- **Block size**: The CUDA path exposes a device block-size constant used for launches; the implementation uses a 512-thread block (`kCudaBlockSize`) and provides `GPUFermat::get_block_size()` to query it.
- **Operand / modulus size**: Montgomery arithmetic is implemented for a 320-bit operand (10 x 32-bit words). This corresponds to `gpu_op_size` / `kOperandSize = 10` in the CUDA code (320-bit montgomery operations).
- **Device capacity (elementsNum)**: The GPU code queries the device per-launch capacity (internal `elementsNum`) and packs candidate offsets to fill the device rather than relying on a small hard-coded cap. This yields better throughput on modern cards.
- **Experimental CUDA sieve prototype**: When `--cuda-sieve-proto` is enabled the miner batches up to several sieve windows per launch, supports residue-snapshot fallbacks, and requires proper sizing of the `--bitmap-pool-buffers` and `--snapshot-pool-buffers` settings to avoid falling back to CPU scans.
- **Memory optimizations**: GPU buffers use pinned memory for faster PCIe transfers, with fallback to pageable memory. The prototype outputs candidates as sparse offset lists for efficient processing.
- **Diagnostics & benchmarks**: The CUDA implementation includes diagnostic kernels and benchmark helpers (montgomery traces, candidate dumps, and `GPUFermat::benchmark_montgomery_mul()` / `GPUFermat::test_gpu()`) useful when investigating correctness or performance regressions. Enable extra-verbose logs (`-e`) to see prototype batching and related messages.
- **Tuning notes**: Prefer to keep total candidates per launch aligned to the device block size and tune the trio `--work-items`, `--num-gpu-tests`, and `--queue-size` together. Use `--gpu-launch-divisor` and `--gpu-launch-wait-ms` to control batching latency vs. queue depth.

These details reflect the current CUDA sources (`src/CUDAFermat.cu`, `src/HybridSieve.cpp`) and aim to help with tuning and debugging the CUDA backend.

### GPU-Specific Tuning Guide

#### NVIDIA RTX 3060 Optimization
The RTX 3060 (3584 CUDA cores, 28 SMs, 12GB GDDR6) works best with these parameter ranges:

- **--work-items (-w)**: 4096-8192 (start at 6144)
- **--num-gpu-tests (-n)**: 32-64 (start at 48)  
- **--queue-size (-z)**: 16384-32768 (start at 24576)
- **--bitmap-pool-buffers**: 2048-4096 (start at 3072)

**Example RTX 3060 configuration:**
```bash
./bin/gapminer-cuda -o pool.url -p port -u user -x pass -g -a nvidia \
  -w 8192 -n 96 --cuda-sieve-proto \
  --sieve-size 20000000 --sieve-primes 5000000 \
  --queue-size 65536 --bitmap-pool-buffers 2048 \
  --snapshot-pool-buffers 64 --gpu-launch-divisor 24 \
  --gpu-launch-wait-ms 100 --min-gap 1000 -e
```

**Tuning methodology:**
1. Start with conservative values (-w 4096, -n 32)
2. Increase -w by 1024 until performance drops or memory usage exceeds 10GB
3. Adjust -n to maintain 60-80% queue depth
4. Increase --queue-size if you see "waiting to batch" messages
5. Monitor with `nvidia-smi dmon` for PCIe utilization
6. **Troubleshooting queue starvation**: If you see frequent "waiting to batch" messages, the GPU is consuming work faster than CPU can produce it:
   - Increase `--gpu-launch-divisor` (higher values = lower launch threshold, e.g., 24-32)
   - Increase `--gpu-launch-wait-ms` (longer wait for batches, e.g., 100-200ms)
   - Reduce `-w` (fewer work items per GPU launch)
   - Increase `--sieve-size` and `--sieve-primes` (faster CPU sieve generation)

**Expected performance:** 200k-400k tests/second, 80-95% GPU utilization, 6-10GB memory usage.

### OpenCL Backend Details

- **Operand sizes / Montgomery widths**: The OpenCL kernels implement both 320-bit and 352-bit Montgomery arithmetic (10 and 11 32-bit limbs respectively). Functions are implemented as `FermatTest320` / `FermatTest352` and supporting `monSqr320`/`monMul320` and `monSqr352`/`monMul352` in `gpu/fermat.cl` and `gpu/procs.cl`.
- **Work-group / launch hints**: The `fermat_kernel*` OpenCL kernels use a required work-group size attribute (`reqd_work_group_size(256,1,1)`) for host-side launches; benchmarking kernels choose work sizes and iterate over `elementsNum` with `get_global_size(0)` loop stride.
- **Small-prime prefilter**: A small-primes prefilter (configurable list, e.g., 3,5,7,11) is used on-device in `fermatTest320` to quickly discard many composite candidates before expensive Montgomery tests.
- **Benchmark & diagnostic kernels**: `gpu/benchmarks.cl` contains microbench kernels (`squareBenchmark*`, `multiplyBenchmark*`, `fermatTestBenchMark*`) and small-prime reciprocal tables. These are useful to exercise the big-int arithmetic and tune `elementsNum` / launch sizes.
- **Data packing**: Kernels operate on packed `uint4` vectors and split big integers across those vectors for efficient 32× vector operations; operand layout matches the CPU/CUDA paths (least-significant limb first).
- **Tuning notes**: Align host batches to device capacity (`elementsNum`) and the `256` work-group size used in kernels. Use the benchmark kernels to measure device throughput for `OperandSize=10/11` paths and tune `--work-items`/`--num-gpu-tests`/`--queue-size` accordingly.

References: `gpu/fermat.cl`, `gpu/procs.cl`, and `gpu/benchmarks.cl` in the source tree.
