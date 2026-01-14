#ifndef CPU_ONLY
#ifdef USE_CUDA_BACKEND

#include "CUDAFermat.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "utils.h"

using namespace std;

bool GPUFermat::initialized = false;

namespace {
constexpr int kOperandSize = 10;
constexpr unsigned kCudaBlockSize = 512u;
constexpr int kSmallPrimeCount = 4;
constexpr int kMontgomeryShiftBits = 2 * kOperandSize * 32;  // 640

constexpr uint32_t kSmallPrimesHost[kSmallPrimeCount] = {3u, 5u, 7u, 11u};

__device__ __constant__ uint32_t kSmallPrimesDevice[kSmallPrimeCount] = {
    3u, 5u, 7u, 11u};
__device__ __constant__ uint32_t kPrimeReciprocalsDevice[kSmallPrimeCount] = {
    0x55555556u, 0x33333334u, 0x24924925u, 0x1745D175u};
__device__ __constant__ uint32_t kPrimeBaseConst[kOperandSize];
__device__ __constant__ uint32_t kHighResiduesConst[kSmallPrimeCount];

struct BigInt {
  uint32_t limb[kOperandSize];
};

uint32_t mod_high_part_host(const uint32_t *prime_base, uint32_t prime) {
  unsigned long long acc = 0ull;
  for (int idx = kOperandSize - 1; idx >= 1; --idx)
    acc = ((acc << 32) + prime_base[idx]) % prime;
  return static_cast<uint32_t>(acc);
}

#define CUDA_CHECK(cmd)                                                             \
  do {                                                                              \
    cudaError_t _err = (cmd);                                                       \
    if (_err != cudaSuccess) {                                                      \
      std::ostringstream _oss;                                                      \
      _oss << "CUDA error " << cudaGetErrorString(_err) << " at " << __FILE__     \
           << ":" << __LINE__;                                                    \
      throw std::runtime_error(_oss.str());                                         \
    }                                                                               \
  } while (0)

__device__ __forceinline__ void big_set_zero(BigInt &n) {
#pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    n.limb[i] = 0u;
}

__device__ __forceinline__ void big_copy(BigInt &dst, const BigInt &src) {
#pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    dst.limb[i] = src.limb[i];
}

__device__ __forceinline__ bool big_is_zero(const BigInt &n) {
  uint32_t acc = 0u;
#pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    acc |= n.limb[i];
  return acc == 0u;
}

__device__ __forceinline__ bool big_is_one(const BigInt &n) {
  if (n.limb[0] != 1u)
    return false;
  uint32_t acc = 0u;
#pragma unroll
  for (int i = 1; i < kOperandSize; ++i)
    acc |= n.limb[i];
  return acc == 0u;
}

__device__ __forceinline__ int big_compare(const BigInt &a, const BigInt &b) {
  for (int i = kOperandSize - 1; i >= 0; --i) {
    if (a.limb[i] > b.limb[i])
      return 1;
    if (a.limb[i] < b.limb[i])
      return -1;
  }
  return 0;
}

__device__ __forceinline__ void big_sub(BigInt &a, const BigInt &b) {
  uint64_t borrow = 0;
  for (int i = 0; i < kOperandSize; ++i) {
    uint64_t diff = static_cast<uint64_t>(a.limb[i]) - b.limb[i] - borrow;
    a.limb[i] = static_cast<uint32_t>(diff);
    borrow = (diff >> 63) & 1u;
  }
}

__device__ __forceinline__ bool big_decrement(BigInt &a) {
  for (int i = 0; i < kOperandSize; ++i) {
    if (a.limb[i] > 0u) {
      a.limb[i]--;
      for (int j = 0; j < i; ++j)
        a.limb[j] = 0xFFFFFFFFu;
      return true;
    }
    a.limb[i] = 0xFFFFFFFFu;
  }
  return false;
}

__device__ __forceinline__ void big_shift_right_one(BigInt &a) {
  uint32_t carry = 0u;
  for (int i = kOperandSize - 1; i >= 0; --i) {
    uint32_t next = a.limb[i] & 1u;
    a.limb[i] = (a.limb[i] >> 1) | (carry << 31);
    carry = next;
  }
}

__device__ __forceinline__ void big_shift_right_two(BigInt &a) {
  uint32_t carry = 0u;
  for (int i = kOperandSize - 1; i >= 0; --i) {
    uint32_t next = a.limb[i] & 0x3u;
    a.limb[i] = (a.limb[i] >> 2) | (carry << 30);
    carry = next;
  }
}

__device__ __forceinline__ uint32_t montgomery_inverse32(uint32_t n0) {
  uint32_t inv = 1u;
  for (int i = 0; i < 5; ++i) {
    uint64_t prod = static_cast<uint64_t>(n0) * inv;
    inv *= 2u - static_cast<uint32_t>(prod);
  }
  return ~inv + 1u;
}

__device__ void montgomery_mul(const BigInt &a,
                               const BigInt &b,
                               const BigInt &mod,
                               uint32_t nPrime,
                               BigInt &out) {
  const int totalLimbs = kOperandSize * 2;
  uint64_t t[totalLimbs];
#pragma unroll
  for (int i = 0; i < totalLimbs; ++i)
    t[i] = 0ull;

  for (int i = 0; i < kOperandSize; ++i) {
    uint64_t carry = 0ull;
    for (int j = 0; j < kOperandSize; ++j) {
      const int idx = i + j;
      uint64_t sum = t[idx] +
                     static_cast<uint64_t>(a.limb[j]) * b.limb[i] + carry;
      t[idx] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    t[i + kOperandSize] += carry;

    uint32_t m = static_cast<uint32_t>(t[i]) * nPrime;
    carry = 0ull;
    for (int j = 0; j < kOperandSize; ++j) {
      const int idx = i + j;
      uint64_t sum = t[idx] + static_cast<uint64_t>(m) * mod.limb[j] + carry;
      t[idx] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    t[i + kOperandSize] += carry;
  }

  BigInt tmp;
  for (int i = 0; i < kOperandSize; ++i)
    tmp.limb[i] = static_cast<uint32_t>(t[i + kOperandSize]);
  if (big_compare(tmp, mod) >= 0)
    big_sub(tmp, mod);
  big_copy(out, tmp);
}

__device__ __forceinline__ int compare_ext(const uint32_t *value,
                                            const BigInt &mod) {
  if (value[kOperandSize])
    return 1;
  for (int i = kOperandSize - 1; i >= 0; --i) {
    if (value[i] > mod.limb[i])
      return 1;
    if (value[i] < mod.limb[i])
      return -1;
  }
  return 0;
}

__device__ __forceinline__ void sub_ext(uint32_t *value, const BigInt &mod) {
  uint64_t borrow = 0ull;
  for (int i = 0; i < kOperandSize; ++i) {
    uint64_t diff = static_cast<uint64_t>(value[i]) - mod.limb[i] - borrow;
    value[i] = static_cast<uint32_t>(diff);
    borrow = (diff >> 63) & 1u;
  }
  uint64_t diff = static_cast<uint64_t>(value[kOperandSize]) - borrow;
  value[kOperandSize] = static_cast<uint32_t>(diff);
}

__device__ void compute_r2(const BigInt &mod, BigInt &r2) {
  uint32_t value[kOperandSize + 1] = {0};
  value[0] = 1u;

  for (int bit = 0; bit < kMontgomeryShiftBits; ++bit) {
    uint64_t carry = 0ull;
    for (int i = 0; i < kOperandSize + 1; ++i) {
      uint64_t sum = (static_cast<uint64_t>(value[i]) << 1) + carry;
      value[i] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    while (value[kOperandSize] || compare_ext(value, mod) >= 0)
      sub_ext(value, mod);
  }

  for (int i = 0; i < kOperandSize; ++i)
    r2.limb[i] = value[i];
}

__device__ __forceinline__ void build_candidate(BigInt &dst, uint32_t offset) {
  uint64_t sum = static_cast<uint64_t>(kPrimeBaseConst[0]) + offset;
  dst.limb[0] = static_cast<uint32_t>(sum);
  uint64_t carry = sum >> 32;
  for (int i = 1; i < kOperandSize; ++i) {
    sum = static_cast<uint64_t>(kPrimeBaseConst[i]) + carry;
    dst.limb[i] = static_cast<uint32_t>(sum);
    carry = sum >> 32;
  }
}

__device__ __forceinline__ uint32_t fast_mod_u32(uint32_t value,
                                               uint32_t prime,
                                               uint32_t recip) {
  uint32_t q = __umulhi(value, recip);
  uint32_t r = value - q * prime;
  return (r >= prime) ? r - prime : r;
}

__device__ bool quick_composite(uint32_t offset) {
  for (int i = 0; i < kSmallPrimeCount; ++i) {
    const uint32_t prime = kSmallPrimesDevice[i];
    const uint32_t recip = kPrimeReciprocalsDevice[i];
    uint32_t candidate = kHighResiduesConst[i] + fast_mod_u32(offset, prime, recip);
    if (candidate >= prime)
      candidate -= prime;
    if (candidate == 0u)
      return true;
  }
  return false;
}

__device__ bool fermat_probable_prime(const BigInt &n) {
  if ((n.limb[0] & 1u) == 0u)
    return false;
  if (big_is_zero(n))
    return false;

  BigInt exponent;
  big_copy(exponent, n);
  if (!big_decrement(exponent))
    return false;

  uint32_t nPrime = montgomery_inverse32(n.limb[0]);

  BigInt r2;
  compute_r2(n, r2);

  BigInt one;
  big_set_zero(one);
  one.limb[0] = 1u;

  BigInt oneMont;
  montgomery_mul(one, r2, n, nPrime, oneMont);

  BigInt base;
  big_set_zero(base);
  base.limb[0] = 2u;

  BigInt baseMont;
  montgomery_mul(base, r2, n, nPrime, baseMont);

  BigInt result;
  big_copy(result, oneMont);

  BigInt expCopy;
  big_copy(expCopy, exponent);
  while (!big_is_zero(expCopy)) {
    const uint32_t bits = expCopy.limb[0] & 0x3u;
    if (bits & 0x1u) {
      BigInt tmp;
      montgomery_mul(result, baseMont, n, nPrime, tmp);
      big_copy(result, tmp);
    }

    BigInt tmp;
    montgomery_mul(baseMont, baseMont, n, nPrime, tmp);
    big_copy(baseMont, tmp);

    if (bits & 0x2u) {
      BigInt tmpMul;
      montgomery_mul(result, baseMont, n, nPrime, tmpMul);
      big_copy(result, tmpMul);
    }

    BigInt tmpSquare;
    montgomery_mul(baseMont, baseMont, n, nPrime, tmpSquare);
    big_copy(baseMont, tmpSquare);

    big_shift_right_two(expCopy);
  }

  BigInt finalRes;
  montgomery_mul(result, one, n, nPrime, finalRes);
  return big_is_one(finalRes);
}

__global__ void fermatTest320Kernel(const uint32_t *numbers,
                                    uint32_t *results,
                                    uint32_t elementsNum) {
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < elementsNum;
       idx += stride) {
    const uint32_t offset = numbers[idx];

    if (quick_composite(offset)) {
      results[idx] = 0u;
      continue;
    }

    BigInt candidate;
    build_candidate(candidate, offset);
    bool probable = fermat_probable_prime(candidate);
    results[idx] = probable ? 1u : 0u;
  }
}

}  // namespace

pthread_mutex_t GPUFermat::creation_mutex = PTHREAD_MUTEX_INITIALIZER;
GPUFermat *GPUFermat::only_instance = nullptr;

unsigned GPUFermat::GroupSize = kCudaBlockSize;
unsigned GPUFermat::operandSize = kOperandSize;

unsigned GPUFermat::get_group_size() {
  return GroupSize;
}

GPUFermat::CudaBuffer::CudaBuffer()
    : Size(0), HostData(nullptr), DeviceData(nullptr), storage_() {}

GPUFermat::CudaBuffer::~CudaBuffer() { reset(); }

void GPUFermat::CudaBuffer::reset() {
  if (DeviceData) {
    cudaFree(DeviceData);
    DeviceData = nullptr;
  }
  HostData = nullptr;
  Size = 0;
  storage_.clear();
}

void GPUFermat::CudaBuffer::init(size_t size) {
  reset();
  Size = size;
  storage_.assign(Size, 0u);
  HostData = storage_.data();
  CUDA_CHECK(cudaMalloc(&DeviceData, Size * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(DeviceData, 0, Size * sizeof(uint32_t)));
}

void GPUFermat::CudaBuffer::copyToDevice(cudaStream_t stream,
                                         size_t count,
                                         size_t offset) {
  if (!DeviceData || !HostData)
    return;
  const size_t elems = (count == 0) ? (Size - offset) : count;
  if (elems == 0)
    return;
  CUDA_CHECK(cudaMemcpyAsync(DeviceData + offset,
                             HostData + offset,
                             elems * sizeof(uint32_t),
                             cudaMemcpyHostToDevice,
                             stream));
}

void GPUFermat::CudaBuffer::copyToHost(cudaStream_t stream,
                                       size_t count,
                                       size_t offset) {
  if (!DeviceData || !HostData)
    return;
  const size_t elems = (count == 0) ? (Size - offset) : count;
  if (elems == 0)
    return;
  CUDA_CHECK(cudaMemcpyAsync(HostData + offset,
                             DeviceData + offset,
                             elems * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost,
                             stream));
}

GPUFermat::GPUFermat(unsigned device_id,
                     const char *platformId,
                     unsigned workItems)
    : workItems(workItems),
      elementsNum(0),
      numberLimbsNum(0),
      groupsNum(0),
  streams{nullptr, nullptr},
  computeUnits(0),
  smallPrimeResidues{0u, 0u, 0u, 0u} {
  log_str("Creating CUDA GPUFermat", LOG_D);
  if (workItems == 0)
    throw std::runtime_error("workItems must be greater than zero");
  if (!init_cuda(device_id, platformId))
    throw std::runtime_error("Failed to initialize CUDA backend");

  elementsNum = GroupSize * workItems;
  numberLimbsNum = elementsNum * operandSize;
  groupsNum = std::max(1u, computeUnits * 8u);

  initializeBuffers();
}

GPUFermat::~GPUFermat() {
  for (cudaStream_t &s : streams) {
    if (s) {
      cudaStreamDestroy(s);
      s = nullptr;
    }
  }
  numbers.reset();
  gpuResults.reset();
  primeBase.reset();
}

bool GPUFermat::init_cuda(unsigned device_id, const char *platformId) {
  const char *requested = platformId ? platformId : "cuda";
  if (requested && strcmp(requested, "cuda") != 0 && strcmp(requested, "nvidia") != 0) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "ERROR: unsupported platform '" << requested
         << "' for CUDA backend" << endl;
    pthread_mutex_unlock(&io_mutex);
    return false;
  }

  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount <= 0) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "ERROR: no CUDA devices detected" << endl;
    pthread_mutex_unlock(&io_mutex);
    return false;
  }

  if (device_id >= static_cast<unsigned>(deviceCount)) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "ERROR: requested CUDA device " << device_id
         << " but only " << deviceCount << " device(s) available" << endl;
    pthread_mutex_unlock(&io_mutex);
    return false;
  }

  CUDA_CHECK(cudaSetDevice(static_cast<int>(device_id)));
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, static_cast<int>(device_id)));
  computeUnits = props.multiProcessorCount;
  for (cudaStream_t &s : streams)
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  pthread_mutex_lock(&io_mutex);
  cout << get_time() << "Using CUDA GPU " << device_id << " [" << props.name
       << "] with " << computeUnits << " SMs" << endl;
  pthread_mutex_unlock(&io_mutex);
  return true;
}

void GPUFermat::initializeBuffers() {
  numbers.init(elementsNum);
  gpuResults.init(elementsNum);
  primeBase.init(operandSize);
}

void GPUFermat::uploadPrimeBaseConstants() {
  if (!primeBase.HostData)
    return;

  CUDA_CHECK(cudaMemcpyToSymbol(kPrimeBaseConst,
                                primeBase.HostData,
                                operandSize * sizeof(uint32_t),
                                0,
                                cudaMemcpyHostToDevice));

  for (int i = 0; i < kSmallPrimeCount; ++i)
    smallPrimeResidues[i] =
        mod_high_part_host(primeBase.HostData, kSmallPrimesHost[i]);

  CUDA_CHECK(cudaMemcpyToSymbol(kHighResiduesConst,
                                smallPrimeResidues.data(),
                                kSmallPrimeCount * sizeof(uint32_t),
                                0,
                                cudaMemcpyHostToDevice));
}

GPUFermat *GPUFermat::get_instance(unsigned device_id,
                                   const char *platformId,
                                   unsigned workItems) {
  pthread_mutex_lock(&creation_mutex);
  if (!initialized && device_id != static_cast<unsigned>(-1) &&
      platformId != nullptr && workItems != 0) {
    only_instance = new GPUFermat(device_id, platformId, workItems);
    initialized = true;
  }
  pthread_mutex_unlock(&creation_mutex);
  return only_instance;
}

void GPUFermat::destroy_instance() {
  pthread_mutex_lock(&creation_mutex);
  delete only_instance;
  only_instance = nullptr;
  initialized = false;
  pthread_mutex_unlock(&creation_mutex);
}

uint32_t *GPUFermat::get_results_buffer() { return gpuResults.HostData; }

uint32_t *GPUFermat::get_prime_base_buffer() { return primeBase.HostData; }

uint32_t *GPUFermat::get_candidates_buffer() { return numbers.HostData; }

void GPUFermat::run_cuda(uint32_t batchElements) {
  if (batchElements == 0)
    return;
  const uint32_t work = std::min(batchElements, elementsNum);
  log_str("running " + std::to_string(work) + " fermat tests on the gpu", LOG_D);
  uploadPrimeBaseConstants();

  const uint32_t threads = GroupSize;
  const uint32_t maxBlocks = std::max(1u, groupsNum * 4u);
  const uint32_t streamCount = static_cast<uint32_t>(streams.size());
  const uint32_t chunkSize =
      std::max<uint32_t>(threads, (work + streamCount - 1u) / streamCount);

  uint32_t processed = 0;
  uint32_t chunkIndex = 0;
  while (processed < work) {
    const uint32_t chunk = std::min(chunkSize, work - processed);
    cudaStream_t chunkStream = streams[chunkIndex % streamCount];

    numbers.copyToDevice(chunkStream, chunk, processed);

    const uint32_t requiredBlocks = (chunk + threads - 1u) / threads;
    const uint32_t blocks = std::max(1u, std::min(requiredBlocks, maxBlocks));

    fermatTest320Kernel<<<blocks, threads, 0, chunkStream>>>(
        numbers.DeviceData + processed,
        gpuResults.DeviceData + processed,
        chunk);
    CUDA_CHECK(cudaGetLastError());

    gpuResults.copyToHost(chunkStream, chunk, processed);

    processed += chunk;
    ++chunkIndex;
  }

  for (cudaStream_t s : streams)
    CUDA_CHECK(cudaStreamSynchronize(s));
}

void GPUFermat::fermat_gpu(uint32_t elements) {
  const uint32_t work = (elements == 0) ? elementsNum : elements;
  run_cuda(work);
}

void GPUFermat::benchmark() { test_gpu(); }

uint32_t GPUFermat::rand32() {
  uint32_t value = rand();
  value = (value << 16) | rand();
  return value;
}

void GPUFermat::test_gpu() {
  if (!elementsNum) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "test_gpu skipped: no elements configured" << endl;
    pthread_mutex_unlock(&io_mutex);
    return;
  }

  if (!primeBase.HostData || !numbers.HostData || !gpuResults.HostData) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "test_gpu aborted: host buffers not initialized" << endl;
    pthread_mutex_unlock(&io_mutex);
    return;
  }

  const unsigned size = elementsNum;
  uint32_t *prime_base = primeBase.HostData;
  uint32_t *primes = numbers.HostData;
  uint32_t *results = gpuResults.HostData;

  mpz_class mpz(rand32());
  for (int i = 0; i < 8; ++i) {
    mpz <<= 32;
    mpz += rand32();
  }

  if (mpz.get_ui() & 0x1)
    mpz += 1;

  memset(prime_base, 0, operandSize * sizeof(uint32_t));
  size_t exported_size = 0;
  mpz_export(prime_base,
             &exported_size,
             -1,
             sizeof(uint32_t),
             0,
             0,
             mpz.get_mpz_t());

  for (unsigned i = 0; i < size; ++i) {
    primes[i] = mpz.get_ui() & 0xffffffffu;

    if (i % 2 == 0)
      mpz_nextprime(mpz.get_mpz_t(), mpz.get_mpz_t());
    else
      mpz_add_ui(mpz.get_mpz_t(), mpz.get_mpz_t(), 1);

    if (i % 23 == 0)
      printf("\rcreating test data: %u  \r", size - i);
  }
  printf("\r                                             \r");

  fermat_gpu(size);

  unsigned failures = 0;
  for (unsigned i = 0; i < size; ++i) {
    const uint32_t expected = (i % 2 == 0) ? 0u : 1u;
    if (results[i] != expected) {
      if (failures == 0) {
        printf("Result %u is wrong: expected %u but got %u\n",
               i,
               expected,
               results[i]);
      }
      failures++;
    }
  }

  if (failures)
    printf("GPU Test failed: %u mismatched result(s)\n", failures);
  else
    printf("GPU Test worked\n");
}

#endif /* USE_CUDA_BACKEND */
#endif /* CPU_ONLY */
