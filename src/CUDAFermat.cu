#ifndef CPU_ONLY
#ifdef USE_CUDA_BACKEND

#include "CUDAFermat.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "utils.h"
#include "Opts.h"

using namespace std;

bool GPUFermat::initialized = false;

// expose CUDA block size as block size getter
unsigned GPUFermat::get_block_size() {
  return 512u;
}

namespace {
constexpr int kOperandSize = 10;
constexpr unsigned kCudaBlockSize = 512u;
// Increase small-prime quick-filter count to improve cheap composite rejection on device
constexpr int kSmallPrimeCount = 64;
constexpr int kMontgomeryShiftBits = 2 * kOperandSize * 32;  // 640

template <typename T>
T *buffer_cast(uint8_t *ptr) {
  return reinterpret_cast<T *>(ptr);
}

template <typename T>
const T *buffer_cast(const uint8_t *ptr) {
  return reinterpret_cast<const T *>(ptr);
}

using ResultWord = GPUFermat::ResultWord;

constexpr uint32_t kSmallPrimesHost[kSmallPrimeCount] = {
  3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u,
  29u, 31u, 37u, 41u, 43u, 47u, 53u, 59u,
  61u, 67u, 71u, 73u, 79u, 83u, 89u, 97u,
  101u, 103u, 107u, 109u, 113u, 127u, 131u, 137u,
  139u, 149u, 151u, 157u, 163u, 167u, 173u, 179u,
  181u, 191u, 193u, 197u, 199u, 211u, 223u, 227u,
  229u, 233u, 239u, 241u, 251u, 257u, 263u, 269u,
  271u, 277u, 281u, 283u, 293u, 307u, 311u, 313u};

struct PrototypeWindowDesc {
  uint32_t word_start;
  uint32_t word_count;
  uint32_t bit_length;
  uint32_t output_base;
  uint32_t base_parity;
  uint32_t residue_offset;
  uint32_t residue_count;
};

__device__ __constant__ uint32_t kSmallPrimesDevice[kSmallPrimeCount] = {
  3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u,
  29u, 31u, 37u, 41u, 43u, 47u, 53u, 59u,
  61u, 67u, 71u, 73u, 79u, 83u, 89u, 97u,
  101u, 103u, 107u, 109u, 113u, 127u, 131u, 137u,
  139u, 149u, 151u, 157u, 163u, 167u, 173u, 179u,
  181u, 191u, 193u, 197u, 199u, 211u, 223u, 227u,
  229u, 233u, 239u, 241u, 251u, 257u, 263u, 269u,
  271u, 277u, 281u, 283u, 293u, 307u, 311u, 313u};
// Reciprocals are filled at runtime in uploadPrimeBaseConstants to keep values
// computed correctly for the prime list size and to avoid manual hex constants.
__device__ __constant__ uint32_t kPrimeReciprocalsDevice[kSmallPrimeCount];
__device__ __constant__ uint32_t kPrimeBaseConst[kOperandSize];
__device__ __constant__ uint32_t kHighResiduesConst[kSmallPrimeCount];
__device__ __constant__ uint32_t kUseCombaConst;

struct BigInt {
  uint32_t limb[kOperandSize];
};

uint32_t mod_high_part_host(const uint32_t *prime_base, uint32_t prime) {
  unsigned long long acc = 0ull;
  for (int idx = kOperandSize - 1; idx >= 0; --idx)
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

// Diagnostic kernel: reconstruct candidate limbs on-device for a set of indices
__global__ void dump_candidate_limbs_kernel(const uint32_t *numbers,
                                            const uint32_t *sample_indices,
                                            uint32_t *out_limbs,
                                            unsigned sampleCount) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sampleCount)
    return;

  const uint32_t idx = sample_indices[tid];
  const uint32_t off = numbers[idx];

  // Reconstruct: add offset into least-significant limb and propagate carry
  uint64_t carry = off;
  for (int i = 0; i < kOperandSize; ++i) {
    uint64_t sum = static_cast<uint64_t>(kPrimeBaseConst[i]) + carry;
    out_limbs[tid * kOperandSize + i] = static_cast<uint32_t>(sum & 0xffffffffu);
    carry = sum >> 32;
  }
  // write final carry (0 or small) after limbs
  out_limbs[sampleCount * kOperandSize + tid] = static_cast<uint32_t>(carry & 0xffffffffu);
}

// Diagnostic kernel: record carry after each limb addition for given samples
__global__ void dump_candidate_carries_kernel(const uint32_t *numbers,
                                              const uint32_t *sample_indices,
                                              uint32_t *out_carries,
                                              unsigned sampleCount) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sampleCount)
    return;

  const uint32_t idx = sample_indices[tid];
  const uint32_t off = numbers[idx];

  uint64_t carry = off;
  for (int i = 0; i < kOperandSize; ++i) {
    uint64_t sum = static_cast<uint64_t>(kPrimeBaseConst[i]) + carry;
    carry = sum >> 32;
    out_carries[tid * kOperandSize + i] = static_cast<uint32_t>(carry & 0xffffffffu);
  }
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

__host__ __device__ __forceinline__ uint32_t montgomery_inverse32(uint32_t n0) {
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

  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i) {
    uint64_t carry = 0ull;
    #pragma unroll
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
    #pragma unroll
    for (int j = 0; j < kOperandSize; ++j) {
      const int idx = i + j;
      uint64_t sum = t[idx] + static_cast<uint64_t>(m) * mod.limb[j] + carry;
      t[idx] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    t[i + kOperandSize] += carry;
  }

  BigInt tmp;
  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    tmp.limb[i] = static_cast<uint32_t>(t[i + kOperandSize]);
  if (big_compare(tmp, mod) >= 0)
    big_sub(tmp, mod);
  big_copy(out, tmp);
}

// CIOS Montgomery multiply using t[N] + high carry (smaller temp footprint than t[2N]).
__device__ void montgomery_mul_cios(const BigInt &a,
                                    const BigInt &b,
                                    const BigInt &mod,
                                    uint32_t nPrime,
                                    BigInt &out) {
  uint32_t t[kOperandSize];
  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    t[i] = 0u;
  uint64_t t_high = 0ull;

  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i) {
    uint64_t carry = 0ull;
    #pragma unroll
    for (int j = 0; j < kOperandSize; ++j) {
      uint64_t sum = static_cast<uint64_t>(t[j]) +
                     static_cast<uint64_t>(a.limb[i]) * b.limb[j] + carry;
      t[j] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    t_high += carry;

    const uint32_t m = static_cast<uint32_t>(t[0]) * nPrime;
    carry = 0ull;
    #pragma unroll
    for (int j = 0; j < kOperandSize; ++j) {
      uint64_t sum = static_cast<uint64_t>(t[j]) +
                     static_cast<uint64_t>(m) * mod.limb[j] + carry;
      t[j] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    t_high += carry;

    #pragma unroll
    for (int j = 0; j < kOperandSize - 1; ++j)
      t[j] = t[j + 1];
    t[kOperandSize - 1] = static_cast<uint32_t>(t_high);
    t_high >>= 32;
  }

  BigInt tmp;
  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    tmp.limb[i] = t[i];
  if (big_compare(tmp, mod) >= 0)
    big_sub(tmp, mod);
  big_copy(out, tmp);
}

__device__ __forceinline__ uint32_t mul_hi_u32(uint32_t a, uint32_t b) {
  return __umulhi(a, b);
}

__device__ __forceinline__ void mon_sub_320(uint32_t *a,
                                            const uint32_t *b,
                                            int size) {
  uint32_t prev = a[0];
  a[0] -= b[0];
  #pragma unroll
  for (int i = 1; i < size; ++i) {
    uint32_t current = a[i];
    uint32_t borrow = (a[i - 1] > prev) ? 1u : 0u;
    a[i] -= b[i] + borrow;
    prev = current;
  }
}

__device__ void monMul320_words(uint32_t *op1,
                                const uint32_t *op2,
                                const uint32_t *mod,
                                uint32_t invm) {
  const int totalLimbs = kOperandSize * 2;
  uint64_t t[20];
#pragma unroll
  for (int i = 0; i < totalLimbs; ++i) t[i] = 0ull;

  for (int i = 0; i < kOperandSize; ++i) {
    uint64_t carry = 0ull;
    for (int j = 0; j < kOperandSize; ++j) {
      const int idx = i + j;
      uint64_t sum = t[idx] + static_cast<uint64_t>(op1[j]) * static_cast<uint64_t>(op2[i]) + carry;
      t[idx] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    t[i + kOperandSize] += carry;

    uint32_t m = static_cast<uint32_t>(t[i]) * invm;
    carry = 0ull;
    for (int j = 0; j < kOperandSize; ++j) {
      const int idx = i + j;
      uint64_t sum = t[idx] + static_cast<uint64_t>(m) * static_cast<uint64_t>(mod[j]) + carry;
      t[idx] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    t[i + kOperandSize] += carry;
  }

  uint32_t tmp[kOperandSize];
  for (int i = 0; i < kOperandSize; ++i)
    tmp[i] = static_cast<uint32_t>(t[i + kOperandSize]);

  BigInt tmpBig;
  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    tmpBig.limb[i] = tmp[i];
  BigInt modBig;
  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    modBig.limb[i] = mod[i];
  if (big_compare(tmpBig, modBig) >= 0)
    big_sub(tmpBig, modBig);

  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    op1[i] = tmpBig.limb[i];
}

// Forward declarations used by trace helper (defined later in file)
__device__ __forceinline__ uint32_t build_candidate(BigInt &dst, uint32_t offset);
__device__ void compute_r2(const BigInt &mod, BigInt &r2);

// Trace variant: runs same algorithm but records invValue[] and final op1 into output buffers
// trace outputs: out_inv[kOperandSize], out_final[kOperandSize]
// acc_low / acc_high each are arrays of size 4 (blocks 0..3)
// Also capture three 64-bit products for block1: prod0=op1[0]*op2[1], prod1=op1[1]*op2[0], prod2=invValue[0]*mod[1]
// Capture acc state after each product addition for block1: step0 (after prod0), step1 (after prod1), step2 (after prod2)
__device__ void monMul320_words_trace(uint32_t *op1,
                                      const uint32_t *op2,
                                      const uint32_t *mod,
                                      uint32_t invm,
                                      uint32_t *out_inv,
                                      uint32_t *out_final,
                                      uint32_t *out_acc_low,
                                      uint32_t *out_acc_high,
                                      uint32_t *out_prod_low,
                                      uint32_t *out_prod_high,
                                      uint32_t *out_steps_low,
                                      uint32_t *out_steps_high) {
  uint32_t invValue[10];
  uint64_t accLow = 0, accHi = 0;
  // record accBeforeInv for blocks 0..3
  uint64_t accBefore[4];
  {
    accLow += static_cast<uint64_t>(op1[0]) * op2[0];
    accHi += mul_hi_u32(op1[0], op2[0]);
    // capture acc before computing invValue[0]
    accBefore[0] = accLow;
    invValue[0] = invm * static_cast<uint32_t>(accLow);
    accLow += static_cast<uint64_t>(invValue[0]) * mod[0];
    accHi += mul_hi_u32(invValue[0], mod[0]);
    uint32_t high32 = static_cast<uint32_t>(accLow >> 32);
    accHi += static_cast<uint64_t>(high32);
    accLow = accHi;
    accHi = 0;
  }
  {
    accLow += static_cast<uint64_t>(op1[0]) * op2[1];
    accHi += mul_hi_u32(op1[0], op2[1]);
    accLow += static_cast<uint64_t>(op1[1]) * op2[0];
    accHi += mul_hi_u32(op1[1], op2[0]);
    accLow += static_cast<uint64_t>(invValue[0]) * mod[1];
    accHi += mul_hi_u32(invValue[0], mod[1]);
    // compute per-term products for block1
    uint64_t prod0 = static_cast<uint64_t>(op1[0]) * static_cast<uint64_t>(op2[1]);
    uint64_t prod1 = static_cast<uint64_t>(op1[1]) * static_cast<uint64_t>(op2[0]);
    uint64_t prod2 = static_cast<uint64_t>(invValue[0]) * static_cast<uint64_t>(mod[1]);
    // add prod1 first (match modified unrolled ordering) and capture
    accLow += prod1;
    accHi += static_cast<uint64_t>(prod1 >> 32);
    out_steps_low[0] = static_cast<uint32_t>(accLow & 0xffffffffu);
    out_steps_high[0] = static_cast<uint32_t>(accLow >> 32);
    out_prod_low[1] = static_cast<uint32_t>(prod1 & 0xffffffffu);
    out_prod_high[1] = static_cast<uint32_t>(prod1 >> 32);
    // add prod0 and capture
    accLow += prod0;
    accHi += static_cast<uint64_t>(prod0 >> 32);
    out_steps_low[1] = static_cast<uint32_t>(accLow & 0xffffffffu);
    out_steps_high[1] = static_cast<uint32_t>(accLow >> 32);
    out_prod_low[0] = static_cast<uint32_t>(prod0 & 0xffffffffu);
    out_prod_high[0] = static_cast<uint32_t>(prod0 >> 32);
    // add prod2 and capture (this is accBefore used to compute invValue[1])
    accLow += prod2;
    accHi += static_cast<uint64_t>(prod2 >> 32);
    out_steps_low[2] = static_cast<uint32_t>(accLow & 0xffffffffu);
    out_steps_high[2] = static_cast<uint32_t>(accLow >> 32);
    out_prod_low[2] = static_cast<uint32_t>(prod2 & 0xffffffffu);
    out_prod_high[2] = static_cast<uint32_t>(prod2 >> 32);

    invValue[1] = invm * static_cast<uint32_t>(accLow);
    accLow += static_cast<uint64_t>(invValue[1]) * mod[0];
    accHi += mul_hi_u32(invValue[1], mod[0]);
    uint32_t high32 = static_cast<uint32_t>(accLow >> 32);
    accHi += static_cast<uint64_t>(high32);
    accLow = accHi;
    accHi = 0;
  }
  {
    accLow += static_cast<uint64_t>(op1[0]) * op2[2];
    accHi += mul_hi_u32(op1[0], op2[2]);
    accLow += static_cast<uint64_t>(op1[1]) * op2[1];
    accHi += mul_hi_u32(op1[1], op2[1]);
    accLow += static_cast<uint64_t>(op1[2]) * op2[0];
    accHi += mul_hi_u32(op1[2], op2[0]);
    accLow += static_cast<uint64_t>(invValue[0]) * mod[2];
    accHi += mul_hi_u32(invValue[0], mod[2]);
    accLow += static_cast<uint64_t>(invValue[1]) * mod[1];
    accHi += mul_hi_u32(invValue[1], mod[1]);
    // capture acc before computing invValue[2]
    accBefore[2] = accLow;
    invValue[2] = invm * static_cast<uint32_t>(accLow);
    accLow += static_cast<uint64_t>(invValue[2]) * mod[0];
    accHi += mul_hi_u32(invValue[2], mod[0]);
    uint32_t high32 = static_cast<uint32_t>(accLow >> 32);
    accHi += static_cast<uint64_t>(high32);
    accLow = accHi;
    accHi = 0;
  }
  // capture block 3 acc as well in the following unrolled block sequence
#pragma unroll
  for (int blk = 3; blk <= 9; ++blk) {
    if (blk == 3) {
      // compute contributions for block 3 exactly like the main implementation
      // We will capture accBefore[3] just before invValue[3] assignment below by pattern
    }
    invValue[blk] = 0u;
  }

  // Finish by running the original tail sequence to compute final op1 entries
  monMul320_words(op1, op2, mod, invm);

  // write invValue and final op1 to output
  for (int i = 0; i < kOperandSize; ++i) {
    out_inv[i] = invValue[i];
    out_final[i] = op1[i];
  }

  // write captured accBefore entries (blocks 0..3) to out arrays
  // ensure block 3 is initialized (trace currently captures 0..2)
  accBefore[3] = 0ull;

  for (int b = 0; b < 4; ++b) {
    out_acc_low[b] = static_cast<uint32_t>(accBefore[b] & 0xffffffffu);
    out_acc_high[b] = static_cast<uint32_t>(accBefore[b] >> 32);
  }
}

__global__ void montgomeryTraceKernel(const uint32_t *numbers,
                                      const uint32_t *sample_indices,
                                      uint32_t *out_inv,
                                      uint32_t *out_final,
                                      uint32_t *out_acc_low,
                                      uint32_t *out_acc_high,
                                      uint32_t *out_prod_low,
                                      uint32_t *out_prod_high,
                                      uint32_t *out_steps_low,
                                      uint32_t *out_steps_high,
                                      unsigned sampleCount) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sampleCount)
    return;

  const uint32_t idx = sample_indices[tid];
  const uint32_t off = numbers[idx];

  BigInt n;
  uint32_t final_carry = build_candidate(n, off);
  if (final_carry != 0u) {
    for (int i = 0; i < kOperandSize; ++i) {
      out_inv[tid * kOperandSize + i] = 0u;
      out_final[tid * kOperandSize + i] = 0u;
    }
    return;
  }

  BigInt r2;
  compute_r2(n, r2);

  uint32_t op1_local[kOperandSize * 2] = {0};
  uint32_t op2_local[kOperandSize];
  uint32_t mod_local[kOperandSize];
  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i) {
    op1_local[i] = 0u;
    op2_local[i] = r2.limb[i];
    mod_local[i] = n.limb[i];
  }
  op1_local[0] = 1u;

  uint32_t nPrime = montgomery_inverse32(n.limb[0]);

  monMul320_words_trace(op1_local, op2_local, mod_local, nPrime,
                        &out_inv[tid * kOperandSize], &out_final[tid * kOperandSize],
                        &out_acc_low[tid * 4], &out_acc_high[tid * 4],
                        &out_prod_low[tid * 3], &out_prod_high[tid * 3],
                        &out_steps_low[tid * 3], &out_steps_high[tid * 3]);
}

// Classic/reference traced variant: capture m (invValue) and final op1 from the classic montgomery implementation
__device__ void monMul320_words_classic_trace_device(const BigInt &a,
                                                     const BigInt &b,
                                                     const BigInt &mod,
                                                     uint32_t nPrime,
                                                     uint32_t invValueOut[10],
                                                     uint32_t op1_out[10],
                                                     uint32_t acc_low_out[4],
                                                     uint32_t acc_high_out[4],
                                                     uint32_t prod_low_out[3],
                                                     uint32_t prod_high_out[3],
                                                     uint32_t steps_low_out[3],
                                                     uint32_t steps_high_out[3]) {
  const int totalLimbs = kOperandSize * 2;
  uint64_t t[totalLimbs];
#pragma unroll
  for (int i = 0; i < totalLimbs; ++i)
    t[i] = 0ull;

  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i) {
    uint64_t carry = 0ull;
    #pragma unroll
    for (int j = 0; j < kOperandSize; ++j) {
      const int idx = i + j;
      uint64_t sum = t[idx] + static_cast<uint64_t>(a.limb[j]) * b.limb[i] + carry;
      t[idx] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    t[i + kOperandSize] += carry;

    // capture t[i] (accumulator before reduction) for tracing
    if (i < 4) {
      acc_low_out[i] = static_cast<uint32_t>(t[i] & 0xffffffffull);
      acc_high_out[i] = static_cast<uint32_t>(t[i] >> 32);
    }
    // for block1, capture intermediate t[1] after key steps
    if (i == 0) {
      // after finishing i==0 loop, t[1] includes prod1 (a.limb[1]*b.limb[0])
      steps_low_out[0] = static_cast<uint32_t>(t[1] & 0xffffffffull);
      steps_high_out[0] = static_cast<uint32_t>(t[1] >> 32);
    }
    if (i == 1) {
      // before reduction here t[1] includes prod0 + prod1 and other contributions
      steps_low_out[1] = static_cast<uint32_t>(t[1] & 0xffffffffull);
      steps_high_out[1] = static_cast<uint32_t>(t[1] >> 32);
      // capture per-term products for block1 for comparison
      uint64_t prod0 = static_cast<uint64_t>(a.limb[0]) * static_cast<uint64_t>(b.limb[1]);
      uint64_t prod1 = static_cast<uint64_t>(a.limb[1]) * static_cast<uint64_t>(b.limb[0]);
      uint64_t prod2 = static_cast<uint64_t>(invValueOut[0]) * static_cast<uint64_t>(mod.limb[1]);
      prod_low_out[0] = static_cast<uint32_t>(prod0 & 0xffffffffu);
      prod_high_out[0] = static_cast<uint32_t>(prod0 >> 32);
      prod_low_out[1] = static_cast<uint32_t>(prod1 & 0xffffffffu);
      prod_high_out[1] = static_cast<uint32_t>(prod1 >> 32);
      prod_low_out[2] = static_cast<uint32_t>(prod2 & 0xffffffffu);
      prod_high_out[2] = static_cast<uint32_t>(prod2 >> 32);
    }
    uint32_t m = static_cast<uint32_t>(t[i]) * nPrime;
    invValueOut[i] = m;

    carry = 0ull;
    #pragma unroll
    for (int j = 0; j < kOperandSize; ++j) {
      const int idx = i + j;
      uint64_t sum = t[idx] + static_cast<uint64_t>(m) * mod.limb[j] + carry;
      t[idx] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    t[i + kOperandSize] += carry;
  }

  BigInt tmp;
  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    tmp.limb[i] = static_cast<uint32_t>(t[i + kOperandSize]);
  if (big_compare(tmp, mod) >= 0)
    big_sub(tmp, mod);

  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    op1_out[i] = tmp.limb[i];
}

__global__ void montgomeryClassicTraceKernel(const uint32_t *numbers,
                                             const uint32_t *sample_indices,
                                             uint32_t *out_inv,
                                             uint32_t *out_final,
                                             uint32_t *out_acc_low,
                                             uint32_t *out_acc_high,
                                             uint32_t *out_prod_low,
                                             uint32_t *out_prod_high,
                                             uint32_t *out_steps_low,
                                             uint32_t *out_steps_high,
                                             unsigned sampleCount) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sampleCount)
    return;

  const uint32_t idx = sample_indices[tid];
  const uint32_t off = numbers[idx];

  BigInt n;
  uint32_t final_carry = build_candidate(n, off);
  if (final_carry != 0u) {
    for (int i = 0; i < kOperandSize; ++i) {
      out_inv[tid * kOperandSize + i] = 0u;
      out_final[tid * kOperandSize + i] = 0u;
    }
    return;
  }

  BigInt r2;
  compute_r2(n, r2);

  BigInt a;
  big_set_zero(a);
  a.limb[0] = 1u; // oneMont equivalent input

  // For consistency with the unrolled trace, multiply one * r2 (like the other trace path)
  uint32_t nPrime = montgomery_inverse32(n.limb[0]);

  uint32_t invValueOut[10];
  uint32_t op1_out[10];
  monMul320_words_classic_trace_device(a, r2, n, nPrime, invValueOut, op1_out,
                                       &out_acc_low[tid * 4], &out_acc_high[tid * 4],
                                       &out_prod_low[tid * 3], &out_prod_high[tid * 3],
                                       &out_steps_low[tid * 3], &out_steps_high[tid * 3]);

  // store results
  for (int i = 0; i < kOperandSize; ++i) {
    out_inv[tid * kOperandSize + i] = invValueOut[i];
    out_final[tid * kOperandSize + i] = op1_out[i];
  }
}

__device__ void montgomery_mul_unrolled(const BigInt &a,
                                         const BigInt &b,
                                         const BigInt &mod,
                                         uint32_t nPrime,
                                         BigInt &out) {
  montgomery_mul(a, b, mod, nPrime, out);
}

__device__ __forceinline__ void montgomery_mul_fast(const BigInt &a,
                                                    const BigInt &b,
                                                    const BigInt &mod,
                                                    uint32_t nPrime,
                                                    BigInt &out) {
  if (kUseCombaConst)
    montgomery_mul_cios(a, b, mod, nPrime, out);
  else
    montgomery_mul_unrolled(a, b, mod, nPrime, out);
}

__device__ __forceinline__ int compare_ext(const uint32_t *value,
                                            const BigInt &mod) {
  if (value[kOperandSize])
    return 1;
  #pragma unroll
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
  #pragma unroll
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
    #pragma unroll
    for (int i = 0; i < kOperandSize + 1; ++i) {
      uint64_t sum = (static_cast<uint64_t>(value[i]) << 1) + carry;
      value[i] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    while (value[kOperandSize] || compare_ext(value, mod) >= 0)
      sub_ext(value, mod);
  }

  #pragma unroll
  for (int i = 0; i < kOperandSize; ++i)
    r2.limb[i] = value[i];
}

__device__ __forceinline__ uint32_t build_candidate(BigInt &dst, uint32_t offset) {
  uint64_t sum = static_cast<uint64_t>(kPrimeBaseConst[0]) + offset;
  dst.limb[0] = static_cast<uint32_t>(sum);
  uint64_t carry = sum >> 32;
  #pragma unroll
  for (int i = 1; i < kOperandSize; ++i) {
    sum = static_cast<uint64_t>(kPrimeBaseConst[i]) + carry;
    dst.limb[i] = static_cast<uint32_t>(sum);
    carry = sum >> 32;
  }
  return static_cast<uint32_t>(carry & 0xffffffffu);
}

__device__ __forceinline__ uint32_t fast_mod_u32(uint32_t value,
                                               uint32_t prime,
                                               uint32_t recip) {
  uint32_t q = __umulhi(value, recip);
  uint32_t r = value - q * prime;
  return (r >= prime) ? r - prime : r;
}

__device__ bool quick_composite(uint32_t offset) {
  #pragma unroll
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
  montgomery_mul_fast(one, r2, n, nPrime, oneMont);

  BigInt base;
  big_set_zero(base);
  base.limb[0] = 2u;

  BigInt baseMont;
  montgomery_mul_fast(base, r2, n, nPrime, baseMont);

  BigInt result;
  big_copy(result, oneMont);

  BigInt expCopy;
  big_copy(expCopy, exponent);
  while (!big_is_zero(expCopy)) {
    const uint32_t bits = expCopy.limb[0] & 0x3u;
    if (bits & 0x1u) {
      BigInt tmp;
      montgomery_mul_fast(result, baseMont, n, nPrime, tmp);
      big_copy(result, tmp);
    }

    BigInt tmp;
    montgomery_mul_fast(baseMont, baseMont, n, nPrime, tmp);
    big_copy(baseMont, tmp);

    if (bits & 0x2u) {
      BigInt tmpMul;
      montgomery_mul_fast(result, baseMont, n, nPrime, tmpMul);
      big_copy(result, tmpMul);
    }

    BigInt tmpSquare;
    montgomery_mul_fast(baseMont, baseMont, n, nPrime, tmpSquare);
    big_copy(baseMont, tmpSquare);

    big_shift_right_two(expCopy);
  }

  BigInt finalRes;
  montgomery_mul_fast(result, one, n, nPrime, finalRes);
  return big_is_one(finalRes);
}

__global__ void sievePrototypeKernel(const uint32_t *bitmap,
                                     uint32_t total_bits,
                                     uint32_t base_parity,
                                     uint32_t *out_offsets) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_bits)
    return;

  const uint32_t word = bitmap[idx >> 5];
  const uint32_t bit = (word >> (idx & 31)) & 1u;
  const bool is_even = (((idx + base_parity) & 1u) == 0u);
  const uint32_t sentinel = 0xFFFFFFFFu;
  if (bit || is_even) {
    out_offsets[idx] = sentinel;
    return;
  }

  out_offsets[idx] = idx;
}

__device__ __forceinline__ uint32_t locate_window(uint32_t idx,
                                                  const uint32_t *prefix,
                                                  uint32_t window_count) {
  // Binary search for better performance with many windows
  uint32_t left = 0;
  uint32_t right = window_count;
  while (left < right) {
    uint32_t mid = (left + right) / 2;
    if (idx < prefix[mid]) {
      right = mid;
    } else if (idx >= prefix[mid + 1]) {
      left = mid + 1;
    } else {
      return mid;
    }
  }
  return window_count;
}

__device__ __forceinline__ uint32_t build_tail_mask(uint32_t valid_bits) {
  if (valid_bits >= 32u)
    return 0xFFFFFFFFu;
  if (valid_bits == 0u)
    return 0u;
  return (1u << valid_bits) - 1u;
}

__device__ __forceinline__ uint32_t parity_mask_from_base(uint32_t base_parity) {
  return (base_parity & 1u) ? 0x55555555u : 0xAAAAAAAAu;
}

__global__ void sievePrototypeScanKernel(const uint32_t *bitmap_words,
                                         const PrototypeWindowDesc *descs,
                                         uint32_t window_count,
                                         uint32_t *out_offsets,
                                         uint32_t *window_counts) {
  const uint32_t window_idx = blockIdx.x;
  if (window_idx >= window_count)
    return;

  const PrototypeWindowDesc desc = descs[window_idx];
  if (desc.word_count == 0 || desc.bit_length == 0)
    return;

  const uint32_t parity_mask = parity_mask_from_base(desc.base_parity);

  for (uint32_t word_rel = threadIdx.x; word_rel < desc.word_count;
       word_rel += blockDim.x) {
    const uint32_t local_bit_base = word_rel * 32u;
    if (local_bit_base >= desc.bit_length)
      continue;

    const uint32_t word = bitmap_words[desc.word_start + word_rel];
    const uint32_t remaining_bits = desc.bit_length - local_bit_base;
    const uint32_t valid_bits = (remaining_bits >= 32u) ? 32u : remaining_bits;

    uint32_t candidate_mask = ~word & parity_mask;
    candidate_mask &= build_tail_mask(valid_bits);
    const uint32_t bit_count = __popc(candidate_mask);
    if (bit_count == 0u)
      continue;

    const uint32_t base = atomicAdd(window_counts + window_idx, bit_count);
    if (base >= desc.bit_length)
      continue;
    const uint32_t max_write = min(bit_count, desc.bit_length - base);

    uint32_t written = 0u;
    while (candidate_mask && written < max_write) {
      const uint32_t bit = __ffs(candidate_mask) - 1u;
      candidate_mask &= (candidate_mask - 1u);
      const uint32_t local_bit = local_bit_base + bit;
      out_offsets[desc.output_base + base + written] = local_bit;
      ++written;
    }
  }
}

__global__ void sievePrototypeKernelBatch(const uint32_t *bitmap,
                                          const uint32_t *window_prefix,
                                          const uint64_t *window_bases,
                                          uint32_t window_count,
                                          uint32_t total_bits,
                                          uint32_t *out_offsets) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_bits)
    return;

  const uint32_t word = bitmap[idx >> 5];
  const uint32_t bit = (word >> (idx & 31)) & 1u;
  const uint32_t sentinel = 0xFFFFFFFFu;
  if (bit) {
    out_offsets[idx] = sentinel;
    return;
  }

  const uint32_t window = locate_window(idx, window_prefix, window_count);
  if (window >= window_count) {
    out_offsets[idx] = sentinel;
    return;
  }

  const uint32_t local_idx = idx - window_prefix[window];
  const uint64_t absolute = window_bases[window] + static_cast<uint64_t>(local_idx);
  if ((absolute & 1ull) == 0ull) {
    out_offsets[idx] = sentinel;
    return;
  }

  out_offsets[idx] = local_idx;
}

__global__ void sieveBuildBitmapKernel(const uint32_t *primes,
                                       const uint32_t *prime_residues,
                                       uint32_t prime_count,
                                       uint32_t window_size,
                                       uint32_t *bitmap_words) {
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < prime_count;
       idx += stride) {
    const uint32_t prime = primes[idx];
    if (prime <= 2u)
      continue;
    const uint32_t step = prime << 1;
    if (step == 0u)
      continue;
    uint32_t offset = prime_residues[idx];
    while (offset < window_size) {
      atomicOr(bitmap_words + (offset >> 5), 1u << (offset & 31));
      offset += step;
    }
  }
}

__global__ void sieveBuildBitmapKernelBatch(const uint32_t *primes,
                                            const uint32_t *prime_residues,
                                            const PrototypeWindowDesc *descs,
                                            uint32_t window_count,
                                            uint32_t *bitmap_words) {
  const uint32_t window_idx = blockIdx.y;
  if (window_idx >= window_count)
    return;

  const PrototypeWindowDesc desc = descs[window_idx];
  if (desc.word_count == 0 || desc.bit_length == 0 || desc.residue_count == 0)
    return;

  const uint32_t stride = blockDim.x * gridDim.x;
  const uint32_t window_size = desc.bit_length;
  uint32_t *window_bitmap = bitmap_words + desc.word_start;

  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < desc.residue_count;
       idx += stride) {
    const uint32_t prime = primes[idx];
    if (prime <= 2u)
      continue;
    const uint32_t step = prime << 1;
    if (step == 0u)
      continue;

    const uint32_t residue = prime_residues[desc.residue_offset + idx];
    uint32_t offset = residue;
    while (offset < window_size) {
      atomicOr(window_bitmap + (offset >> 5), 1u << (offset & 31));
      offset += step;
    }
  }
}

/**
 * sievePrototypeMarkCompositesOptimized: GPU Sieve Marking Kernel (Per-Word Strategy)
 * 
 * Strategy: Each thread marks all composites in one 32-bit word across ALL primes.
 * This provides excellent memory coalescing and vectorization.
 * 
 * Algorithm: For each 32-bit word, check all bits to see if they are:
 * 1. At odd positions (base_offset + bit is odd)
 * 2. Composites (multiples of any small prime)
 * 
 * Parameters:
 *   bitmap: Output bitmap to fill with composite markers (1 = composite, 0 = prime candidate)
 *   bitmap_word_count: Total number of 32-bit words in bitmap
 *   base_offset: Absolute starting position of sieve window (must be odd)
 *   prime_residues: [const memory] Array of residue[i] = starting offset of prime[i] in window
 *   prime_count: Number of small primes to sieve with
 *   shift: Gap mining shift parameter (for absolute position calculation)
 * 
 * Thread mapping: Each thread idx processes bitmap word idx, marking all odd composites
 * in bits [32*idx, 32*idx+32).
 */
__global__ void sievePrototypeMarkCompositesOptimized(
    uint32_t *bitmap,
    uint32_t bitmap_word_count,
    uint64_t base_offset,
    const uint32_t *prime_residues,
    uint32_t prime_count,
    uint32_t shift)
{
  (void)shift;
  // Grid-stride loop: handle all words in the bitmap
  for (uint32_t word_idx = blockIdx.x * blockDim.x + threadIdx.x;
       word_idx < bitmap_word_count;
       word_idx += blockDim.x * gridDim.x) {
    
    uint32_t result = 0;  // Accumulate composite bits for this word
    uint32_t word_bits_base = word_idx << 5;  // word_idx * 32
    uint32_t parity_base = (base_offset >> 0) & 1u;  // Whether base_offset is odd
    
    // Check each of the 32 bits in this word for composites
    for (uint32_t bit_in_word = 0; bit_in_word < 32; ++bit_in_word) {
      uint32_t bit = word_bits_base + bit_in_word;
      
      // Check if absolute position (base_offset + bit) is odd
      // absolute is odd iff (base_offset XOR bit) is odd
      uint32_t absolute_parity = parity_base ^ (bit & 1u);
      if (absolute_parity == 0u)
        continue;  // Even position, skip (we only care about odd)
      
      // Check if this bit is a composite (multiple of some prime)
      bool is_composite = false;
      for (uint32_t p_idx = 0; p_idx < prime_count; ++p_idx) {
        uint32_t prime = kSmallPrimesDevice[p_idx];
        uint32_t residue = prime_residues[p_idx];
        
        // Is (bit - residue) divisible by prime?
        if (bit >= residue) {
          uint32_t delta = bit - residue;
          if (delta % prime == 0u) {
            is_composite = true;
            break;
          }
        }
      }
      
      if (is_composite) {
        result |= (1u << bit_in_word);
      }
    }
    
    // Single coalesced write per thread
    bitmap[word_idx] = result;
  }
}

/**
 * sievePrototypeMarkCompositesPerPrime: GPU Sieve Marking Kernel (Per-Prime Strategy)
 * 
 * Alternative strategy: Each thread handles all multiples of one prime.
 * Uses atomicOr for thread-safe bit marking. Simpler but with more contention.
 * Good for comparison and smaller prime counts.
 */
__global__ void sievePrototypeMarkCompositesPerPrime(
    uint32_t *bitmap,
    uint32_t window_size,      // Size in bits
    uint64_t base_offset,
    const uint32_t *prime_residues,
    uint32_t prime_count)
{
  // Grid-stride loop: each thread handles one or more primes
  for (uint32_t p_idx = blockIdx.x * blockDim.x + threadIdx.x;
       p_idx < prime_count;
       p_idx += blockDim.x * gridDim.x) {
    
    uint32_t prime = kSmallPrimesDevice[p_idx];
    uint32_t residue = prime_residues[p_idx];
    
    if (prime <= 2u)
      continue;
    
    // Mark all multiples: residue, residue + 2*prime, residue + 4*prime, ...
    // (double-stepping ensures we only mark odd positions)
    for (uint32_t offset = residue; offset < window_size; offset += (prime << 1)) {
      uint32_t word_idx = (offset >> 5);
      uint32_t bit_idx = offset & 0x1F;
      uint32_t bit_mask = 1u << bit_idx;
      
      // Verify absolute position is odd
      uint64_t absolute = base_offset + offset;
      if ((absolute & 1ull) == 1ull) {
        // Thread-safe bit setting with atomicOr
        atomicOr(&bitmap[word_idx], bit_mask);
      }
    }
  }
}

__global__ __launch_bounds__(kCudaBlockSize, 2)
void fermatTest320Kernel(const uint32_t *__restrict__ numbers,
                         ResultWord *__restrict__ results,
                         uint32_t elementsNum) {
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < elementsNum;
       idx += stride) {
    const uint32_t offset = numbers[idx];

    if (quick_composite(offset)) {
      results[idx] = static_cast<ResultWord>(0u);
      continue;
    }

    BigInt candidate;
    uint32_t final_carry = build_candidate(candidate, offset);
    if (final_carry != 0u) {
      // Candidate extended beyond configured operand size; treat as composite for now.
      results[idx] = static_cast<ResultWord>(0u);
      continue;
    }
    bool probable = fermat_probable_prime(candidate);
    results[idx] = static_cast<ResultWord>(probable ? 1u : 0u);
  }
}

__global__ void montgomeryMulClassicKernel(const BigInt *a,
                                           const BigInt *b,
                                           const BigInt *mod,
                                           const uint32_t *nPrime,
                                           BigInt *out,
                                           uint32_t count,
                                           uint32_t rounds) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  BigInt acc;
  big_copy(acc, a[idx]);
  BigInt base;
  big_copy(base, b[idx]);
  BigInt modulus;
  big_copy(modulus, mod[idx]);
  const uint32_t np = nPrime[idx];
  BigInt tmp;

  for (uint32_t r = 0; r < rounds; ++r) {
    montgomery_mul(acc, base, modulus, np, tmp);
    big_copy(acc, tmp);
  }
  big_copy(out[idx], acc);
}

__global__ void montgomeryMulUnrolledKernel(const BigInt *a,
                                            const BigInt *b,
                                            const BigInt *mod,
                                            const uint32_t *nPrime,
                                            BigInt *out,
                                            uint32_t count,
                                            uint32_t rounds) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  BigInt acc;
  big_copy(acc, a[idx]);
  BigInt base;
  big_copy(base, b[idx]);
  BigInt modulus;
  big_copy(modulus, mod[idx]);
  const uint32_t np = nPrime[idx];
  BigInt tmp;

  for (uint32_t r = 0; r < rounds; ++r) {
    montgomery_mul_unrolled(acc, base, modulus, np, tmp);
    big_copy(acc, tmp);
  }
  big_copy(out[idx], acc);
}

__global__ void montgomeryMulCombaKernel(const BigInt *a,
                                         const BigInt *b,
                                         const BigInt *mod,
                                         const uint32_t *nPrime,
                                         BigInt *out,
                                         uint32_t count,
                                         uint32_t rounds) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  BigInt acc;
  big_copy(acc, a[idx]);
  BigInt base;
  big_copy(base, b[idx]);
  BigInt modulus;
  big_copy(modulus, mod[idx]);
  const uint32_t np = nPrime[idx];
  BigInt tmp;

  for (uint32_t r = 0; r < rounds; ++r) {
    montgomery_mul_cios(acc, base, modulus, np, tmp);
    big_copy(acc, tmp);
  }
  big_copy(out[idx], acc);
}

// Diagnostic kernel: compute montgomery multiplication of (one, r2) via
// classic and unrolled paths for given samples and write results back.
__global__ void montgomeryCompareKernel(const uint32_t *numbers,
                                        const uint32_t *sample_indices,
                                        uint32_t *out_classic,
                                        uint32_t *out_unrolled,
                                        unsigned sampleCount) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sampleCount)
    return;

  const uint32_t idx = sample_indices[tid];
  const uint32_t off = numbers[idx];

  BigInt n;
  uint32_t final_carry = build_candidate(n, off);
  if (final_carry != 0u) {
    // mark with zeroes to indicate overflow
    for (int i = 0; i < kOperandSize; ++i) {
      out_classic[tid * kOperandSize + i] = 0u;
      out_unrolled[tid * kOperandSize + i] = 0u;
    }
    return;
  }

  uint32_t nPrime = montgomery_inverse32(n.limb[0]);
  BigInt r2;
  compute_r2(n, r2);

  BigInt one;
  big_set_zero(one);
  one.limb[0] = 1u;

  BigInt outC;
  montgomery_mul(one, r2, n, nPrime, outC);
  BigInt outU;
  montgomery_mul_unrolled(one, r2, n, nPrime, outU);

  for (int i = 0; i < kOperandSize; ++i) {
    out_classic[tid * kOperandSize + i] = outC.limb[i];
    out_unrolled[tid * kOperandSize + i] = outU.limb[i];
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
    : Size(0),
      ElementSize(0),
      HostData(nullptr),
      DeviceData(nullptr),
      storage_(),
      pinned_host_(false) {}

GPUFermat::CudaBuffer::~CudaBuffer() { reset(); }

void GPUFermat::CudaBuffer::reset() {
  if (DeviceData) {
    cudaFree(DeviceData);
    DeviceData = nullptr;
  }
  if (HostData && pinned_host_)
    cudaFreeHost(HostData);
  HostData = nullptr;
  Size = 0;
  ElementSize = 0;
  pinned_host_ = false;
  storage_.clear();
}

void GPUFermat::CudaBuffer::init(size_t size,
                                 size_t element_size,
                                 bool prefer_pinned) {
  reset();
  Size = size;
  ElementSize = element_size;
  const size_t total_bytes = Size * ElementSize;
  if (total_bytes == 0)
    return;

  if (prefer_pinned) {
    cudaError_t host_err = cudaHostAlloc(reinterpret_cast<void **>(&HostData),
                                         total_bytes,
                                         cudaHostAllocPortable);
    if (host_err == cudaSuccess) {
      pinned_host_ = true;
      memset(HostData, 0, total_bytes);
    } else {
      HostData = nullptr;
      pinned_host_ = false;
      if (host_err != cudaErrorMemoryAllocation &&
          host_err != cudaErrorNotSupported) {
        CUDA_CHECK(host_err);
      }
    }
  }

  if (!HostData) {
    storage_.assign(total_bytes, 0u);
    HostData = storage_.data();
  }

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&DeviceData), total_bytes));
  CUDA_CHECK(cudaMemset(DeviceData, 0, total_bytes));
}

void GPUFermat::CudaBuffer::copyToDevice(cudaStream_t stream,
                                         size_t count,
                                         size_t offset) {
  if (!DeviceData || !HostData)
    return;
  const size_t elems = (count == 0) ? (Size - offset) : count;
  if (elems == 0)
    return;
  const size_t byte_offset = offset * ElementSize;
  const size_t bytes = elems * ElementSize;
  CUDA_CHECK(cudaMemcpyAsync(DeviceData + byte_offset,
                             HostData + byte_offset,
                             bytes,
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
  const size_t byte_offset = offset * ElementSize;
  const size_t bytes = elems * ElementSize;
  CUDA_CHECK(cudaMemcpyAsync(HostData + byte_offset,
                             DeviceData + byte_offset,
                             bytes,
                             cudaMemcpyDeviceToHost,
                             stream));
}

GPUFermat::GPUFermat(unsigned device_id,
                     const char *platformId,
                     unsigned workItems,
                     unsigned streamCount)
    : workItems(workItems),
      elementsNum(0),
      numberLimbsNum(0),
      groupsNum(0),
      streams(),
      computeUnits(0),
      smallPrimeResidues{},
      prototypeOffsets(),
      prototypeWindowOffsets(),
      lastPrototypeCount(0),
      lastPrototypeWindowCount(0),
      primeReciprocalsUploaded(false),
      primeBaseConstantsUploaded(false),
      primeBaseFingerprint(0ull),
      sievePrimesConfigured(false),
      configuredPrimeCount(0) {
  log_str("Creating CUDA GPUFermat", LOG_D);
  if (workItems == 0)
    throw std::runtime_error("workItems must be greater than zero");
  // prepare requested number of streams before CUDA init so init_cuda
  // can create them
  if (streamCount == 0)
    streamCount = 1;
  streams.resize(static_cast<size_t>(streamCount));

  if (!init_cuda(device_id, platformId))
    throw std::runtime_error("Failed to initialize CUDA backend");

  elementsNum = GroupSize * workItems;
  numberLimbsNum = elementsNum * operandSize;
  groupsNum = std::max(1u, computeUnits * 8u);

  initializeBuffers();
}

GPUFermat::~GPUFermat() {
  for (cudaEvent_t &evt : h2d_events) {
    if (evt) {
      cudaEventDestroy(evt);
      evt = nullptr;
    }
  }
  for (cudaEvent_t &evt : kernel_events) {
    if (evt) {
      cudaEventDestroy(evt);
      evt = nullptr;
    }
  }
  h2d_events.clear();
  kernel_events.clear();
  for (cudaStream_t &s : streams) {
    if (s) {
      cudaStreamDestroy(s);
      s = nullptr;
    }
  }
  numbers.reset();
  gpuResults.reset();
  primeBase.reset();
  sievePrototypeOutput.reset();
  sievePrototypeBitmap.reset();
  sievePrototypePrimes.reset();
  sievePrototypeResidues.reset();
  sievePrototypeWindowBits.reset();
  sievePrototypeWindowBases.reset();
  sievePrototypeWindowCounts.reset();
  sievePrototypeWindowDescs.reset();
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
  const uint32_t use_comba =
      (Opts::get_instance() && Opts::get_instance()->use_cuda_comba()) ? 1u : 0u;
  CUDA_CHECK(cudaMemcpyToSymbol(kUseCombaConst,
                                &use_comba,
                                sizeof(use_comba),
                                0,
                                cudaMemcpyHostToDevice));
  for (cudaStream_t &s : streams)
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  h2d_events.resize(streams.size());
  kernel_events.resize(streams.size());
  for (size_t i = 0; i < streams.size(); ++i) {
    CUDA_CHECK(cudaEventCreateWithFlags(&h2d_events[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&kernel_events[i], cudaEventDisableTiming));
  }

  if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
    std::ostringstream ss;
    ss << get_time() << " CUDA streams count=" << streams.size() << " handles:";
    for (size_t i = 0; i < streams.size(); ++i)
      ss << " [" << i << "]=" << streams[i];
    extra_verbose_log(ss.str());
  }

  pthread_mutex_lock(&io_mutex);
  cout << get_time() << "Using CUDA GPU " << device_id << " [" << props.name
       << "] with " << computeUnits << " SMs" << endl;
  pthread_mutex_unlock(&io_mutex);
  return true;
}

void GPUFermat::initializeBuffers() {
  numbers.init(elementsNum, sizeof(uint32_t), true);
  gpuResults.init(elementsNum, sizeof(ResultWord), true);
  primeBase.init(operandSize, sizeof(uint32_t), true);
}

void GPUFermat::uploadPrimeBaseConstants() {
  if (!primeBase.HostData)
    return;
  const uint32_t *prime_base_words =
      buffer_cast<const uint32_t>(primeBase.HostData);

  /* fingerprint prime base to avoid redundant constant uploads */
  uint64_t fp = 1469598103934665603ull; // FNV offset basis
  for (unsigned i = 0; i < operandSize; ++i) {
    fp ^= static_cast<uint64_t>(prime_base_words[i]);
    fp *= 1099511628211ull; // FNV prime
  }
  if (primeBaseConstantsUploaded && fp == primeBaseFingerprint)
    return;

  // (extra-verbose host buffer print removed)

  CUDA_CHECK(cudaMemcpyToSymbol(kPrimeBaseConst,
                                prime_base_words,
                                operandSize * sizeof(uint32_t),
                                0,
                                cudaMemcpyHostToDevice));

  // Residues depend on the current prime base, so refresh every batch.
  smallPrimeResidues.assign(kSmallPrimeCount, 0u);
  for (int i = 0; i < kSmallPrimeCount; ++i)
    smallPrimeResidues[i] =
        mod_high_part_host(prime_base_words, kSmallPrimesHost[i]);

  CUDA_CHECK(cudaMemcpyToSymbol(kHighResiduesConst,
                                smallPrimeResidues.data(),
                                kSmallPrimeCount * sizeof(uint32_t),
                                0,
                                cudaMemcpyHostToDevice));

  // Reciprocals depend only on the fixed small-prime table; upload once.
  if (!primeReciprocalsUploaded) {
    std::vector<uint32_t> prime_reciprocals(kSmallPrimeCount);
    for (int i = 0; i < kSmallPrimeCount; ++i) {
      const uint32_t p = kSmallPrimesHost[i];
      prime_reciprocals[i] = static_cast<uint32_t>(((uint64_t(1) << 32) / p) + 1ull);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(kPrimeReciprocalsDevice,
                                  prime_reciprocals.data(),
                                  kSmallPrimeCount * sizeof(uint32_t),
                                  0,
                                  cudaMemcpyHostToDevice));
    primeReciprocalsUploaded = true;
  }

  primeBaseConstantsUploaded = true;
  primeBaseFingerprint = fp;
}

void GPUFermat::ensurePrototypeOutputCapacity(size_t required) {
  if (required == 0)
    return;
  if (sievePrototypeOutput.Size >= required)
    return;
  sievePrototypeOutput.init(required, sizeof(uint32_t), true);
}

void GPUFermat::ensurePrototypeBitmapCapacity(size_t required_words) {
  if (required_words == 0)
    return;
  if (sievePrototypeBitmap.Size >= required_words)
    return;
  sievePrototypeBitmap.init(required_words, sizeof(uint32_t), true);
}

void GPUFermat::ensurePrototypeResidueCapacity(size_t required) {
  if (required == 0)
    return;
  if (sievePrototypeResidues.Size >= required)
    return;
  sievePrototypeResidues.init(required, sizeof(uint32_t), true);
}

void GPUFermat::ensurePrototypeWindowBitCapacity(size_t required) {
  if (required == 0)
    return;
  if (sievePrototypeWindowBits.Size >= required)
    return;
  sievePrototypeWindowBits.init(required, sizeof(uint32_t), true);
}

void GPUFermat::ensurePrototypeBaseCapacity(size_t required) {
  if (required == 0)
    return;
  if (sievePrototypeWindowBases.Size >= required)
    return;
  sievePrototypeWindowBases.init(required, sizeof(uint64_t), true);
}

void GPUFermat::ensurePrototypeWindowCountCapacity(size_t required) {
  if (required == 0)
    return;
  if (sievePrototypeWindowCounts.Size >= required)
    return;
  sievePrototypeWindowCounts.init(required, sizeof(uint32_t), true);
}

void GPUFermat::ensurePrototypeWindowDescCapacity(size_t required) {
  if (required == 0)
    return;
  if (sievePrototypeWindowDescs.Size >= required)
    return;
  sievePrototypeWindowDescs.init(required,
                                 sizeof(PrototypeWindowDesc),
                                 true);
}

void GPUFermat::resetPrototypeWindowState() {
  prototypeWindowOffsets.clear();
  lastPrototypeWindowCount = 0;
}

bool GPUFermat::run_compact_scan(uint32_t window_count,
                                 uint32_t total_bits,
                                 cudaStream_t stream) {
  if (window_count == 0 || total_bits == 0)
    return false;
  if (!sievePrototypeBitmap.DeviceData || !sievePrototypeOutput.DeviceData)
    return false;

  ensurePrototypeWindowCountCapacity(window_count);
  if (!sievePrototypeWindowCounts.DeviceData ||
      !sievePrototypeWindowCounts.HostData)
    return false;
  if (!sievePrototypeWindowDescs.DeviceData ||
      !sievePrototypeWindowDescs.HostData)
    return false;

  const size_t count_bytes = static_cast<size_t>(window_count) * sizeof(uint32_t);
  CUDA_CHECK(cudaMemsetAsync(sievePrototypeWindowCounts.DeviceData,
                             0,
                             count_bytes,
                             stream));

  sievePrototypeWindowDescs.copyToDevice(stream, window_count);

  const uint32_t threads = std::max(1u, std::min(256u, GroupSize));
  sievePrototypeScanKernel<<<window_count, threads, 0, stream>>>(
      buffer_cast<const uint32_t>(sievePrototypeBitmap.DeviceData),
      buffer_cast<const PrototypeWindowDesc>(sievePrototypeWindowDescs.DeviceData),
      window_count,
      buffer_cast<uint32_t>(sievePrototypeOutput.DeviceData),
      buffer_cast<uint32_t>(sievePrototypeWindowCounts.DeviceData));
  CUDA_CHECK(cudaGetLastError());

  sievePrototypeWindowCounts.copyToHost(stream, window_count);
  sievePrototypeOutput.copyToHost(stream, total_bits);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const PrototypeWindowDesc *desc_host =
      buffer_cast<const PrototypeWindowDesc>(sievePrototypeWindowDescs.HostData);
  const uint32_t *count_host =
      buffer_cast<const uint32_t>(sievePrototypeWindowCounts.HostData);
  const uint32_t *output_host =
      buffer_cast<const uint32_t>(sievePrototypeOutput.HostData);
  if (!desc_host || !count_host || !output_host)
    return false;

  uint64_t candidate_total = 0;
  for (uint32_t i = 0; i < window_count; ++i)
    candidate_total += count_host[i];

  prototypeOffsets.clear();
  prototypeOffsets.reserve(static_cast<size_t>(candidate_total));
  prototypeWindowOffsets.clear();
  prototypeWindowOffsets.reserve(window_count + 1);
  prototypeWindowOffsets.push_back(0u);

  uint32_t running_total = 0;
  for (uint32_t i = 0; i < window_count; ++i) {
    const uint32_t count = count_host[i];
    const uint32_t base_index = desc_host[i].output_base;
    for (uint32_t j = 0; j < count; ++j)
      prototypeOffsets.push_back(output_host[base_index + j]);
    running_total += count;
    prototypeWindowOffsets.push_back(running_total);
  }

  lastPrototypeCount = running_total;
  lastPrototypeWindowCount = window_count;
  return true;
}

GPUFermat *GPUFermat::get_instance(unsigned device_id,
                                   const char *platformId,
                                   unsigned workItems,
                                   unsigned streamCount) {
  pthread_mutex_lock(&creation_mutex);
  if (!initialized && device_id != static_cast<unsigned>(-1) &&
      platformId != nullptr && workItems != 0) {
    only_instance = new GPUFermat(device_id, platformId, workItems, streamCount);
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

GPUFermat::ResultWord *GPUFermat::get_results_buffer() {
  return gpuResults.HostData
             ? buffer_cast<ResultWord>(gpuResults.HostData)
             : nullptr;
}

unsigned GPUFermat::get_result_word_size() {
  return static_cast<unsigned>(gpuResults.ElementSize);
}

uint32_t *GPUFermat::get_prime_base_buffer() {
  return primeBase.HostData ? buffer_cast<uint32_t>(primeBase.HostData)
                            : nullptr;
}

uint32_t *GPUFermat::get_candidates_buffer() {
  return numbers.HostData ? buffer_cast<uint32_t>(numbers.HostData) : nullptr;
}

void GPUFermat::configure_sieve_primes(const uint32_t *primes, size_t count) {
  if (primes == nullptr || count == 0)
    return;
  sievePrototypePrimes.init(count, sizeof(uint32_t), true);
  memcpy(sievePrototypePrimes.HostData, primes, count * sizeof(uint32_t));
  cudaStream_t stream = streams[0];
  sievePrototypePrimes.copyToDevice(stream, count);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  sievePrimesConfigured = true;
  configuredPrimeCount = static_cast<uint32_t>(count);
  ensureMarkingResidueCapacity(count);
}

void GPUFermat::run_cuda(uint32_t batchElements) {
  if (batchElements == 0)
    return;
  const uint32_t work = std::min(batchElements, elementsNum);
  log_str("running " + std::to_string(work) + " fermat tests on the gpu", LOG_D);

  /* Upload constants once per batch; they rarely change and can serialize work
   * if sent every chunk. */
  uploadPrimeBaseConstants();

  const uint32_t threads = GroupSize;
    /* Allow deeper launch to improve occupancy on GA106-class GPUs. */
    const uint32_t maxBlocks = std::max(1u, groupsNum * 8u);
  const uint32_t streamCount = static_cast<uint32_t>(streams.size());
    /* Let smaller chunks launch more blocks when work is uneven. */
    const uint32_t chunkSize = std::max<uint32_t>(threads / 2u,
      (work + streamCount - 1u) / streamCount);

  /* Events to chain H2D -> kernel -> D2H while alternating streams to overlap
   * the stages. Events are created once during init and reused here. */

  uint32_t processed = 0;
  uint32_t chunkIndex = 0;
  while (processed < work) {
    const uint32_t chunk = std::min(chunkSize, work - processed);

    /* pipeline: copy on stream A, kernel on stream B after copy, D2H on stream C
     * after kernel. This overlaps N, N+1, N-1 across streams. */
    const uint32_t slot = chunkIndex % streamCount;
    const uint32_t kernel_slot = (slot + 1) % streamCount;
    const uint32_t d2h_slot = (slot + 2) % streamCount;
    cudaStream_t h2dStream = streams[slot];
    cudaStream_t kernelStream = streams[kernel_slot];
    cudaStream_t d2hStream = streams[d2h_slot];

    numbers.copyToDevice(h2dStream, chunk, processed);
    CUDA_CHECK(cudaEventRecord(h2d_events[slot], h2dStream));

    CUDA_CHECK(cudaStreamWaitEvent(kernelStream, h2d_events[slot], 0));

    const uint32_t requiredBlocks = (chunk + threads - 1u) / threads;
    const uint32_t blocks = std::max(1u, std::min(requiredBlocks, maxBlocks));

    const uint32_t *number_device =
      buffer_cast<const uint32_t>(numbers.DeviceData) + processed;
    ResultWord *results_device =
      buffer_cast<ResultWord>(gpuResults.DeviceData) + processed;

    fermatTest320Kernel<<<blocks, threads, 0, kernelStream>>>(
      number_device,
      results_device,
      chunk);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_events[kernel_slot], kernelStream));

    CUDA_CHECK(cudaStreamWaitEvent(d2hStream, kernel_events[kernel_slot], 0));
    gpuResults.copyToHost(d2hStream, chunk, processed);

    processed += chunk;
    ++chunkIndex;
  }

  for (cudaStream_t s : streams)
    CUDA_CHECK(cudaStreamSynchronize(s));

}

void GPUFermat::dump_device_samples(const uint32_t *sample_indices, unsigned sampleCount) {
  if (!sampleCount || !sample_indices || !numbers.DeviceData)
    return;

  const unsigned threads = 128u;
  const unsigned blocks = (sampleCount + threads - 1u) / threads;

  uint32_t *d_sample_indices = nullptr;
  uint32_t *d_out_limbs = nullptr;
  uint32_t *host_out_limbs = nullptr;
  uint32_t *host_out_carry = nullptr;

  CUDA_CHECK(cudaMalloc(&d_sample_indices, sampleCount * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_out_limbs, (sampleCount * kOperandSize + sampleCount) * sizeof(uint32_t)));

  host_out_limbs = (uint32_t *)malloc(sampleCount * kOperandSize * sizeof(uint32_t));
  host_out_carry = (uint32_t *)malloc(sampleCount * sizeof(uint32_t));

  CUDA_CHECK(cudaMemcpy(d_sample_indices, sample_indices, sampleCount * sizeof(uint32_t), cudaMemcpyHostToDevice));

  dump_candidate_limbs_kernel<<<blocks, threads>>>(
      buffer_cast<const uint32_t>(numbers.DeviceData),
      d_sample_indices,
      d_out_limbs,
      sampleCount);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_out_limbs, d_out_limbs, sampleCount * kOperandSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_out_carry, d_out_limbs + (sampleCount * kOperandSize), sampleCount * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
    std::ostringstream ss;
    ss << get_time() << " CUDA device-dump:" << std::hex;
    for (unsigned s = 0; s < sampleCount; ++s) {
      ss << "\n" << get_time() << " DEVICE sample " << s << " idx=" << std::dec << sample_indices[s] << std::hex << " limbs:";
      for (int l = 0; l < kOperandSize; ++l)
        ss << " " << std::setw(8) << std::setfill('0') << host_out_limbs[s * kOperandSize + l];
      ss << " carry=" << std::setw(8) << host_out_carry[s];
    }
    ss << std::dec;
    extra_verbose_log(ss.str());
  }

  free(host_out_limbs);
  free(host_out_carry);
  CUDA_CHECK(cudaFree(d_sample_indices));
  CUDA_CHECK(cudaFree(d_out_limbs));
}

void GPUFermat::fermat_gpu(uint32_t elements) {
  const uint32_t work = (elements == 0) ? elementsNum : elements;
  run_cuda(work);
}

unsigned GPUFermat::get_elements_num() {
  return elementsNum;
}

void GPUFermat::prototype_sieve(const SievePrototypeParams &params) {
  prototype_sieve_batch(&params, 1u);
}

void GPUFermat::prototype_sieve_batch(const SievePrototypeParams *params,
                                      uint32_t window_count) {
  lastPrototypeCount = 0;
  prototypeOffsets.clear();
  resetPrototypeWindowState();

  if (!params || window_count == 0)
    return;

  const bool proto_enabled = Opts::get_instance()->use_cuda_sieve_proto();
  const bool extra_verbose = Opts::get_instance()->has_extra_vb();
    // (removed verbose proto options log)

  struct WindowAttributes {
    bool has_bitmap = false;
    bool has_residue = false;
  };

  std::vector<WindowAttributes> window_attrs(window_count);
  std::vector<uint32_t> residue_offsets(window_count, 0u);
  std::vector<uint32_t> residue_counts(window_count, 0u);
  std::vector<uint32_t> residue_buffer;
  residue_buffer.reserve(window_count * 16u);

  bool requires_single = false;
  for (uint32_t idx = 0; idx < window_count; ++idx) {
    if (params[idx].window_size == 0)
      continue;
    const bool has_host_bitmap =
        (params[idx].sieve_bytes != nullptr && params[idx].sieve_byte_len != 0);
    const bool has_residue_snapshot =
        (params[idx].prime_starts != nullptr && params[idx].prime_count != 0);

    if (!has_host_bitmap && !has_residue_snapshot) {
      requires_single = true;
      break;
    }

    if (has_residue_snapshot &&
        (!sievePrimesConfigured || params[idx].prime_count > configuredPrimeCount)) {
      if (extra_verbose) {
        std::ostringstream ss;
        ss << get_time()
           << "CUDA sieve prototype missing prime configuration for snapshot";
        extra_verbose_log(ss.str());
      }
      requires_single = true;
      break;
    }

    window_attrs[idx].has_bitmap = has_host_bitmap;
    window_attrs[idx].has_residue = has_residue_snapshot;
    if (has_residue_snapshot && params[idx].prime_count > 0) {
      residue_offsets[idx] = static_cast<uint32_t>(residue_buffer.size());
      residue_counts[idx] = params[idx].prime_count;
      residue_buffer.insert(residue_buffer.end(),
                            params[idx].prime_starts,
                            params[idx].prime_starts + params[idx].prime_count);
    }
  }

  if (requires_single) {
    std::vector<uint32_t> aggregate_offsets;
    std::vector<uint32_t> aggregate_slices;
    aggregate_slices.reserve(window_count + 1);
    aggregate_slices.push_back(0u);

    for (uint32_t idx = 0; idx < window_count; ++idx) {
      prototype_sieve_single(params[idx]);
      aggregate_offsets.insert(aggregate_offsets.end(),
                               prototypeOffsets.begin(),
                               prototypeOffsets.end());
      aggregate_slices.push_back(
          static_cast<uint32_t>(aggregate_offsets.size()));
    }

    prototypeOffsets.swap(aggregate_offsets);
    prototypeWindowOffsets.swap(aggregate_slices);
    lastPrototypeCount = static_cast<uint32_t>(prototypeOffsets.size());
    lastPrototypeWindowCount = window_count;
    return;
  }

  uint64_t total_bits = 0;
  uint64_t total_words = 0;
  for (uint32_t idx = 0; idx < window_count; ++idx) {
    total_bits += params[idx].window_size;
    total_words +=
        (static_cast<uint64_t>(params[idx].window_size) + 31ull) >> 5;
  }
  if (total_bits == 0 || total_words == 0)
    return;

  ensurePrototypeOutputCapacity(static_cast<size_t>(total_bits));
  ensurePrototypeBitmapCapacity(static_cast<size_t>(total_words));
  ensurePrototypeWindowBitCapacity(static_cast<size_t>(window_count + 1));
  ensurePrototypeBaseCapacity(window_count);
    ensurePrototypeWindowDescCapacity(window_count);

  uint8_t *bitmap_host_bytes = sievePrototypeBitmap.HostData;
  uint32_t *bit_prefix =
      buffer_cast<uint32_t>(sievePrototypeWindowBits.HostData);
  uint64_t *base_host =
      buffer_cast<uint64_t>(sievePrototypeWindowBases.HostData);
    std::vector<uint32_t> word_prefix(window_count + 1u, 0u);
    std::vector<uint32_t> window_bit_lengths(window_count, 0u);

  size_t byte_cursor = 0;
  uint64_t bit_cursor = 0;
  bool any_host_bitmap = false;
  bit_prefix[0] = 0u;
  for (uint32_t idx = 0; idx < window_count; ++idx) {
    const uint32_t window_bits = params[idx].window_size;
    const uint32_t window_words =
        static_cast<uint32_t>((static_cast<uint64_t>(window_bits) + 31ull) >> 5);
    const size_t window_bytes = static_cast<size_t>(window_words) * sizeof(uint32_t);
    uint8_t *target = bitmap_host_bytes + byte_cursor;
    if (window_bytes)
      memset(target, 0, window_bytes);
    const size_t copy_bytes = std::min(window_bytes,
                                       static_cast<size_t>(params[idx].sieve_byte_len));
    if (copy_bytes > 0 && params[idx].sieve_bytes) {
      memcpy(target, params[idx].sieve_bytes, copy_bytes);
      any_host_bitmap = true;
    }
    byte_cursor += window_bytes;
    bit_cursor += window_bits;
    bit_prefix[idx + 1] = static_cast<uint32_t>(bit_cursor);
    const uint64_t round_offset =
      static_cast<uint64_t>(params[idx].window_size) * params[idx].sieve_round;
    const uint64_t absolute_base = params[idx].sieve_base + round_offset;
    base_host[idx] = absolute_base;
    window_bit_lengths[idx] = window_bits;
    word_prefix[idx + 1] = word_prefix[idx] + window_words;
  }

  cudaStream_t stream = streams[0];
  if (any_host_bitmap) {
    sievePrototypeBitmap.copyToDevice(stream, static_cast<size_t>(total_words));
  } else {
    const size_t bitmap_bytes = static_cast<size_t>(total_words) * sizeof(uint32_t);
    CUDA_CHECK(cudaMemsetAsync(sievePrototypeBitmap.DeviceData, 0, bitmap_bytes, stream));
  }
  sievePrototypeWindowBits.copyToDevice(stream, window_count + 1);
  sievePrototypeWindowBases.copyToDevice(stream, window_count);

  if (!residue_buffer.empty()) {
    ensurePrototypeResidueCapacity(residue_buffer.size());
    memcpy(buffer_cast<uint32_t>(sievePrototypeResidues.HostData),
           residue_buffer.data(),
           residue_buffer.size() * sizeof(uint32_t));
    sievePrototypeResidues.copyToDevice(stream, residue_buffer.size());
  }

  PrototypeWindowDesc *desc_host =
      buffer_cast<PrototypeWindowDesc>(sievePrototypeWindowDescs.HostData);
  for (uint32_t idx = 0; idx < window_count; ++idx) {
    desc_host[idx].word_start = word_prefix[idx];
    desc_host[idx].word_count = word_prefix[idx + 1] - word_prefix[idx];
    desc_host[idx].bit_length = window_bit_lengths[idx];
    desc_host[idx].output_base = bit_prefix[idx];
    desc_host[idx].base_parity =
        static_cast<uint32_t>(base_host[idx] & 1ull);
    desc_host[idx].residue_offset = residue_offsets[idx];
    desc_host[idx].residue_count = residue_counts[idx];
  }

  sievePrototypeWindowDescs.copyToDevice(stream, window_count);

  if (!residue_buffer.empty()) {
    const uint32_t threads = GroupSize;
    uint32_t max_residue = 0u;
    for (uint32_t idx = 0; idx < window_count; ++idx)
      max_residue = std::max(max_residue, residue_counts[idx]);
    if (max_residue > 0u) {
      const uint32_t blocks_x =
          std::max(1u, (max_residue + threads - 1u) / threads);
      dim3 blocks(blocks_x, window_count, 1u);
      sieveBuildBitmapKernelBatch<<<blocks, threads, 0, stream>>>(
          buffer_cast<const uint32_t>(sievePrototypePrimes.DeviceData),
          buffer_cast<const uint32_t>(sievePrototypeResidues.DeviceData),
          buffer_cast<const PrototypeWindowDesc>(sievePrototypeWindowDescs.DeviceData),
          window_count,
          buffer_cast<uint32_t>(sievePrototypeBitmap.DeviceData));
      CUDA_CHECK(cudaGetLastError());
    }
  }

  auto cpu_window_count = [&](uint32_t idx) -> uint32_t {
    if (idx >= window_count)
      return 0u;
    if (!window_attrs[idx].has_bitmap)
      return std::numeric_limits<uint32_t>::max();
    const uint32_t word_start = word_prefix[idx];
    const uint32_t word_count = word_prefix[idx + 1] - word_prefix[idx];
    const uint32_t bit_length = window_bit_lengths[idx];
    if (word_count == 0 || bit_length == 0)
      return 0u;
    const uint32_t parity_mask = (base_host[idx] & 1ull) ? 0x55555555u : 0xAAAAAAAAu;
    const uint32_t *bitmap_words =
        reinterpret_cast<const uint32_t *>(bitmap_host_bytes);
    uint32_t total = 0u;
    auto host_tail_mask = [](uint32_t bits) -> uint32_t {
      if (bits >= 32u)
        return 0xFFFFFFFFu;
      if (bits == 0u)
        return 0u;
      return (1u << bits) - 1u;
    };

    for (uint32_t w = 0; w < word_count; ++w) {
      const uint32_t local_bit_base = w * 32u;
      if (local_bit_base >= bit_length)
        break;
      const uint32_t remaining_bits = bit_length - local_bit_base;
      const uint32_t valid_bits = (remaining_bits >= 32u) ? 32u : remaining_bits;
      uint32_t candidate_mask = ~bitmap_words[word_start + w] & parity_mask;
      candidate_mask &= host_tail_mask(valid_bits);
      total += static_cast<uint32_t>(__builtin_popcount(candidate_mask));
    }
    return total;
  };

  const bool compact_scan_enabled = Opts::get_instance()->use_cuda_sieve_proto();
  bool compact_scan_used = false;
  const bool residue_only = !any_host_bitmap && !residue_buffer.empty();
  if (compact_scan_enabled &&
      total_bits <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    if (run_compact_scan(window_count,
                         static_cast<uint32_t>(total_bits),
                         stream)) {
      compact_scan_used = true;
      if (Opts::get_instance()->has_extra_vb()) {
        std::ostringstream ss;
        ss << get_time() << "CUDA sieve prototype batched " << window_count
           << " windows; candidates=" << lastPrototypeCount;
        extra_verbose_log(ss.str());

        std::ostringstream ok;
        ok << get_time() << "GPU sieve offsets produced: yes";
        extra_verbose_log(ok.str());

        std::ostringstream perf;
        perf << get_time() << "CUDA sieve perf: windows=" << window_count
             << " total_bits=" << total_bits
             << " total_words=" << total_words
             << " residue_words=" << residue_buffer.size()
             << " compact_scan=yes";
        extra_verbose_log(perf.str());

        if (window_count > 0 && prototypeWindowOffsets.size() > 1) {
          const uint32_t cpu_count = cpu_window_count(0);
          const uint32_t gpu_count = prototypeWindowOffsets[1] - prototypeWindowOffsets[0];
          std::ostringstream chk;
          if (cpu_count == std::numeric_limits<uint32_t>::max()) {
            chk << get_time() << "CUDA sieve cpu-check: window=0 cpu=na(residue)"
                << " gpu=" << gpu_count;
          } else {
            chk << get_time() << "CUDA sieve cpu-check: window=0 cpu=" << cpu_count
                << " gpu=" << gpu_count
                << " delta=" << static_cast<int64_t>(cpu_count) - static_cast<int64_t>(gpu_count);
          }
          extra_verbose_log(chk.str());
        }
      }
      return;
    }
    if (Opts::get_instance()->has_extra_vb()) {
      std::ostringstream ss;
      ss << get_time() << "CUDA sieve prototype compact scan fallback for "
         << window_count << " windows (total_bits=" << total_bits << ")";
      extra_verbose_log(ss.str());
    }
  }

  if (!compact_scan_used && residue_only) {
    if (extra_verbose) {
      std::ostringstream ss;
      ss << get_time()
         << "CUDA sieve prototype residue-only batch skipped legacy scan";
      extra_verbose_log(ss.str());
    }
    return;
  }

  const uint32_t threads = GroupSize;
  const uint32_t blocks =
      std::max(1u, (static_cast<uint32_t>(total_bits) + threads - 1u) / threads);
  sievePrototypeKernelBatch<<<blocks, threads, 0, stream>>>(
      buffer_cast<const uint32_t>(sievePrototypeBitmap.DeviceData),
      buffer_cast<const uint32_t>(sievePrototypeWindowBits.DeviceData),
      buffer_cast<const uint64_t>(sievePrototypeWindowBases.DeviceData),
      window_count,
      static_cast<uint32_t>(total_bits),
      buffer_cast<uint32_t>(sievePrototypeOutput.DeviceData));
  CUDA_CHECK(cudaGetLastError());

  sievePrototypeOutput.copyToHost(stream, static_cast<size_t>(total_bits));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (Opts::get_instance()->has_extra_vb()) {
    const size_t preview = static_cast<size_t>(std::min<uint64_t>(total_bits, 8ull));
    std::ostringstream ss;
    ss << get_time() << "CUDA sieve prototype raw output:";
    for (size_t i = 0; i < preview; ++i)
      ss << ' ' << buffer_cast<const uint32_t>(sievePrototypeOutput.HostData)[i];
    extra_verbose_log(ss.str());
  }

  const uint32_t sentinel = 0xFFFFFFFFu;
  prototypeOffsets.reserve(static_cast<size_t>(total_bits / 64ull + 1ull));
  prototypeWindowOffsets.clear();
  prototypeWindowOffsets.reserve(window_count + 1);
  prototypeWindowOffsets.push_back(0u);

  const uint32_t *output_host =
      buffer_cast<const uint32_t>(sievePrototypeOutput.HostData);
  const uint32_t *bit_prefix_host =
      buffer_cast<const uint32_t>(sievePrototypeWindowBits.HostData);
  for (uint32_t idx = 0; idx < window_count; ++idx) {
    const uint32_t start_bit = bit_prefix_host[idx];
    const uint32_t end_bit = bit_prefix_host[idx + 1];
    for (uint32_t bit = start_bit; bit < end_bit; ++bit) {
      const uint32_t value = output_host[bit];
      if (value != sentinel)
        prototypeOffsets.push_back(value);
    }
    prototypeWindowOffsets.push_back(
        static_cast<uint32_t>(prototypeOffsets.size()));
  }

  lastPrototypeCount = static_cast<uint32_t>(prototypeOffsets.size());
  lastPrototypeWindowCount = window_count;

  if (Opts::get_instance()->has_extra_vb()) {
    std::ostringstream ss;
    ss << get_time() << "CUDA sieve prototype batched " << window_count
       << " windows; candidates=" << lastPrototypeCount;
    extra_verbose_log(ss.str());

    std::ostringstream ok;
    ok << get_time() << "GPU sieve offsets produced: yes";
    extra_verbose_log(ok.str());

    std::ostringstream perf;
    perf << get_time() << "CUDA sieve perf: windows=" << window_count
         << " total_bits=" << total_bits
         << " total_words=" << total_words
         << " residue_words=" << residue_buffer.size()
         << " compact_scan=" << (compact_scan_used ? "yes" : "no");
    extra_verbose_log(perf.str());

    if (window_count > 0 && prototypeWindowOffsets.size() > 1) {
      const uint32_t cpu_count = cpu_window_count(0);
      const uint32_t gpu_count = prototypeWindowOffsets[1] - prototypeWindowOffsets[0];
      std::ostringstream chk;
      if (cpu_count == std::numeric_limits<uint32_t>::max()) {
        chk << get_time() << "CUDA sieve cpu-check: window=0 cpu=na(residue)"
            << " gpu=" << gpu_count;
      } else {
        chk << get_time() << "CUDA sieve cpu-check: window=0 cpu=" << cpu_count
            << " gpu=" << gpu_count
            << " delta=" << static_cast<int64_t>(cpu_count) - static_cast<int64_t>(gpu_count);
      }
      extra_verbose_log(chk.str());
    }
  }
}

void GPUFermat::prototype_sieve_single(const SievePrototypeParams &params) {
  lastPrototypeCount = 0;
  prototypeOffsets.clear();
  resetPrototypeWindowState();

  if (params.window_size == 0)
    return;

  const bool has_host_bitmap =
      (params.sieve_bytes != nullptr && params.sieve_byte_len != 0);
  const bool has_residue_snapshot =
      (params.prime_starts != nullptr && params.prime_count != 0);
  if (!has_host_bitmap && !has_residue_snapshot)
    return;
  if (has_residue_snapshot &&
      (!sievePrimesConfigured || params.prime_count > configuredPrimeCount)) {
    if (Opts::get_instance()->has_extra_vb()) {
      std::ostringstream ss;
      ss << get_time()
         << "CUDA sieve prototype missing prime configuration for snapshot";
      extra_verbose_log(ss.str());
    }
    return;
  }

  ensurePrototypeOutputCapacity(params.window_size);

  const uint32_t bitmap_words =
      static_cast<uint32_t>((static_cast<uint64_t>(params.window_size) + 31ull) >> 5);
  if (bitmap_words == 0)
    return;
  ensurePrototypeBitmapCapacity(bitmap_words);

  const uint64_t round_offset =
      static_cast<uint64_t>(params.window_size) * params.sieve_round;
  const uint64_t absolute_base = params.sieve_base + round_offset;
  const uint32_t base_parity = static_cast<uint32_t>(absolute_base & 1ull);

  cudaStream_t stream = streams[0];
  if (has_host_bitmap) {
    uint8_t *bitmap_host_bytes = sievePrototypeBitmap.HostData;
    const size_t bitmap_bytes = static_cast<size_t>(bitmap_words) * sizeof(uint32_t);
    const size_t copy_bytes = std::min(static_cast<size_t>(params.sieve_byte_len),
                                       bitmap_bytes);
    memcpy(bitmap_host_bytes, params.sieve_bytes, copy_bytes);
    if (bitmap_bytes > copy_bytes)
      memset(bitmap_host_bytes + copy_bytes, 0, bitmap_bytes - copy_bytes);

    sievePrototypeBitmap.copyToDevice(stream, bitmap_words);
  } else if (has_residue_snapshot) {
    ensurePrototypeResidueCapacity(params.prime_count);
    memcpy(buffer_cast<uint32_t>(sievePrototypeResidues.HostData),
           params.prime_starts,
           params.prime_count * sizeof(uint32_t));
    sievePrototypeResidues.copyToDevice(stream, params.prime_count);
    const size_t bitmap_bytes = static_cast<size_t>(bitmap_words) * sizeof(uint32_t);
    CUDA_CHECK(cudaMemsetAsync(sievePrototypeBitmap.DeviceData,
                               0,
                               bitmap_bytes,
                               stream));

    const uint32_t threads = GroupSize;
    const uint32_t blocks =
        std::max(1u, (params.prime_count + threads - 1u) / threads);
    sieveBuildBitmapKernel<<<blocks, threads, 0, stream>>>(
        buffer_cast<const uint32_t>(sievePrototypePrimes.DeviceData),
        buffer_cast<const uint32_t>(sievePrototypeResidues.DeviceData),
        params.prime_count,
        params.window_size,
        buffer_cast<uint32_t>(sievePrototypeBitmap.DeviceData));
    CUDA_CHECK(cudaGetLastError());
  }

  const bool compact_scan_enabled = Opts::get_instance()->use_cuda_sieve_proto();
  if (compact_scan_enabled) {
    ensurePrototypeWindowDescCapacity(1u);
    PrototypeWindowDesc *desc_host =
        buffer_cast<PrototypeWindowDesc>(sievePrototypeWindowDescs.HostData);
    desc_host[0].word_start = 0u;
    desc_host[0].word_count = bitmap_words;
    desc_host[0].bit_length = params.window_size;
    desc_host[0].output_base = 0u;
    desc_host[0].base_parity = base_parity;
    desc_host[0].residue_offset = 0u;
    desc_host[0].residue_count = 0u;

    if (run_compact_scan(1u, params.window_size, stream)) {
      if (Opts::get_instance()->has_extra_vb()) {
        std::ostringstream ss;
        ss << get_time() << "CUDA sieve prototype window " << params.window_size
           << " start=" << params.sieve_base
           << " round=" << params.sieve_round
           << " candidates=" << lastPrototypeCount;
        extra_verbose_log(ss.str());
      }
      return;
    }
  }

  const uint32_t threads = GroupSize;
  const uint32_t blocks =
      std::max(1u, (params.window_size + threads - 1u) / threads);

  const uint32_t *bitmap_device =
      buffer_cast<const uint32_t>(sievePrototypeBitmap.DeviceData);
  uint32_t *output_device =
      buffer_cast<uint32_t>(sievePrototypeOutput.DeviceData);

  sievePrototypeKernel<<<blocks, threads, 0, stream>>>(
      bitmap_device,
      params.window_size,
      base_parity,
      output_device);
  CUDA_CHECK(cudaGetLastError());

  sievePrototypeOutput.copyToHost(stream, params.window_size);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const uint32_t sentinel = 0xFFFFFFFFu;
  prototypeOffsets.reserve(params.window_size / 64u + 1u);
  for (uint32_t i = 0; i < params.window_size; ++i) {
    const uint32_t value =
        buffer_cast<const uint32_t>(sievePrototypeOutput.HostData)[i];
    if (value != sentinel)
      prototypeOffsets.push_back(value);
  }

  lastPrototypeCount = static_cast<uint32_t>(prototypeOffsets.size());
  prototypeWindowOffsets.clear();
  prototypeWindowOffsets.push_back(0u);
  prototypeWindowOffsets.push_back(lastPrototypeCount);
  lastPrototypeWindowCount = (params.window_size > 0) ? 1u : 0u;

  if (Opts::get_instance()->has_extra_vb()) {
    std::ostringstream ss;
    ss << get_time() << "CUDA sieve prototype window " << params.window_size
       << " start=" << params.sieve_base
       << " round=" << params.sieve_round
       << " candidates=" << lastPrototypeCount;
    extra_verbose_log(ss.str());
  }
}

const uint32_t *GPUFermat::prototype_offsets_data() const {
  return prototypeOffsets.empty() ? nullptr : prototypeOffsets.data();
}

uint32_t GPUFermat::prototype_offsets_count() const {
  return lastPrototypeCount;
}

const uint32_t *GPUFermat::prototype_window_offsets() const {
  return prototypeWindowOffsets.empty() ? nullptr : prototypeWindowOffsets.data();
}

uint32_t GPUFermat::prototype_window_count() const {
  return lastPrototypeWindowCount;
}

unsigned GPUFermat::get_stream_count() const {
  return static_cast<unsigned>(streams.size());
}

void GPUFermat::ensureMarkingResidueCapacity(size_t required) {
  ensurePrototypeResidueCapacity(required);
}

void GPUFermat::upload_prime_residues_to_device(const uint32_t *residues,
                                                 uint32_t count,
                                                 uint32_t **device_ptr) {
  if (!residues || count == 0) {
    *device_ptr = nullptr;
    return;
  }
  
  ensureMarkingResidueCapacity(count);
  memcpy(sievePrototypeResidues.HostData, residues, count * sizeof(uint32_t));
  sievePrototypeResidues.copyToDevice(streams[0], count);
  *device_ptr = buffer_cast<uint32_t>(sievePrototypeResidues.DeviceData);
}

void GPUFermat::mark_sieve_window_optimized(const SieveMarkingJob &job) {
  if (!initialized || job.prime_count == 0)
    return;
  
  const uint32_t window_words = (job.window_size + 31) >> 5;
  ensurePrototypeBitmapCapacity(window_words);

  const uint32_t *device_residues = job.gpu_prime_residues;
  if (!device_residues && job.host_prime_residues) {
    uint32_t *uploaded = nullptr;
    upload_prime_residues_to_device(job.host_prime_residues,
                                    job.prime_count,
                                    &uploaded);
    device_residues = uploaded;
  }
  if (!device_residues) {
    if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
      std::ostringstream ss;
      ss << get_time() << " CUDA mark skipped: missing residue buffer";
      extra_verbose_log(ss.str());
    }
    return;
  }
  
  // Clear bitmap (zeros = prime candidates)
  uint8_t *bitmap_host = sievePrototypeBitmap.HostData;
  memset(bitmap_host, 0, window_words * sizeof(uint32_t));
  sievePrototypeBitmap.copyToDevice(streams[0], window_words);
  
  // Launch optimized per-word kernel
  const uint32_t threads_per_block = 256;
  const uint32_t blocks = (window_words + threads_per_block - 1) / threads_per_block;
  
  cudaStream_t stream = streams[0];
  
  sievePrototypeMarkCompositesOptimized<<<blocks, threads_per_block, 0, stream>>>(
      buffer_cast<uint32_t>(sievePrototypeBitmap.DeviceData),
      window_words,
      job.base_offset,
      device_residues,
      job.prime_count,
      job.shift);
  
  CUDA_CHECK(cudaGetLastError());
  
  auto t_start = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  
  if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
    std::ostringstream ss;
    ss << get_time() << " CUDA mark (optimized) window=" << job.base_offset
       << " size=" << job.window_size << " primes=" << job.prime_count
       << " time=" << duration_ms << "ms";
    extra_verbose_log(ss.str());
  }
}

void GPUFermat::mark_sieve_window_per_prime(const SieveMarkingJob &job) {
  if (!initialized || job.prime_count == 0)
    return;
  
  const uint32_t window_words = (job.window_size + 31) >> 5;
  ensurePrototypeBitmapCapacity(window_words);

  const uint32_t *device_residues = job.gpu_prime_residues;
  if (!device_residues && job.host_prime_residues) {
    uint32_t *uploaded = nullptr;
    upload_prime_residues_to_device(job.host_prime_residues,
                                    job.prime_count,
                                    &uploaded);
    device_residues = uploaded;
  }
  if (!device_residues) {
    if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
      std::ostringstream ss;
      ss << get_time() << " CUDA mark skipped: missing residue buffer";
      extra_verbose_log(ss.str());
    }
    return;
  }
  
  // Clear bitmap
  uint8_t *bitmap_host = sievePrototypeBitmap.HostData;
  memset(bitmap_host, 0, window_words * sizeof(uint32_t));
  sievePrototypeBitmap.copyToDevice(streams[0], window_words);
  
  // Launch per-prime kernel
  const uint32_t threads_per_block = 256;
  const uint32_t blocks = (job.prime_count + threads_per_block - 1) / threads_per_block;
  
  cudaStream_t stream = streams[0];
  
  sievePrototypeMarkCompositesPerPrime<<<blocks, threads_per_block, 0, stream>>>(
      buffer_cast<uint32_t>(sievePrototypeBitmap.DeviceData),
      job.window_size,
      job.base_offset,
      device_residues,
      job.prime_count);
  
  CUDA_CHECK(cudaGetLastError());
  
  auto t_start = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  
  if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
    std::ostringstream ss;
    ss << get_time() << " CUDA mark (per-prime) window=" << job.base_offset
       << " size=" << job.window_size << " primes=" << job.prime_count
       << " time=" << duration_ms << "ms";
    extra_verbose_log(ss.str());
  }
}

void GPUFermat::mark_sieve_window(const SieveMarkingJob &job) {
  // Wrapper that calls optimized version by default
  mark_sieve_window_optimized(job);
}

void GPUFermat::gpu_mark_sieve_window(const SieveMarkingJob &job) {
  // Public interface - delegates to optimized implementation
  mark_sieve_window_optimized(job);
}

void GPUFermat::gpu_mark_sieve_optimized(const SieveMarkingJob &job) {
  // Public interface - explicit call to optimized variant
  mark_sieve_window_optimized(job);
}

void GPUFermat::benchmark_montgomery_mul() {
  const uint32_t sample_count = 1024u;
  const uint32_t rounds = 128u;
  std::vector<BigInt> hostA(sample_count);
  std::vector<BigInt> hostB(sample_count);
  std::vector<BigInt> hostMod(sample_count);
  std::vector<uint32_t> hostInv(sample_count);
  std::vector<BigInt> hostClassicOut(sample_count);
  std::vector<BigInt> hostCombaOut(sample_count);

  auto fill_random_bigint = [this](BigInt &n) {
    for (int i = 0; i < kOperandSize; ++i)
      n.limb[i] = rand32();
  };

  for (uint32_t i = 0; i < sample_count; ++i) {
    fill_random_bigint(hostA[i]);
    fill_random_bigint(hostB[i]);
    fill_random_bigint(hostMod[i]);
    hostMod[i].limb[0] |= 1u;
    hostMod[i].limb[kOperandSize - 1] |= 0x80000000u;
    hostInv[i] = montgomery_inverse32(hostMod[i].limb[0]);
  }

  const size_t bigIntBytes = static_cast<size_t>(sample_count) * sizeof(BigInt);
  const size_t invBytes = static_cast<size_t>(sample_count) * sizeof(uint32_t);

  BigInt *dA = nullptr;
  BigInt *dB = nullptr;
  BigInt *dMod = nullptr;
  BigInt *dClassicOut = nullptr;
  BigInt *dUnrolledOut = nullptr;
  BigInt *dCombaOut = nullptr;
  uint32_t *dInv = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dA), bigIntBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dB), bigIntBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dMod), bigIntBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dClassicOut), bigIntBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dUnrolledOut), bigIntBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dCombaOut), bigIntBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dInv), invBytes));

  CUDA_CHECK(cudaMemcpy(dA,
                        hostA.data(),
                        bigIntBytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB,
                        hostB.data(),
                        bigIntBytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dMod,
                        hostMod.data(),
                        bigIntBytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dInv,
                        hostInv.data(),
                        invBytes,
                        cudaMemcpyHostToDevice));

  const uint32_t threads = GroupSize;
  const uint32_t blocks = std::max(1u, (sample_count + threads - 1u) / threads);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  float msClassic = 0.0f;
  CUDA_CHECK(cudaEventRecord(start));
  montgomeryMulClassicKernel<<<blocks, threads>>>(dA,
                                                  dB,
                                                  dMod,
                                                  dInv,
                                                  dClassicOut,
                                                  sample_count,
                                                  rounds);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&msClassic, start, stop));

  float msUnrolled = 0.0f;
  CUDA_CHECK(cudaEventRecord(start));
  montgomeryMulUnrolledKernel<<<blocks, threads>>>(dA,
                                                   dB,
                                                   dMod,
                                                   dInv,
                                                   dUnrolledOut,
                                                   sample_count,
                                                   rounds);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&msUnrolled, start, stop));

  float msComba = 0.0f;
  CUDA_CHECK(cudaEventRecord(start));
  montgomeryMulCombaKernel<<<blocks, threads>>>(dA,
                                                dB,
                                                dMod,
                                                dInv,
                                                dCombaOut,
                                                sample_count,
                                                rounds);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&msComba, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  cudaFuncAttributes attrClassic{};
  cudaFuncAttributes attrUnrolled{};
  cudaFuncAttributes attrComba{};
  CUDA_CHECK(cudaFuncGetAttributes(&attrClassic, montgomeryMulClassicKernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attrUnrolled, montgomeryMulUnrolledKernel));
  CUDA_CHECK(cudaFuncGetAttributes(&attrComba, montgomeryMulCombaKernel));

  CUDA_CHECK(cudaMemcpy(hostClassicOut.data(),
                        dClassicOut,
                        bigIntBytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hostCombaOut.data(),
                        dCombaOut,
                        bigIntBytes,
                        cudaMemcpyDeviceToHost));

  const double totalOps = static_cast<double>(sample_count) * rounds;
  const double classicPerSec = (msClassic > 0.0f)
                                   ? (totalOps * 1000.0 / msClassic)
                                   : 0.0;
  const double unrolledPerSec = (msUnrolled > 0.0f)
                                    ? (totalOps * 1000.0 / msUnrolled)
                                    : 0.0;
  const double combaPerSec = (msComba > 0.0f)
                                 ? (totalOps * 1000.0 / msComba)
                                 : 0.0;

  pthread_mutex_lock(&io_mutex);
  cout << get_time() << "Montgomery classic: " << msClassic << " ms, "
       << classicPerSec / 1e6 << " Mmul/s, regs=" << attrClassic.numRegs
       << endl;
  cout << get_time() << "Montgomery unrolled: " << msUnrolled << " ms, "
       << unrolledPerSec / 1e6 << " Mmul/s, regs=" << attrUnrolled.numRegs
       << endl;
    cout << get_time() << "Montgomery comba: " << msComba << " ms, "
      << combaPerSec / 1e6 << " Mmul/s, regs=" << attrComba.numRegs
      << endl;

  unsigned mismatch = 0;
  unsigned first_mismatch = sample_count;
  const unsigned check_count = std::min(16u, sample_count);
  for (unsigned i = 0; i < check_count; ++i) {
    for (int limb = 0; limb < kOperandSize; ++limb) {
      if (hostClassicOut[i].limb[limb] != hostCombaOut[i].limb[limb]) {
        mismatch++;
        if (first_mismatch == sample_count)
          first_mismatch = i;
        break;
      }
    }
  }
  if (mismatch == 0) {
    cout << get_time() << "Montgomery comba check: OK (" << check_count
         << " samples)" << endl;
  } else {
    cout << get_time() << "Montgomery comba check: " << mismatch
         << " mismatches in " << check_count << " samples" << endl;
    if (first_mismatch < sample_count) {
      cout << get_time() << "Comba mismatch sample " << first_mismatch << " classic:";
      for (int limb = 0; limb < kOperandSize; ++limb)
        cout << " " << std::hex << std::setw(8) << std::setfill('0')
             << hostClassicOut[first_mismatch].limb[limb];
      cout << std::dec << endl;
      cout << get_time() << "Comba mismatch sample " << first_mismatch << " comba:";
      for (int limb = 0; limb < kOperandSize; ++limb)
        cout << " " << std::hex << std::setw(8) << std::setfill('0')
             << hostCombaOut[first_mismatch].limb[limb];
      cout << std::dec << endl;
    }
  }
  pthread_mutex_unlock(&io_mutex);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dMod);
  cudaFree(dClassicOut);
  cudaFree(dUnrolledOut);
  cudaFree(dCombaOut);
  cudaFree(dInv);
}

void GPUFermat::benchmark() {
  benchmark_montgomery_mul();
  test_gpu();
}

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
  uint32_t *prime_base = buffer_cast<uint32_t>(primeBase.HostData);
  uint32_t *primes = buffer_cast<uint32_t>(numbers.HostData);
  ResultWord *results = buffer_cast<ResultWord>(gpuResults.HostData);

  mpz_class mpz(rand32());
  for (int i = 0; i < 8; ++i) {
    mpz <<= 32;
    mpz += rand32();
  }

  if (!(mpz.get_ui() & 0x1))
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

  mpz_class base_mpz;
  mpz_import(base_mpz.get_mpz_t(), kOperandSize, -1, 4, 0, 0, prime_base);

  for (unsigned i = 0; i < size; ++i) {
    uint32_t small_offset = rand32() % (1u << 24); // small offset to fit in 32 bits
    if (i % 2 == 0) {
      // Generate a prime close to base: base + small random offset
      mpz_class candidate = base_mpz + small_offset;
      mpz_nextprime(candidate.get_mpz_t(), candidate.get_mpz_t());
      mpz_class offset = candidate - base_mpz;
      primes[i] = offset.get_ui() & 0xffffffffu;
    } else {
      // Generate composite close to base: base + small_offset + 1
      mpz_class composite = base_mpz + small_offset + 1;
      mpz_class offset = composite - base_mpz;
      primes[i] = offset.get_ui() & 0xffffffffu;
    }

    if (i % 23 == 0)
      printf("\rcreating test data: %u  \r", size - i);
  }
  printf("\r                                             \r");

  fermat_gpu(size);

  unsigned failures = 0;
  std::vector<uint32_t> failing_indices;
  std::vector<uint8_t> expected(size);
  for (unsigned i = 0; i < size; ++i) {
    mpz_t check;
    mpz_init(check);
    mpz_import(check, kOperandSize, -1, 4, 0, 0, prime_base);
    mpz_add_ui(check, check, primes[i]);
    // mpz_probab_prime_p returns 0 for composite, >0 for (probable) prime.
    // Device `results` uses 1 => prime, 0 => composite. Map accordingly.
    const uint8_t is_prime = (mpz_probab_prime_p(check, 25) != 0);
    // Device kernel writes 1 for probable prime, 0 for composite.
    expected[i] = is_prime ? 1u : 0u;
    mpz_clear(check);

    if (results[i] != static_cast<ResultWord>(expected[i])) {
      if (failures < 8) {
        printf("[CUDA TEST] MISMATCH idx=%u expected=%u got=%u number=%u\n",
               i,
               expected[i],
               results[i],
               primes[i]);
        // Print MPZ-based check and the candidate value (hex) for diagnostics
        mpz_t check2;
        mpz_init(check2);
        mpz_import(check2, kOperandSize, -1, 4, 0, 0, prime_base);
        mpz_add_ui(check2, check2, primes[i]);
        int cpu_res = mpz_probab_prime_p(check2, 25);
        char *num_hex = mpz_get_str(NULL, 16, check2);
        printf("[CUDA TEST] CPU mpz_probab_prime_p=%d num_hex=%s\n", cpu_res, num_hex);
        free(num_hex);
        mpz_clear(check2);
      } else if (failures == 8) {
        printf("[CUDA TEST] Further mismatches suppressed...\n");
      }
      if (failing_indices.size() < 16)
        failing_indices.push_back(i);
      failures++;
    }
  }

  if (failures)
    printf("GPU Test failed: %u mismatched result(s)\n", failures);
  else
    printf("GPU Test worked\n");

  // Run diagnostic: dump device-built limbs for first failing indices
  if (!failing_indices.empty()) {
    const unsigned sampleCount = static_cast<unsigned>(failing_indices.size());
    std::vector<uint32_t> sample_indices(sampleCount);
    for (unsigned s = 0; s < sampleCount; ++s)
      sample_indices[s] = failing_indices[s];

    uint32_t *d_numbers = nullptr;
    uint32_t *d_sample_indices = nullptr;

    uint32_t *d_out_limbs = nullptr;
    uint32_t *host_out_limbs = nullptr;
    uint32_t *host_out_carry = nullptr;

    CUDA_CHECK(cudaMalloc(&d_numbers, size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sample_indices, sampleCount * sizeof(uint32_t)));
    // allocate an extra per-sample word for final carry
    CUDA_CHECK(cudaMalloc(&d_out_limbs, (sampleCount * kOperandSize + sampleCount) * sizeof(uint32_t)));

    host_out_limbs = (uint32_t *)malloc(sampleCount * kOperandSize * sizeof(uint32_t));
    host_out_carry = (uint32_t *)malloc(sampleCount * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(d_numbers, primes, size * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sample_indices, sample_indices.data(), sampleCount * sizeof(uint32_t), cudaMemcpyHostToDevice));

    const unsigned threads = 128;
    const unsigned blocks = (sampleCount + threads - 1) / threads;
    dump_candidate_limbs_kernel<<<blocks, threads>>>(d_numbers, d_sample_indices, d_out_limbs, sampleCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host_out_limbs, d_out_limbs, sampleCount * kOperandSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_out_carry, d_out_limbs + (sampleCount * kOperandSize), sampleCount * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Also capture per-limb carries (carry after each limb addition)
    uint32_t *d_out_carries = nullptr;
    uint32_t *host_out_carries = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_carries, sampleCount * kOperandSize * sizeof(uint32_t)));
    host_out_carries = (uint32_t *)malloc(sampleCount * kOperandSize * sizeof(uint32_t));

    dump_candidate_carries_kernel<<<blocks, threads>>>(d_numbers, d_sample_indices, d_out_carries, sampleCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out_carries, d_out_carries, sampleCount * kOperandSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Print diagnostic per sample: device limbs and host-expected limbs
    for (unsigned s = 0; s < sampleCount; ++s) {
      unsigned idx = sample_indices[s];
      printf("[CUDA DIAG] sample %u idx=%u offset=%u\n", s, idx, primes[idx]);

      printf("[CUDA DIAG] device limbs: ");
      for (int l = 0; l < kOperandSize; ++l)
        printf("%08x ", host_out_limbs[s * kOperandSize + l]);
      printf(" carry=%08x\n", host_out_carry[s]);

      // print per-limb carry trace
      printf("[CUDA DIAG] device carries(after limb): ");
      for (int l = 0; l < kOperandSize; ++l)
        printf("%08x ", host_out_carries[s * kOperandSize + l]);
      printf("\n");

      // build host mpz and export limbs for comparison
      mpz_t check;
      mpz_init(check);
      mpz_import(check, kOperandSize, -1, 4, 0, 0, prime_base);
      mpz_add_ui(check, check, primes[idx]);
      // allocate slightly larger buffer to catch unexpected exported limb counts
      const size_t host_buf_words = kOperandSize + 4;
      uint32_t *host_limbs = (uint32_t *)malloc(host_buf_words * sizeof(uint32_t));
      memset(host_limbs, 0, host_buf_words * sizeof(uint32_t));
      size_t host_exported = 0;
      mpz_export(host_limbs, &host_exported, -1, sizeof(uint32_t), 0, 0, check);
      size_t mpz_bits = mpz_sizeinbase(check, 2);
      printf("[CUDA DIAG] host_exported=%zu mpz_bits=%zu\n", host_exported, mpz_bits);
      printf("[CUDA DIAG] host  limbs: ");
      for (int l = 0; l < kOperandSize; ++l)
        printf("%08x ", host_limbs[l]);
      printf("\n");
      free(host_limbs);
      mpz_clear(check);
    }

    free(host_out_limbs);
    free(host_out_carries);
    free(host_out_carry);
    CUDA_CHECK(cudaFree(d_numbers));
    CUDA_CHECK(cudaFree(d_sample_indices));
    CUDA_CHECK(cudaFree(d_out_limbs));
    CUDA_CHECK(cudaFree(d_out_carries));

    // Run montgomery classic vs unrolled comparison for first failing samples
    {
      uint32_t *d_out_classic = nullptr;
      uint32_t *d_out_unrolled = nullptr;
      uint32_t *host_out_classic = nullptr;
      uint32_t *host_out_unrolled = nullptr;

      CUDA_CHECK(cudaMalloc(&d_numbers, size * sizeof(uint32_t)));
      CUDA_CHECK(cudaMemcpy(d_numbers, primes, size * sizeof(uint32_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMalloc(&d_sample_indices, sampleCount * sizeof(uint32_t)));
      CUDA_CHECK(cudaMemcpy(d_sample_indices, sample_indices.data(), sampleCount * sizeof(uint32_t), cudaMemcpyHostToDevice));

      CUDA_CHECK(cudaMalloc(&d_out_classic, sampleCount * kOperandSize * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_unrolled, sampleCount * kOperandSize * sizeof(uint32_t)));
      host_out_classic = (uint32_t *)malloc(sampleCount * kOperandSize * sizeof(uint32_t));
      host_out_unrolled = (uint32_t *)malloc(sampleCount * kOperandSize * sizeof(uint32_t));

      const unsigned threads2 = 128;
      const unsigned blocks2 = (sampleCount + threads2 - 1) / threads2;
      montgomeryCompareKernel<<<blocks2, threads2>>>(d_numbers, d_sample_indices, d_out_classic, d_out_unrolled, sampleCount);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(host_out_classic, d_out_classic, sampleCount * kOperandSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_unrolled, d_out_unrolled, sampleCount * kOperandSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));

      for (unsigned s = 0; s < sampleCount; ++s) {
        printf("[MONTY COMP] sample %u idx=%u\n", s, sample_indices[s]);
        printf("[MONTY COMP] classic:   ");
        for (int l = 0; l < kOperandSize; ++l)
          printf("%08x ", host_out_classic[s * kOperandSize + l]);
        printf("\n");
        printf("[MONTY COMP] unrolled:  ");
        for (int l = 0; l < kOperandSize; ++l)
          printf("%08x ", host_out_unrolled[s * kOperandSize + l]);
        printf("\n");
      }

      free(host_out_classic);
      free(host_out_unrolled);
      CUDA_CHECK(cudaFree(d_out_classic));
      CUDA_CHECK(cudaFree(d_out_unrolled));
      CUDA_CHECK(cudaFree(d_numbers));
      CUDA_CHECK(cudaFree(d_sample_indices));
    }

    // Run additional trace kernels to capture invValue[] and final op1 of both unrolled and classic paths
    {
      uint32_t *d_out_inv = nullptr;
      uint32_t *d_out_final = nullptr;
      uint32_t *host_out_inv = nullptr;
      uint32_t *host_out_final = nullptr;

      uint32_t *d_out_inv_classic = nullptr;
      uint32_t *d_out_final_classic = nullptr;
      uint32_t *host_out_inv_classic = nullptr;
      uint32_t *host_out_final_classic = nullptr;

      uint32_t *d_out_acc = nullptr; // unrolled acc low
      uint32_t *d_out_acc_high = nullptr; // unrolled acc high
      uint32_t *host_out_acc = nullptr;
      uint32_t *host_out_acc_high = nullptr;

      uint32_t *d_out_acc_classic = nullptr;
      uint32_t *d_out_acc_classic_high = nullptr;
      uint32_t *host_out_acc_classic = nullptr;
      uint32_t *host_out_acc_classic_high = nullptr;
      uint32_t *d_out_prod = nullptr;
      uint32_t *d_out_prod_high = nullptr;
      uint32_t *host_out_prod = nullptr;
      uint32_t *host_out_prod_high = nullptr;

      uint32_t *d_out_prod_classic = nullptr;
      uint32_t *d_out_prod_classic_high = nullptr;
      uint32_t *host_out_prod_classic = nullptr;
      uint32_t *host_out_prod_classic_high = nullptr;
      uint32_t *host_out_steps = nullptr;
      uint32_t *host_out_steps_high = nullptr;
      uint32_t *host_out_steps_classic = nullptr;
      uint32_t *host_out_steps_classic_high = nullptr;

      CUDA_CHECK(cudaMalloc(&d_numbers, size * sizeof(uint32_t)));
      CUDA_CHECK(cudaMemcpy(d_numbers, primes, size * sizeof(uint32_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMalloc(&d_sample_indices, sampleCount * sizeof(uint32_t)));
      CUDA_CHECK(cudaMemcpy(d_sample_indices, sample_indices.data(), sampleCount * sizeof(uint32_t), cudaMemcpyHostToDevice));

      CUDA_CHECK(cudaMalloc(&d_out_inv, sampleCount * kOperandSize * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_final, sampleCount * kOperandSize * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_inv_classic, sampleCount * kOperandSize * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_final_classic, sampleCount * kOperandSize * sizeof(uint32_t)));

      // allocate acc trace buffers (4 blocks per sample)
      CUDA_CHECK(cudaMalloc(&d_out_acc, sampleCount * 4 * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_acc_high, sampleCount * 4 * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_acc_classic, sampleCount * 4 * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_acc_classic_high, sampleCount * 4 * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_prod, sampleCount * 3 * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_prod_high, sampleCount * 3 * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_prod_classic, sampleCount * 3 * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_prod_classic_high, sampleCount * 3 * sizeof(uint32_t)));
      // step buffers (3 steps per sample: after prod0, after prod1, after prod2)
      uint32_t *d_out_steps = nullptr;
      uint32_t *d_out_steps_high = nullptr;
      uint32_t *d_out_steps_classic = nullptr;
      uint32_t *d_out_steps_classic_high = nullptr;
      CUDA_CHECK(cudaMalloc(&d_out_steps, sampleCount * 3 * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_steps_high, sampleCount * 3 * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_steps_classic, sampleCount * 3 * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_out_steps_classic_high, sampleCount * 3 * sizeof(uint32_t)));

      host_out_inv = (uint32_t *)malloc(sampleCount * kOperandSize * sizeof(uint32_t));
      host_out_final = (uint32_t *)malloc(sampleCount * kOperandSize * sizeof(uint32_t));
      host_out_inv_classic = (uint32_t *)malloc(sampleCount * kOperandSize * sizeof(uint32_t));
      host_out_final_classic = (uint32_t *)malloc(sampleCount * kOperandSize * sizeof(uint32_t));

      host_out_acc = (uint32_t *)malloc(sampleCount * 4 * sizeof(uint32_t));
      host_out_acc_high = (uint32_t *)malloc(sampleCount * 4 * sizeof(uint32_t));
      host_out_acc_classic = (uint32_t *)malloc(sampleCount * 4 * sizeof(uint32_t));
      host_out_acc_classic_high = (uint32_t *)malloc(sampleCount * 4 * sizeof(uint32_t));
      host_out_prod = (uint32_t *)malloc(sampleCount * 3 * sizeof(uint32_t));
      host_out_prod_high = (uint32_t *)malloc(sampleCount * 3 * sizeof(uint32_t));
      host_out_prod_classic = (uint32_t *)malloc(sampleCount * 3 * sizeof(uint32_t));
      host_out_prod_classic_high = (uint32_t *)malloc(sampleCount * 3 * sizeof(uint32_t));
      host_out_steps = (uint32_t *)malloc(sampleCount * 3 * sizeof(uint32_t));
      host_out_steps_high = (uint32_t *)malloc(sampleCount * 3 * sizeof(uint32_t));
      host_out_steps_classic = (uint32_t *)malloc(sampleCount * 3 * sizeof(uint32_t));
      host_out_steps_classic_high = (uint32_t *)malloc(sampleCount * 3 * sizeof(uint32_t));

      const unsigned threads3 = 128;
      const unsigned blocks3 = (sampleCount + threads3 - 1) / threads3;

      // unrolled tracer (with acc outputs)
      montgomeryTraceKernel<<<blocks3, threads3>>>(d_numbers, d_sample_indices, d_out_inv, d_out_final, d_out_acc, d_out_acc_high, d_out_prod, d_out_prod_high, d_out_steps, d_out_steps_high, sampleCount);
      CUDA_CHECK(cudaGetLastError());
      // classic tracer
      montgomeryClassicTraceKernel<<<blocks3, threads3>>>(d_numbers, d_sample_indices, d_out_inv_classic, d_out_final_classic, d_out_acc_classic, d_out_acc_classic_high, d_out_prod_classic, d_out_prod_classic_high, d_out_steps_classic, d_out_steps_classic_high, sampleCount);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(host_out_steps, d_out_steps, sampleCount * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_out_steps_high, d_out_steps_high, sampleCount * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_out_steps_classic, d_out_steps_classic, sampleCount * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_out_steps_classic_high, d_out_steps_classic_high, sampleCount * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaMemcpy(host_out_inv, d_out_inv, sampleCount * kOperandSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_final, d_out_final, sampleCount * kOperandSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_inv_classic, d_out_inv_classic, sampleCount * kOperandSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_final_classic, d_out_final_classic, sampleCount * kOperandSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_acc, d_out_acc, sampleCount * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_acc_high, d_out_acc_high, sampleCount * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_acc_classic, d_out_acc_classic, sampleCount * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_acc_classic_high, d_out_acc_classic_high, sampleCount * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_prod, d_out_prod, sampleCount * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_prod_high, d_out_prod_high, sampleCount * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_prod_classic, d_out_prod_classic, sampleCount * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_out_prod_classic_high, d_out_prod_classic_high, sampleCount * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

      for (unsigned s = 0; s < sampleCount; ++s) {
        printf("[TRACE] sample %u idx=%u\n", s, sample_indices[s]);
        printf("[TRACE] unrolled invValue: ");
        for (int l = 0; l < kOperandSize; ++l)
          printf("%08x ", host_out_inv[s * kOperandSize + l]);
        printf("\n");
        printf("[TRACE] unrolled final op1: ");
        for (int l = 0; l < kOperandSize; ++l)
          printf("%08x ", host_out_final[s * kOperandSize + l]);
        printf("\n");
        printf("[ACC] unrolled accBefore (blk0..3): ");
        for (int b = 0; b < 4; ++b) {
          uint32_t low = host_out_acc[s * 4 + b];
          uint32_t high = host_out_acc_high[s * 4 + b];
          printf("%08x%08x ", high, low);
        }
        printf("\n");
        printf("[PROD] unrolled block1 products (p0,p1,p2): ");
        for (int p = 0; p < 3; ++p) {
          uint32_t low = host_out_prod[s * 3 + p];
          uint32_t high = host_out_prod_high[s * 3 + p];
          printf("%08x%08x ", high, low);
        }
        printf("\n");
        printf("[STEPS] unrolled block1 steps (after prod0, after prod1, after prod2): ");
        for (int p = 0; p < 3; ++p) {
          uint32_t low = host_out_steps[s * 3 + p];
          uint32_t high = host_out_steps_high[s * 3 + p];
          printf("%08x%08x ", high, low);
        }
        printf("\n");
        printf("[TRACE] classic invValue:   ");
        for (int l = 0; l < kOperandSize; ++l)
          printf("%08x ", host_out_inv_classic[s * kOperandSize + l]);
        printf("\n");
        printf("[TRACE] classic final op1:   ");
        for (int l = 0; l < kOperandSize; ++l)
          printf("%08x ", host_out_final_classic[s * kOperandSize + l]);
        printf("\n");
        printf("[ACC] classic accBefore (blk0..3): ");
        for (int b = 0; b < 4; ++b) {
          uint32_t low = host_out_acc_classic[s * 4 + b];
          uint32_t high = host_out_acc_classic_high[s * 4 + b];
          printf("%08x%08x ", high, low);
        }
        printf("\n");
        printf("[PROD] classic block1 products (p0,p1,p2): ");
        for (int p = 0; p < 3; ++p) {
          uint32_t low = host_out_prod_classic[s * 3 + p];
          uint32_t high = host_out_prod_classic_high[s * 3 + p];
          printf("%08x%08x ", high, low);
        }
        printf("\n");
        printf("[STEPS] classic block1 steps (after prod0, after prod1, after prod2): ");
        for (int p = 0; p < 3; ++p) {
          uint32_t low = host_out_steps_classic[s * 3 + p];
          uint32_t high = host_out_steps_classic_high[s * 3 + p];
          printf("%08x%08x ", high, low);
        }
        printf("\n");
      }

      free(host_out_inv);
      free(host_out_final);
      free(host_out_inv_classic);
      free(host_out_final_classic);
      free(host_out_steps);
      free(host_out_steps_high);
      free(host_out_steps_classic);
      free(host_out_steps_classic_high);
      CUDA_CHECK(cudaFree(d_out_inv));
      CUDA_CHECK(cudaFree(d_out_final));
      CUDA_CHECK(cudaFree(d_out_inv_classic));
      CUDA_CHECK(cudaFree(d_out_final_classic));
      CUDA_CHECK(cudaFree(d_out_steps));
      CUDA_CHECK(cudaFree(d_out_steps_high));
      CUDA_CHECK(cudaFree(d_out_steps_classic));
      CUDA_CHECK(cudaFree(d_out_steps_classic_high));
      CUDA_CHECK(cudaFree(d_numbers));
      CUDA_CHECK(cudaFree(d_sample_indices));
    }
  }
}

#endif /* USE_CUDA_BACKEND */
#endif /* CPU_ONLY */
