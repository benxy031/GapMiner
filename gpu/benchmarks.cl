__kernel void squareBenchmark320(__global uint32_t *m1,
                                 __global uint32_t *out,
                                 unsigned elementsNum)
{
#define OperandSize 10
#define GmpOperandSize 10
  unsigned globalSize = get_global_size(0);
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {
    uint32_t op1[OperandSize];
    for (unsigned j = 0; j < OperandSize; j++)
      op1[j] = m1[i*GmpOperandSize + j];  
    
    uint4 result[6];    
    
    uint4 op1v[3] = {
      (uint4){op1[0], op1[1], op1[2], op1[3]}, 
      (uint4){op1[4], op1[5], op1[6], op1[7]}, 
      (uint4){op1[8], op1[9], 0, 0}
    };

    for (unsigned repeatNum = 0; repeatNum < 512; repeatNum++) {
      sqrProductScan320(result, op1v);
      op1v[0] = result[0];
      op1v[1] = result[1];
      op1v[2].xy = result[2].xy;
    }
    
    uint32_t *pResult = (uint32_t*)result;
    for (unsigned j = 0; j < OperandSize*2; j++)
      out[i*OperandSize*2 + j] = pResult[j];
  }
#undef GmpOperandSize
#undef OperandSize
}

// Montgomery helpers are defined in procs.cl
void monMul320(uint32_t *op1, uint32_t *op2, uint32_t *mod, uint32_t invm);
void monSqr320(uint32_t *op, uint32_t *mod, uint32_t invm);
void monMul352(uint32_t *op1, uint32_t *op2, uint32_t *mod, uint32_t invm);
void monSqr352(uint32_t *op, uint32_t *mod, uint32_t invm);

__kernel void montgomeryMulBenchmark320(__global uint32_t *m1,
                                        __global uint32_t *m2,
                                        __global uint32_t *mod,
                                        __global uint32_t *invm,
                                        __global uint32_t *out,
                                        unsigned elementsNum,
                                        unsigned rounds)
{
#define OperandSize 10
  unsigned globalSize = get_global_size(0);
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {
    uint32_t op1[OperandSize];
    uint32_t op2[OperandSize];
    uint32_t modl[OperandSize];
    for (unsigned j = 0; j < OperandSize; ++j) {
      op1[j] = m1[i * OperandSize + j];
      op2[j] = m2[i * OperandSize + j];
      modl[j] = mod[i * OperandSize + j];
    }

    uint32_t inv = invm[i];
    for (unsigned r = 0; r < rounds; ++r)
      monMul320(op1, op2, modl, inv);

    for (unsigned j = 0; j < OperandSize; ++j)
      out[i * OperandSize + j] = op1[j];
  }
#undef OperandSize
}

__kernel void montgomerySqrBenchmark320(__global uint32_t *m1,
                                        __global uint32_t *mod,
                                        __global uint32_t *invm,
                                        __global uint32_t *out,
                                        unsigned elementsNum,
                                        unsigned rounds)
{
#define OperandSize 10
  unsigned globalSize = get_global_size(0);
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {
    uint32_t op1[OperandSize];
    uint32_t modl[OperandSize];
    for (unsigned j = 0; j < OperandSize; ++j) {
      op1[j] = m1[i * OperandSize + j];
      modl[j] = mod[i * OperandSize + j];
    }

    uint32_t inv = invm[i];
    for (unsigned r = 0; r < rounds; ++r)
      monSqr320(op1, modl, inv);

    for (unsigned j = 0; j < OperandSize; ++j)
      out[i * OperandSize + j] = op1[j];
  }
#undef OperandSize
}

__kernel void squareBenchmark352(__global uint32_t *m1,
                                 __global uint32_t *out,
                                 unsigned elementsNum)
{
#define OperandSize 11
#define GmpOperandSize 12  
  unsigned globalSize = get_global_size(0);
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {
    uint32_t op1[OperandSize];
    for (unsigned j = 0; j < OperandSize; j++)
      op1[j] = m1[i*GmpOperandSize + j];  
    
    uint4 result[6];    
    
    uint4 op1v[3] = {
      (uint4){op1[0], op1[1], op1[2], op1[3]}, 
      (uint4){op1[4], op1[5], op1[6], op1[7]}, 
      (uint4){op1[8], op1[9], op1[10], 0}
    };

    for (unsigned repeatNum = 0; repeatNum < 512; repeatNum++) {
      sqrProductScan352(result, op1v);
      op1v[0] = result[0];
      op1v[1] = result[1];
      op1v[2].xyz = result[2].xyz;
    }
    
    uint32_t *pResult = (uint32_t*)result;
    for (unsigned j = 0; j < OperandSize*2; j++)
      out[i*OperandSize*2 + j] = pResult[j];
  }
#undef GmpOperandSize
#undef OperandSize
}


__kernel void multiplyBenchmark320(__global uint32_t *m1,
                                   __global uint32_t *m2,
                                   __global uint32_t *out,
                                   unsigned elementsNum)
{
#define OperandSize 10
#define GmpOperandSize 10  
  unsigned globalSize = get_global_size(0);

  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {
    uint32_t op1[OperandSize];
    uint32_t op2[OperandSize];
    for (unsigned j = 0; j < OperandSize; j++)
      op1[j] = m1[i*GmpOperandSize + j];
    for (unsigned j = 0; j < OperandSize; j++)
      op2[j] = m2[i*GmpOperandSize + j];    
    
    uint4 result[6];    
    
    uint4 op1v[3] = {
      (uint4){op1[0], op1[1], op1[2], op1[3]}, 
      (uint4){op1[4], op1[5], op1[6], op1[7]}, 
      (uint4){op1[8], op1[9], 0, 0}
    };
    
    uint4 op2v[3] = {
      (uint4){op2[0], op2[1], op2[2], op2[3]}, 
      (uint4){op2[4], op2[5], op2[6], op2[7]}, 
      (uint4){op2[8], op2[9], 0, 0}
    };   

    for (unsigned repeatNum = 0; repeatNum < 512; repeatNum++) {
      mulProductScan320to320(result, op1v, op2v);
      op1v[0] = result[0];
      op1v[1] = result[1];
      op1v[2].xy = result[2].xy;
    }
    
    uint32_t *pResult = (uint32_t*)result;
    for (unsigned j = 0; j < OperandSize*2; j++)
      out[i*OperandSize*2 + j] = pResult[j];
  }
#undef GmpOperandSize
#undef OperandSize
}

__kernel void multiplyBenchmark352(__global uint32_t *m1,
                                   __global uint32_t *m2,
                                   __global uint32_t *out,
                                   unsigned elementsNum)
{
#define OperandSize 11
#define GmpOperandSize 12  
  unsigned globalSize = get_global_size(0);
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {
    uint32_t op1[OperandSize];
    uint32_t op2[OperandSize];
    for (unsigned j = 0; j < OperandSize; j++)
      op1[j] = m1[i*GmpOperandSize + j];
    for (unsigned j = 0; j < OperandSize; j++)
      op2[j] = m2[i*GmpOperandSize + j];    
    
    uint4 result[6];    
    
    uint4 op1v[3] = {
      (uint4){op1[0], op1[1], op1[2], op1[3]}, 
      (uint4){op1[4], op1[5], op1[6], op1[7]}, 
      (uint4){op1[8], op1[9], op1[10], 0}
    };
    
    uint4 op2v[3] = {
      (uint4){op2[0], op2[1], op2[2], op2[3]}, 
      (uint4){op2[4], op2[5], op2[6], op2[7]}, 
      (uint4){op2[8], op2[9], op2[10], 0}
    };   

    for (unsigned repeatNum = 0; repeatNum < 512; repeatNum++) {
      mulProductScan352to352(result, op1v, op2v);
      op1v[0] = result[0];
      op1v[1] = result[1];
      op1v[2].xyz = result[2].xyz;
    }
    
    uint32_t *pResult = (uint32_t*)&result;
    for (unsigned j = 0; j < OperandSize*2; j++)
      out[i*OperandSize*2 + j] = pResult[j];
  }
  
#undef GmpOperandSize
#undef OperandSize
}

__kernel void montgomeryMulBenchmark352(__global uint32_t *m1,
                                        __global uint32_t *m2,
                                        __global uint32_t *mod,
                                        __global uint32_t *invm,
                                        __global uint32_t *out,
                                        unsigned elementsNum,
                                        unsigned rounds)
{
#define OperandSize 11
#define GmpOperandSize 12
  unsigned globalSize = get_global_size(0);
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {
    uint32_t op1[OperandSize];
    uint32_t op2[OperandSize];
    uint32_t modl[OperandSize];
    for (unsigned j = 0; j < OperandSize; ++j) {
      op1[j] = m1[i * GmpOperandSize + j];
      op2[j] = m2[i * GmpOperandSize + j];
      modl[j] = mod[i * GmpOperandSize + j];
    }

    uint32_t inv = invm[i];
    for (unsigned r = 0; r < rounds; ++r)
      monMul352(op1, op2, modl, inv);

    for (unsigned j = 0; j < OperandSize; ++j)
      out[i * OperandSize + j] = op1[j];
  }
#undef GmpOperandSize
#undef OperandSize
}

__kernel void montgomerySqrBenchmark352(__global uint32_t *m1,
                                        __global uint32_t *mod,
                                        __global uint32_t *invm,
                                        __global uint32_t *out,
                                        unsigned elementsNum,
                                        unsigned rounds)
{
#define OperandSize 11
#define GmpOperandSize 12
  unsigned globalSize = get_global_size(0);
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {
    uint32_t op1[OperandSize];
    uint32_t modl[OperandSize];
    for (unsigned j = 0; j < OperandSize; ++j) {
      op1[j] = m1[i * GmpOperandSize + j];
      modl[j] = mod[i * GmpOperandSize + j];
    }

    uint32_t inv = invm[i];
    for (unsigned r = 0; r < rounds; ++r)
      monSqr352(op1, modl, inv);

    for (unsigned j = 0; j < OperandSize; ++j)
      out[i * OperandSize + j] = op1[j];
  }
#undef GmpOperandSize
#undef OperandSize
}

#define SMALL_PRIME_COUNT 4

inline uint mod_high_part(__global uint32_t *prime_base, uint prime) {
  ulong acc = 0;
  for (int idx = 9; idx >= 1; --idx)
    acc = ((acc << 32) + prime_base[idx]) % prime;
  return (uint)acc;
}

inline uint fast_mod_u32(uint value, uint prime, uint recip) {
  uint q = mul_hi(value, recip);
  uint r = value - q * prime;
  return (r >= prime) ? r - prime : r;
}

__kernel void fermatTest320(__global uint32_t *restrict numbers,
                            __global uint32_t *restrict out,
                            __global uint32_t *restrict prime_base,
                            unsigned elementsNum,
                            __global const uint *kSmallPrimes,
                            __global const uint *kPrimeReciprocals)
{
#define OperandSize 10  
  unsigned globalSize = get_global_size(0);
  const uint lid = get_local_id(0);

  __local uint4 sharedPrimeBase[3];
  if (lid < 3) {
    sharedPrimeBase[lid] = (uint4){prime_base[lid * 4 + 0],
                                   prime_base[lid * 4 + 1],
                                   prime_base[lid * 4 + 2],
                                   (lid == 2) ? 0 : prime_base[lid * 4 + 3]};
  }

  __local uint sharedHighResidues[SMALL_PRIME_COUNT];
  if (lid < SMALL_PRIME_COUNT)
    sharedHighResidues[lid] = mod_high_part(prime_base, kSmallPrimes[lid]);

  barrier(CLK_LOCAL_MEM_FENCE);

  uint4 lNumbersv[3] = {
    sharedPrimeBase[0],
    sharedPrimeBase[1],
    sharedPrimeBase[2]
  };

  const uint4 one  = {1,0,0,0};
  const uint4 zero = {0,0,0,0};

#pragma unroll
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {

    const uint offset = numbers[i];
    uint compositeFlag = 0;
    for (int primeIdx = 0; primeIdx < SMALL_PRIME_COUNT; ++primeIdx) {
      const uint prime = kSmallPrimes[primeIdx];
      const uint recip = kPrimeReciprocals[primeIdx];
      uint candidateMod = sharedHighResidues[primeIdx] + fast_mod_u32(offset, prime, recip);
      candidateMod -= (candidateMod >= prime) ? prime : 0;
      if (candidateMod == 0) {
        compositeFlag = 1;
        break;
      }
    }

    if (compositeFlag) {
      out[i] = 0;
      continue;
    }

    uint4 result[3];
    lNumbersv[0].x = offset;
    
    FermatTest320(lNumbersv, result);

    out[i] = 1;
    if (all(result[0] != one) || all(result[1] != zero) || all(result[2] != zero))
      out[i] = 0;
  }
#undef OperandSize
}

__kernel void fermatTestBenchMark320(__global uint32_t *restrict numbers,
                                     __global uint32_t *restrict out,
                                     unsigned elementsNum)
{
#define OperandSize 10  
  unsigned globalSize = get_global_size(0);
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {
    uint32_t lNumbers[OperandSize];
    for (unsigned j = 0; j < OperandSize; j++)
      lNumbers[j] = numbers[i*OperandSize+j];

    uint4 result[3];
    
    uint4 lNumbersv[3] = {
      (uint4){lNumbers[0], lNumbers[1], lNumbers[2], lNumbers[3]}, 
      (uint4){lNumbers[4], lNumbers[5], lNumbers[6], lNumbers[7]}, 
      (uint4){lNumbers[8], lNumbers[9], 0, 0}
    };    

    FermatTest320(lNumbersv, result);
      
    uint32_t *pResult = (uint32_t*)result;    
    for (unsigned j = 0; j < OperandSize; j++)
      out[i*OperandSize + j] = pResult[j];  
  }
#undef OperandSize
}


__kernel void fermatTestBenchMark352(__global uint32_t *restrict numbers,
                                     __global uint32_t *restrict out,
                                     unsigned elementsNum)
{
#define OperandSize 11  
  unsigned globalSize = get_global_size(0);
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {
    uint32_t lNumbers[OperandSize];
    for (unsigned j = 0; j < OperandSize; j++)
      lNumbers[j] = numbers[i*OperandSize+j];

    uint4 result[3];
    
    uint4 lNumbersv[3] = {
      (uint4){lNumbers[0], lNumbers[1], lNumbers[2], lNumbers[3]}, 
      (uint4){lNumbers[4], lNumbers[5], lNumbers[6], lNumbers[7]}, 
      (uint4){lNumbers[8], lNumbers[9], lNumbers[10], 0}
    };    

    FermatTest352(lNumbersv, result);
      
    uint32_t *pResult = (uint32_t*)result;    
    for (unsigned j = 0; j < OperandSize; j++)
      out[i*OperandSize + j] = pResult[j];  
  }
#undef OperandSize
}
