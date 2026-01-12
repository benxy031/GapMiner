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

__kernel void fermatTest320(__global uint32_t *restrict numbers,
                            __global uint32_t *restrict out,
                            __global uint32_t *restrict prime_base,
                            unsigned elementsNum)
{
#define OperandSize 10  
  unsigned globalSize = get_global_size(0);

  uint4 lNumbersv[3] = {
    (uint4){prime_base[0], prime_base[1], prime_base[2], prime_base[3]}, 
    (uint4){prime_base[4], prime_base[5], prime_base[6], prime_base[7]}, 
    (uint4){prime_base[8], prime_base[9], 0, 0}
  };    

  const uint4 one  = {1,0,0,0};
  const uint4 zero = {0,0,0,0};

#pragma unroll
  for (unsigned i = get_global_id(0); i < elementsNum; i += globalSize) {

    uint4 result[3];
    lNumbersv[0].x = numbers[i];
    
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
