#ifndef CPU_ONLY
#ifdef USE_CUDA_BACKEND
#ifndef GAPMINER_CUDA_FERMAT_H
#define GAPMINER_CUDA_FERMAT_H

#include <cuda_runtime_api.h>
#include <pthread.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gmp.h>
#include <gmpxx.h>

class GPUFermat {
 private:
  class CudaBuffer {
   public:
    CudaBuffer();
    ~CudaBuffer();
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;
    CudaBuffer(CudaBuffer &&) = delete;
    CudaBuffer &operator=(CudaBuffer &&) = delete;

    void init(size_t size);
    void copyToDevice(cudaStream_t stream, size_t count = 0, size_t offset = 0);
    void copyToHost(cudaStream_t stream, size_t count = 0, size_t offset = 0);
    void reset();

    size_t Size;
    uint32_t *HostData;
    uint32_t *DeviceData;

   private:
    std::vector<uint32_t> storage_;
  };

  static pthread_mutex_t creation_mutex;
  static GPUFermat *only_instance;
  static bool initialized;
  static unsigned GroupSize;
  static unsigned operandSize;

  unsigned workItems;
  unsigned elementsNum;
  unsigned numberLimbsNum;
  unsigned groupsNum;

  CudaBuffer numbers;
  CudaBuffer gpuResults;
  CudaBuffer primeBase;

  std::array<cudaStream_t, 2> streams;
  uint32_t computeUnits;
  std::array<uint32_t, 4> smallPrimeResidues;

  GPUFermat(unsigned device_id, const char *platformId, unsigned workItems);
  ~GPUFermat();

  bool init_cuda(unsigned device_id, const char *platformId);
  void initializeBuffers();
  void run_cuda(uint32_t elementsNum);
  uint32_t rand32();
  void uploadPrimeBaseConstants();

 public:
  static GPUFermat *get_instance(unsigned device_id = static_cast<unsigned>(-1),
                                 const char *platformId = nullptr,
                                 unsigned workItems = 0);
  static unsigned get_group_size();
  static void destroy_instance();

  uint32_t *get_results_buffer();
  uint32_t *get_prime_base_buffer();
  uint32_t *get_candidates_buffer();

  void fermat_gpu(uint32_t elementsNum);
  void benchmark();
  void test_gpu();
};

#endif /* GAPMINER_CUDA_FERMAT_H */
#endif /* USE_CUDA_BACKEND */
#endif /* CPU_ONLY */
