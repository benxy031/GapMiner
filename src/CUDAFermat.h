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
 public:
  using ResultWord = uint8_t;

  struct SievePrototypeParams {
    const uint8_t *sieve_bytes;
    uint32_t sieve_byte_len;
    uint64_t sieve_base;
    uint32_t window_size;
    uint32_t sieve_round;
    uint32_t min_gap;
    const uint32_t *prime_starts;
    uint32_t prime_count;
  };

 private:
  class CudaBuffer {
   public:
    CudaBuffer();
    ~CudaBuffer();
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;
    CudaBuffer(CudaBuffer &&) = delete;
    CudaBuffer &operator=(CudaBuffer &&) = delete;

    void init(size_t size,
          size_t element_size = sizeof(uint32_t),
          bool prefer_pinned = false);
    void copyToDevice(cudaStream_t stream, size_t count = 0, size_t offset = 0);
    void copyToHost(cudaStream_t stream, size_t count = 0, size_t offset = 0);
    void reset();

    size_t Size;
    size_t ElementSize;
    uint8_t *HostData;
    uint8_t *DeviceData;

   private:
    std::vector<uint8_t> storage_;
      bool pinned_host_;
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
  CudaBuffer sievePrototypeOutput;
  CudaBuffer sievePrototypeBitmap;
  CudaBuffer sievePrototypePrimes;
  CudaBuffer sievePrototypeResidues;
  CudaBuffer sievePrototypeWindowBits;
  CudaBuffer sievePrototypeWindowBases;
  CudaBuffer sievePrototypeWindowCounts;
  CudaBuffer sievePrototypeWindowDescs;
  bool sievePrimesConfigured;
  uint32_t configuredPrimeCount;

  std::vector<cudaStream_t> streams;
  uint32_t computeUnits;
  std::vector<uint32_t> smallPrimeResidues;
  std::vector<uint32_t> prototypeOffsets;
  std::vector<uint32_t> prototypeWindowOffsets;
  uint32_t lastPrototypeCount;
  uint32_t lastPrototypeWindowCount;

  GPUFermat(unsigned device_id, const char *platformId, unsigned workItems, unsigned streamCount = 2);
  ~GPUFermat();

  bool init_cuda(unsigned device_id, const char *platformId);
  void initializeBuffers();
  void run_cuda(uint32_t elementsNum);
  uint32_t rand32();
  void uploadPrimeBaseConstants();
  void ensurePrototypeOutputCapacity(size_t required);
  void ensurePrototypeBitmapCapacity(size_t required_words);
  void ensurePrototypeResidueCapacity(size_t required);
  void ensurePrototypeWindowBitCapacity(size_t required);
  void ensurePrototypeBaseCapacity(size_t required);
  void ensurePrototypeWindowCountCapacity(size_t required);
  void ensurePrototypeWindowDescCapacity(size_t required);
  void resetPrototypeWindowState();
  bool run_compact_scan(uint32_t window_count,
                        uint32_t total_bits,
                        cudaStream_t stream);
  void prototype_sieve_single(const SievePrototypeParams &params);
  void benchmark_montgomery_mul();

 public:
  static GPUFermat *get_instance(unsigned device_id = static_cast<unsigned>(-1),
                                 const char *platformId = nullptr,
                                 unsigned workItems = 0,
                                 unsigned streamCount = 2);
  static unsigned get_group_size();
  static void destroy_instance();

  ResultWord *get_results_buffer();
  uint32_t *get_prime_base_buffer();
  uint32_t *get_candidates_buffer();
  unsigned get_result_word_size();
  void configure_sieve_primes(const uint32_t *primes, size_t count);

  void fermat_gpu(uint32_t elementsNum);
  unsigned get_elements_num();
  unsigned get_block_size();
  void prototype_sieve(const SievePrototypeParams &params);
  void prototype_sieve_batch(const SievePrototypeParams *params,
                             uint32_t window_count);
  /* Diagnostic: dump device-built candidate limbs for given sample indices
   * Only meaningful for CUDA backend; no-op for other backends. */
  void dump_device_samples(const uint32_t *sample_indices, unsigned sampleCount);
  const uint32_t *prototype_offsets_data() const;
  uint32_t prototype_offsets_count() const;
  const uint32_t *prototype_window_offsets() const;
  uint32_t prototype_window_count() const;
  void benchmark();
  void test_gpu();
  unsigned get_stream_count() const;
};

#endif /* GAPMINER_CUDA_FERMAT_H */
#endif /* USE_CUDA_BACKEND */
#endif /* CPU_ONLY */
