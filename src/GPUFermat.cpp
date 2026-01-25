#if !defined(CPU_ONLY) && !defined(USE_CUDA_BACKEND)
#include </usr/include/CL/cl2.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <array>
#include <iterator>
#include <gmp.h>
#include <gmpxx.h>

#include "GPUFermat.h"
#include "utils.h"

using namespace std;

namespace {
constexpr const char *kKernelBinaryPath = "kernel.bin";
constexpr array<const char *, 3> kKernelSourceFiles = {"gpu/procs.cl",
                                                       "gpu/fermat.cl",
                                                       "gpu/benchmarks.cl"};
constexpr const char *kBuildOptions = "-cl-mad-enable -cl-fast-relaxed-math";
}

#define OCL(error)                                            \
	if(cl_int err = error) {                                    \
    pthread_mutex_lock(&io_mutex);                            \
    cout << get_time() << "OpenCL error: " << err;            \
    cout << " at " << __FILE__ << ":" << __LINE__ << endl;    \
    pthread_mutex_unlock(&io_mutex);                          \
		return;                                                   \
	}

#define OCLR(error, ret)                                      \
	if(cl_int err = error) {                                    \
    pthread_mutex_lock(&io_mutex);                            \
    cout << get_time() << "OpenCL error: " << err;            \
    cout << " at " << __FILE__ << ":" << __LINE__ << endl;    \
    pthread_mutex_unlock(&io_mutex);                          \
		return ret;                                               \
	}

#define OCLE(error)                                           \
	if(cl_int err = error) {                                    \
    pthread_mutex_lock(&io_mutex);                            \
    cout << get_time() << "OpenCL error: " << err;            \
    cout << " at " << __FILE__ << ":" << __LINE__ << endl;    \
    pthread_mutex_unlock(&io_mutex);                          \
		exit(err);                                                \
	}

/* synchronization mutexes */
pthread_mutex_t GPUFermat::creation_mutex = PTHREAD_MUTEX_INITIALIZER;

/* this will be a singleton */
GPUFermat *GPUFermat::only_instance = NULL;

/* indicates if this was initialized */
bool GPUFermat::initialized = false;

/* the opencl context */
static cl_context gContext;

/* the GPU work group size */
unsigned GPUFermat::GroupSize = 256;

/* the array size of uint32_t for the numbers to test */
unsigned GPUFermat::operandSize = 320/32;

unsigned GPUFermat::get_group_size() {
  return GroupSize;
}

unsigned GPUFermat::get_block_size() {
  return GroupSize;
}

GPUFermat::~GPUFermat() {
  if (mFermatBenchmarkKernel320) {
    clReleaseKernel(mFermatBenchmarkKernel320);
    mFermatBenchmarkKernel320 = nullptr;
  }
  if (mFermatKernel320) {
    clReleaseKernel(mFermatKernel320);
    mFermatKernel320 = nullptr;
  }
  if (queue) {
    clReleaseCommandQueue(queue);
    queue = nullptr;
  }
  if (gProgram) {
    clReleaseProgram(gProgram);
    gProgram = nullptr;
  }
  if (gContext) {
    clReleaseContext(gContext);
    gContext = nullptr;
  }
}

  unsigned GPUFermat::get_result_word_size() {
    return static_cast<unsigned>(sizeof(ResultWord));
  }

/* return the only instance of this */
GPUFermat *GPUFermat::get_instance(unsigned device_id, 
                                   const char *platformId,
                                   unsigned workItems) {

  log_str("providing the only instance", LOG_D);
  pthread_mutex_lock(&creation_mutex);
  if (!initialized && 
      device_id != (unsigned)(-1) && 
      platformId != NULL &&
      workItems != 0) {

    only_instance = new GPUFermat(device_id, platformId, workItems);
    initialized   = true;
  }
  pthread_mutex_unlock(&creation_mutex);

  return only_instance;
}

void GPUFermat::destroy_instance() {
  pthread_mutex_lock(&creation_mutex);
  delete only_instance;
  only_instance = NULL;
  initialized = false;
  pthread_mutex_unlock(&creation_mutex);
}

/* initialize this */
// Optimized constructor for GPUFermat class
GPUFermat::GPUFermat(unsigned device_id, const char *platformId, unsigned workItems) {
  log_str("Creating GPUFermat", LOG_D);
  gProgram = nullptr;
  mFermatBenchmarkKernel320 = nullptr;
  mFermatKernel320 = nullptr;
  gpu = nullptr;
  computeUnits = 0;
  queue = nullptr;
  this->workItems = workItems;
  if (!init_cl(device_id, platformId)) {
    throw std::runtime_error("GPUFermat OpenCL initialization failed");
  }

  elementsNum = GroupSize * workItems;
  numberLimbsNum = elementsNum * operandSize;
  groupsNum = computeUnits * 4;

  initializeBuffers(); // Function to handle buffer initialization
}

// Define the function to initialize buffers
void GPUFermat::initializeBuffers() {
  numbers.init(elementsNum, CL_MEM_READ_WRITE);
  gpuResults.init(numberLimbsNum, CL_MEM_READ_WRITE);
  primeBase.init(operandSize, CL_MEM_READ_WRITE);
  // Initialize small primes and reciprocals used by the OpenCL kernels
  const unsigned HOST_SMALL_PRIME_COUNT = 4;
  smallPrimes.init(HOST_SMALL_PRIME_COUNT, CL_MEM_READ_WRITE);
  primeReciprocals.init(HOST_SMALL_PRIME_COUNT, CL_MEM_READ_WRITE);

  // default small primes (skip 2)
  uint32_t defaultPrimes[HOST_SMALL_PRIME_COUNT] = {3u, 5u, 7u, 11u};
  for (unsigned i = 0; i < HOST_SMALL_PRIME_COUNT; ++i)
    smallPrimes.HostData[i] = defaultPrimes[i];

  // recip = floor(2^32 / p) + 1
  for (unsigned i = 0; i < HOST_SMALL_PRIME_COUNT; ++i) {
    uint32_t p = smallPrimes.HostData[i];
    primeReciprocals.HostData[i] = static_cast<uint32_t>(((uint64_t(1) << 32) / p) + 1ull);
  }

  // copy small-prime buffers to device
  smallPrimes.copyToDevice(queue);
  primeReciprocals.copyToDevice(queue);
}

GPUFermat::ResultWord *GPUFermat::get_results_buffer() {
  return gpuResults.HostData;
}

uint32_t *GPUFermat::get_prime_base_buffer() {
  return primeBase.HostData;
}

uint32_t *GPUFermat::get_candidates_buffer() {
  return numbers.HostData;
}

bool GPUFermat::init_cl(unsigned device_id, const char *platformId) {

  log_str("init opencl", LOG_D);
  const char *platformName = "";

  if (strcmp(platformId, "amd") == 0)
    platformName = "AMD Accelerated Parallel Processing";
  else if (strcmp(platformId, "nvidia") == 0)
    platformName = "NVIDIA CUDA";
  else {
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "ERROR: platform " << platformId << " not supported ";
    cout << " use amd or nvidia" << endl;
    pthread_mutex_unlock(&io_mutex);                       
    exit(EXIT_FAILURE);
  }
  
  cl_platform_id platforms[10];
  cl_uint numplatforms;
  OCLR(clGetPlatformIDs(10, platforms, &numplatforms), false);
  if(!numplatforms){
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "ERROR: no OpenCL platform found" << endl;
    pthread_mutex_unlock(&io_mutex);                       
    return false;
  }
  
  int iplatform = -1;
  for(unsigned i = 0; i < numplatforms; ++i){
    char name[1024] = {0};
    OCLR(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, 0), false);
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "Found platform[" << i << "] name = " << name << endl;
    pthread_mutex_unlock(&io_mutex);                       
    if(!strcmp(name, platformName)){
      iplatform = i;
      break;
    }
  }
  
  if(iplatform < 0){
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "ERROR: " << platformName << " not found" << endl;
    pthread_mutex_unlock(&io_mutex);                       
    return false;
  }

  
  unsigned mNumDevices;
  cl_platform_id platform = platforms[iplatform];
  
  cl_device_id devices[10];
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 10, devices, &mNumDevices);
  pthread_mutex_lock(&io_mutex);                            
  cout << get_time() << "Found " << mNumDevices << " device(s)" << endl;
  pthread_mutex_unlock(&io_mutex);                       
  
  if(!mNumDevices){
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "ERROR: no OpenCL GPU devices found" << endl;
    pthread_mutex_unlock(&io_mutex);                       
    return false;
  }


  if (mNumDevices <= device_id) {
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "ERROR " << mNumDevices << " device(s) detected ";
    cout << device_id << " device requested for use" << endl;
    pthread_mutex_unlock(&io_mutex);                       
    return false;
  }
  gpu = devices[device_id];

  {
    cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_int error;
    gContext = clCreateContext(props, 1, &gpu, 0, 0, &error);
    OCLR(error, false);
  }

  std::ifstream testfile(kKernelBinaryPath);
  if(!testfile){
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "Compiling ..." << endl;
    pthread_mutex_unlock(&io_mutex);                       

    std::string sourcefile;
    if (!loadKernelSources(sourcefile)) {
      return false;
    }
    if (!buildKernelBinary(sourcefile)) {
      return false;
    }
  }

  std::ifstream bfile(kKernelBinaryPath, std::ifstream::binary);
  if(!bfile){
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "ERROR: kernel.bin not found" << endl;
    pthread_mutex_unlock(&io_mutex);                       
    return false;
  }
  
  bfile.seekg(0, bfile.end);
  size_t binsize = static_cast<size_t>(bfile.tellg());
  bfile.seekg(0, bfile.beg);
  if(!binsize){
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "ERROR: kernel.bin empty" << endl;
    pthread_mutex_unlock(&io_mutex);                       
    return false;
  }

  std::vector<unsigned char> binary(binsize);
  bfile.read(reinterpret_cast<char*>(binary.data()), binsize);
  bfile.close();
  pthread_mutex_lock(&io_mutex);                            
  cout << get_time() << "Loaded kernel binary size = " << binsize << " bytes" << endl;
  pthread_mutex_unlock(&io_mutex);                       
  
  std::vector<size_t> binsizes(1, binsize);
  std::vector<cl_int> binstatus(1);
  std::vector<const unsigned char*> binaries(1, binary.data());
  cl_int error;
  gProgram = clCreateProgramWithBinary(gContext, 1, &gpu, &binsizes[0], &binaries[0], &binstatus[0], &error);
  OCLR(error, false);
  OCLR(clBuildProgram(gProgram, 1, &gpu, 0, 0, 0), false);

  /** adl support needs to be tested
  init_adl(mNumDevices);
  
  for(unsigned i = 0; i < mNumDevices; ++i){
    
    if(mCoreFreq[i] > 0)
      if(set_engineclock(i, mCoreFreq[i]))
        printf("set_engineclock(%d, %d) failed.\n", i, mCoreFreq[i]);
    if(mMemFreq[i] > 0)
      if(set_memoryclock(i, mMemFreq[i]))
        printf("set_memoryclock(%d, %d) failed.\n", i, mMemFreq[i]);
    if(mPowertune[i] >= -20 && mPowertune[i] <= 20)
      if(set_powertune(i, mPowertune[i]))
        printf("set_powertune(%d, %d) failed.\n", i, mPowertune[i]);
    if (mFanSpeed[i] > 0)
      if(set_fanspeed(i, mFanSpeed[i]))
        printf("set_fanspeed(%d, %d) failed.\n", i, mFanSpeed[i]);
  }
  */
  
  mFermatBenchmarkKernel320 = clCreateKernel(gProgram, "fermatTestBenchMark320", &error);  
  OCLR(error, false);

  mFermatKernel320 = clCreateKernel(gProgram, "fermatTest320", &error);  
  OCLR(error, false);
  
  char deviceName[128] = {0};

  clGetDeviceInfo(gpu, CL_DEVICE_NAME, sizeof(deviceName), deviceName, 0);
  clGetDeviceInfo(gpu, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, 0);
  pthread_mutex_lock(&io_mutex);                            
  cout << get_time() << "Using GPU " << device_id << " [" << deviceName;
  cout << "]: which has " << computeUnits << " CUs" << endl;
  pthread_mutex_unlock(&io_mutex);                       


  clGetDeviceInfo(gpu, CL_DEVICE_NAME, sizeof(deviceName), deviceName, 0);
  clGetDeviceInfo(gpu, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, 0);

  queue = clCreateCommandQueue(gContext, gpu, 0, &error);
  if (!queue || error != CL_SUCCESS) {
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "Error: can't create command queue" << endl;
    pthread_mutex_unlock(&io_mutex);                       
    return false;
  }

  return true;
}

bool GPUFermat::loadKernelSources(std::string &sourcefile) {
  sourcefile.clear();
  for (const auto &path : kKernelSourceFiles) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
      pthread_mutex_lock(&io_mutex);
      cout << get_time() << "ERROR: " << path << " cannot be opened" << endl;
      pthread_mutex_unlock(&io_mutex);
      return false;
    }
    sourcefile.append(std::istreambuf_iterator<char>(stream),
                      std::istreambuf_iterator<char>());
  }

  if (sourcefile.empty()) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "ERROR: kernel sources are empty" << endl;
    pthread_mutex_unlock(&io_mutex);
    return false;
  }

  pthread_mutex_lock(&io_mutex);
  cout << get_time() << "Source: " << static_cast<unsigned>(sourcefile.size())
       << " bytes" << endl;
  pthread_mutex_unlock(&io_mutex);
  return true;
}

bool GPUFermat::buildKernelBinary(const std::string &sourcefile) {
  cl_int error;
  const char *sources[] = { sourcefile.c_str(), nullptr };
  gProgram = clCreateProgramWithSource(gContext, 1, sources, NULL, &error);
  OCLR(error, false);

  cl_int buildStatus = clBuildProgram(gProgram, 1, &gpu, kBuildOptions, 0, 0);
  if (buildStatus != CL_SUCCESS) {
    size_t logSize = 0;
    clGetProgramBuildInfo(gProgram, gpu, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::vector<char> buildLog(logSize + 1, 0);
    if (logSize) {
      clGetProgramBuildInfo(gProgram, gpu, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), 0);
    }
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "ERROR: OpenCL build failed\n" << buildLog.data() << endl;
    pthread_mutex_unlock(&io_mutex);
    clReleaseProgram(gProgram);
    gProgram = nullptr;
    return false;
  }

  size_t binsizes[1] = {0};
  OCLR(clGetProgramInfo(gProgram, CL_PROGRAM_BINARY_SIZES, sizeof(binsizes), binsizes, 0), false);
  size_t binsize = binsizes[0];
  if (!binsize) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "No binary available!" << endl;
    pthread_mutex_unlock(&io_mutex);
    clReleaseProgram(gProgram);
    gProgram = nullptr;
    return false;
  }

  pthread_mutex_lock(&io_mutex);
  cout << get_time() << "Compiled kernel binary size = " << binsize << " bytes" << endl;
  pthread_mutex_unlock(&io_mutex);

  std::vector<unsigned char> binary(binsize);
  unsigned char *binaries[] = { binary.data() };
  OCLR(clGetProgramInfo(gProgram, CL_PROGRAM_BINARIES, sizeof(binaries), binaries, 0), false);

  std::ofstream bin(kKernelBinaryPath, std::ofstream::binary | std::ofstream::trunc);
  if (!bin) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "ERROR: cannot write " << kKernelBinaryPath << endl;
    pthread_mutex_unlock(&io_mutex);
    clReleaseProgram(gProgram);
    gProgram = nullptr;
    return false;
  }
  bin.write(reinterpret_cast<const char*>(binary.data()), binsize);
  bin.close();

  OCLR(clReleaseProgram(gProgram), false);
  gProgram = nullptr;
  return true;
}

/* run a benchmark test */
void GPUFermat::benchmark() {

  test_gpu();
  return;

  for (int i = 1; i <= 10; i++) {
    
    clBuffer numbers;

    unsigned elementsNum    = 131072 * i;
    unsigned operandSize    = 320/32;
    unsigned numberLimbsNum = elementsNum*operandSize;
    numbers.init(numberLimbsNum, CL_MEM_READ_WRITE);

    for (unsigned j = 0; j < elementsNum; j++) {
      for (unsigned k = 0; k < operandSize; k++)
        numbers[j*operandSize + k] = (k == operandSize-1) ? (1 << (j % 32)) : rand32();
      numbers[j*operandSize] |= 0x1; 
    }


    fermatTestBenchmark(queue, mFermatBenchmarkKernel320, numbers, elementsNum);
  }
}

/* generates a 32 bit random number */
uint32_t GPUFermat::rand32() {
  uint32_t result = rand();
  result = (result << 16) | rand();
  return result;
}

/* clBuffer constructor */
GPUFermat::clBuffer::clBuffer() {
  
  Size = 0;
  HostData = 0;
  DeviceData = 0;
}

/* clBuffer destructor */
GPUFermat::clBuffer::~clBuffer() {
  
  if(HostData)
    delete [] HostData;
  
  if(DeviceData)
    clReleaseMemObject(DeviceData);
}

/* inits a clBuffer */
void GPUFermat::clBuffer::init(int size, cl_mem_flags flags) {
  
  Size = size;
  
  if(!(flags & CL_MEM_HOST_NO_ACCESS)){
    HostData = new uint32_t[Size];
    memset(HostData, 0, Size*sizeof(uint32_t));
  }else
    HostData = 0;
  
  cl_int error;
  DeviceData = clCreateBuffer(gContext, flags, Size*sizeof(uint32_t), 0, &error);
  OCL(error);
}

/* copy the clBuffer content to gpu */
void GPUFermat::clBuffer::copyToDevice(cl_command_queue cq, bool blocking) {
  
  OCL(clEnqueueWriteBuffer(cq, DeviceData, blocking, 0, Size*sizeof(uint32_t), HostData, 0, 0, 0));
}

/* copy the clBuffer content to host */
void GPUFermat::clBuffer::copyToHost(cl_command_queue cq, bool blocking, unsigned size) {
  
  if(size == 0)
    size = Size;
  
  OCL(clEnqueueReadBuffer(cq, DeviceData, blocking, 0, size*sizeof(uint32_t), HostData, 0, 0, 0));
}

/* access the host data of a clBuffer */
uint32_t &GPUFermat::clBuffer::get(int index) {
  return HostData[index];
}

/* access the host data of a clBuffer */
uint32_t &GPUFermat::clBuffer::operator[](int index) {
  return HostData[index];
}

/* public interface to the gpu Fermat test */
void GPUFermat::fermat_gpu(uint32_t batchElements) {
  const uint32_t work = (batchElements == 0) ? elementsNum : batchElements;
  run_fermat(queue, mFermatKernel320, numbers, gpuResults, work);
  gpuResults.copyToHost(queue);
  clFinish(queue);
}

/* Diagnostic no-op for non-CUDA backend */
void GPUFermat::dump_device_samples(const uint32_t *sample_indices, unsigned sampleCount) {
  (void)sample_indices;
  (void)sampleCount;
  return;
}

unsigned GPUFermat::get_elements_num() {
  return elementsNum;
}

/* run the Fermat test on the gpu */
void GPUFermat::run_fermat(cl_command_queue queue,
                           cl_kernel kernel,
                           clBuffer &numbers,
                           clBuffer &gpuResults,
                           unsigned elementsNum) {

                            
  log_str("running " + std::to_string(elementsNum) + " fermat tests on the gpu", LOG_D);
  numbers.copyToDevice(queue);
  gpuResults.copyToDevice(queue);
  primeBase.copyToDevice(queue);

  // ensure small-prime buffers are on device
  smallPrimes.copyToDevice(queue);
  primeReciprocals.copyToDevice(queue);

  OCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &numbers.DeviceData));
  OCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpuResults.DeviceData));
  OCL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &primeBase.DeviceData));
  OCL(clSetKernelArg(kernel, 3, sizeof(elementsNum), &elementsNum));
  OCL(clSetKernelArg(kernel, 4, sizeof(cl_mem), &smallPrimes.DeviceData));
  OCL(clSetKernelArg(kernel, 5, sizeof(cl_mem), &primeReciprocals.DeviceData));
  
  size_t globalThreads[1] = { groupsNum*GroupSize };
  size_t localThreads[1] = { GroupSize };
  cl_event event = nullptr;
  cl_int result = clEnqueueNDRangeKernel(queue,
                                         kernel,
                                         1,
                                         0,
                                         globalThreads,
                                         localThreads,
                                         0, 0, &event);
  if (result != CL_SUCCESS) {
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "clEnqueueNDRangeKernel error: " << result << endl;
    pthread_mutex_unlock(&io_mutex);                       
    return;
  }
    
  cl_int waitResult = clWaitForEvents(1, &event);
  if (waitResult != CL_SUCCESS) {
    pthread_mutex_lock(&io_mutex);                            
    cout << get_time() << "clWaitForEvents error " << waitResult << "!" << endl;
    pthread_mutex_unlock(&io_mutex);                       
    clReleaseEvent(event);
    return;
  }
  clReleaseEvent(event);
  clFlush(queue);
}

/* test the gpu results */
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
    uint32_t *primes  = numbers.HostData;
    uint32_t *results = gpuResults.HostData;

    mpz_class mpz(rand32());

    /* init with random number */
    for (int i = 0; i < 8; i++) {
      mpz <<= 32;
      mpz += rand32();
    }

    /* make sure mpz is not a prime */
    if (mpz.get_ui() & 0x1)
      mpz += 1;

    memset(prime_base, 0, operandSize * sizeof(uint32_t));
    size_t exported_size = 0;
    mpz_export(prime_base, &exported_size, -1, sizeof(uint32_t), 0, 0, mpz.get_mpz_t());
  
    /* create the test numbers, every second will be prime */
    for (unsigned i = 0; i < size; i++) {

      primes[i] = mpz.get_ui() & 0xffffffff;

      if (i % 2 == 0)
        mpz_nextprime(mpz.get_mpz_t(), mpz.get_mpz_t());
      else
        mpz_add_ui(mpz.get_mpz_t(), mpz.get_mpz_t(), 1);

      if (i % 23 == 0)
        printf("\rcreating test data: %d  \r", size - i);
    }
    printf("\r                                             \r");

    /* run the gpu test */
    fermat_gpu(size);

    /* check the results */
    unsigned failures = 0;
    for (unsigned i = 0; i < size; i++) {

      const uint32_t expected = (i % 2 == 0) ? 0u : 1u;

      if (results[i] != expected) {
        if (failures == 0) {
          printf("Result %u is wrong: expected %u but got %u\n", i, expected, results[i]);
        }
        failures++;
      }
   }

   if (failures) {
     printf("GPU Test failed: %u mismatched result(s)\n", failures);
   } else {
     printf("GPU Test worked\n");
   }
}

/* run the Fermat test on the gpu */
void GPUFermat::run_fermat_benchmark(cl_command_queue queue,
                                     cl_kernel kernel,
                                     clBuffer &numbers,
                                     clBuffer &gpuResults,
                                     unsigned elementsNum) {

  numbers.copyToDevice(queue);
  gpuResults.copyToDevice(queue);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &numbers.DeviceData);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpuResults.DeviceData);
  clSetKernelArg(kernel, 2, sizeof(elementsNum), &elementsNum);
  
  clFinish(queue);
  {
    size_t globalThreads[1] = { groupsNum*GroupSize };
    size_t localThreads[1] = { GroupSize };
    cl_event event;
    cl_int result;
    if ((result = clEnqueueNDRangeKernel(queue,
                                         kernel,
                                         1,
                                         0,
                                         globalThreads,
                                         localThreads,
                                         0, 0, &event)) != CL_SUCCESS) {
      pthread_mutex_lock(&io_mutex);                            
      cout << get_time() << "clEnqueueNDRangeKernel error!" << endl;
      pthread_mutex_unlock(&io_mutex);                       
      return;
    }
      
    cl_int error;
    if ((error = clWaitForEvents(1, &event)) != CL_SUCCESS) {
      pthread_mutex_lock(&io_mutex);                            
      cout << get_time() << "clWaitForEvents error " << error << "!" << endl;
      pthread_mutex_unlock(&io_mutex);                       
      return;
    }
      
    clReleaseEvent(event);
  }
}

/* run a benchmark and print results */
void GPUFermat::fermatTestBenchmark(cl_command_queue queue,
                                    cl_kernel kernel,
                                    clBuffer &numbers,
                                    unsigned elementsNum) { 

  unsigned numberLimbsNum = elementsNum*operandSize;
  
  clBuffer gpuResults;
  clBuffer cpuResults;
  
  gpuResults.init(numberLimbsNum, CL_MEM_READ_WRITE);
  cpuResults.init(numberLimbsNum, CL_MEM_READ_WRITE);
  
  
  std::unique_ptr<mpz_t[]> cpuNumbersBuffer(new mpz_t[elementsNum]);
  std::unique_ptr<mpz_t[]> cpuResultsBuffer(new mpz_t[elementsNum]);
  mpz_class mpzTwo = 2;
  mpz_class mpzE;
  mpz_import(mpzE.get_mpz_t(), operandSize, -1, 4, 0, 0, &numbers[0]);
  for (unsigned i = 0; i < elementsNum; i++) {
    mpz_init(cpuNumbersBuffer[i]);
    mpz_init(cpuResultsBuffer[i]);
    mpz_import(cpuNumbersBuffer[i], operandSize, -1, 4, 0, 0, &numbers[i*operandSize]);
    mpz_import(cpuResultsBuffer[i], operandSize, -1, 4, 0, 0, &cpuResults[i*operandSize]);
  }
  
  auto gpuBegin = std::chrono::steady_clock::now();  
  run_fermat_benchmark(queue, kernel, numbers, gpuResults, elementsNum);
  auto gpuEnd = std::chrono::steady_clock::now();  
  
  
  auto cpuBegin = std::chrono::steady_clock::now();  
  for (unsigned i = 0; i < elementsNum; i++) {
    mpz_sub_ui(mpzE.get_mpz_t(), cpuNumbersBuffer[i], 1);
    mpz_powm(cpuResultsBuffer[i], mpzTwo.get_mpz_t(), mpzE.get_mpz_t(), cpuNumbersBuffer[i]);
  }
  auto cpuEnd = std::chrono::steady_clock::now();  

  gpuResults.copyToHost(queue);
  clFinish(queue);
  
  memset(&cpuResults[0], 0, 4*operandSize*elementsNum);
  for (unsigned i = 0; i < elementsNum; i++) {
    size_t exportedLimbs;
    mpz_export(&cpuResults[i*operandSize], &exportedLimbs, -1, 4, 0, 0, cpuResultsBuffer[i]);
    if (memcmp(&gpuResults[i*operandSize], &cpuResults[i*operandSize], 4*operandSize) != 0) {
      fprintf(stderr, "element index: %u\n", i);
      fprintf(stderr, "gmp: ");
      for (unsigned j = 0; j < operandSize; j++)
        fprintf(stderr, "%08X ", cpuResults[i*operandSize + j]);
      fprintf(stderr, "\ngpu: ");
      for (unsigned j = 0; j < operandSize; j++)
        fprintf(stderr, "%08X ", gpuResults[i*operandSize + j]);
      fprintf(stderr, "\n");
      fprintf(stderr, "results differ!\n");
      break;
    }
  }
  
  double gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(gpuEnd-gpuBegin).count() / 1000.0;  
  double cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd-cpuBegin).count() / 1000.0;  
  double opsNum = ((elementsNum) / 1000000.0) / gpuTime * 1000.0;
  
  std::cout << std::endl << "Running benchmark with " << elementsNum / GroupSize << " work items:" << std::endl;
  std::cout << "  GPU with 320 bits: " << gpuTime << "ms (" << opsNum << "fM ops/sec)" << std::endl;

  opsNum = ((elementsNum) / 1000000.0) / cpuTime * 1000.0;
  std::cout << "  CPU with 320 bits: " << cpuTime << "ms (" << opsNum << "fM ops/sec)" << std::endl;
  std::cout << "  GPU is " <<  ((double) cpuTime) / ((double) gpuTime) << "times faster" << std::endl;
}
#endif /* !CPU_ONLY && !USE_CUDA_BACKEND */
