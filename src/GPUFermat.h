#ifndef CPU_ONLY
#ifndef __GPU_FERMAT_H__
#define __GPU_FERMAT_H__
#include </usr/include/CL/cl2.hpp>
#include <vector>
#include <string>
#include <gmp.h>
#include <gmpxx.h>

/**
 * A singleton class which is used to run fermat tests on one GPU
 */
class GPUFermat {

  private:

    /* a buffer to synchronize gpu and host memory */
    class clBuffer {
      
      public:

        /* size of this buffer */
        int Size;

        /* host data */
        uint32_t* HostData;

        /* gpu data */
        cl_mem DeviceData;
        
        clBuffer();
        ~clBuffer();
        
        /* initialize this */
        void init(int size, cl_mem_flags flags = 0);
        
        /* copy the clBuffer content to gpu */
        void copyToDevice(cl_command_queue cq, bool blocking = true);
        
        /* copy the clBuffer content to host */
        void copyToHost(cl_command_queue cq, bool blocking = true, unsigned size = 0);
        
        /* access the host data of a clBuffer */
        uint32_t &get(int index);
        
        /* access the host data of a clBuffer */
        uint32_t &operator[](int index);
    };


    /* synchronization mutexes */
    static pthread_mutex_t creation_mutex;

    /* the only instance of this */
    static GPUFermat *only_instance;

    /* indicates if this was initialized */
    static bool initialized;

    /* the GPU work group size */
    static unsigned GroupSize;

    /* the array size of uint32_t for the numbers to test */
    static unsigned operandSize;

    /* the work items for the gpu */
    unsigned workItems;


    /* the number of prime candidates to test at once */
    unsigned elementsNum;

    /* total number of array elements used fr storing the testing numbers */
    unsigned numberLimbsNum;

    /* number of work groups for the GPU */
    unsigned groupsNum;

    /* the numbers to test for primality */
    clBuffer numbers;

    /* the gpu fermat results */
    clBuffer gpuResults;
    
    /* the prime base */
    clBuffer primeBase;

    /* the opencl program */
    cl_program gProgram;

    /* the fermat kernel to run on the gpu */
    cl_kernel mFermatBenchmarkKernel320;  

    /* the fermat kernel to run on the gpu */
    cl_kernel mFermatKernel320;  

    /* cl device to run the test on */
    cl_device_id gpu;

    /* computing units of the used cpu */
    cl_uint computeUnits;

    /* command queue for the GPU */
    cl_command_queue queue;

    /* this is a singleton */
    GPUFermat(unsigned device_id, const char *platformId, unsigned workItems);


    /* initialize opencl for the given device */
    bool init_cl(unsigned device_id = 0, 
                 const char *platformId = "amd");


    void benchmark2();

    uint32_t rand32();

    /* run the Fermat test on the gpu */
    void run_fermat(cl_command_queue queue,
                    cl_kernel kernel,
                    clBuffer &numbers,
                    clBuffer &gpuResults,
                    unsigned elementsNum);

    /* run the Fermat test on the gpu */
    void run_fermat_benchmark(cl_command_queue queue,
                              cl_kernel kernel,
                              clBuffer &numbers,
                              clBuffer &gpuResults,
                              unsigned elementsNum);

    /* run a benchmark and print results */
    void fermatTestBenchmark(cl_command_queue queue,
                             cl_kernel kernel,
                             clBuffer &numbers,
                             unsigned elementsNum);

                             void initializeBuffers(); // Function prototype                 

  public :

    /* returns a pointer to the gpu results (host) buffer */
    uint32_t *get_results_buffer();

    /* returns a pointer to the prime_base (host) buffer */
    uint32_t *get_prime_base_buffer();

    /* returns a pointer to the candidates (host) buffer */
    uint32_t *get_candidates_buffer();

    /* return the only instance of this */
    static GPUFermat *get_instance(unsigned device_id = (unsigned)(-1), 
                                   const char *platformId = NULL,
                                   unsigned workItems = 0);

    /* public interface to the gpu Fermat test */
    void fermat_gpu();

    /* run a benchmark test */
    void benchmark();

    /* test the gpu results */
    void test_gpu();

};
#endif /* __GPU_FERMAT_H__ */
#endif /* CPU_ONLY */
