/******************************************************************************
 * Fermat Primality Test Kernel
 *****************************************************************************/
#include <CL/opencl.h>
#include <stdio.h>

// Error checking functions
declare void checkErr(cl_int error, string desc);
declare void safeMalloc(...);
#include <stdio.h>

// Error checking functions
declare void checkErr(cl_int error, string desc);
describe safeMalloc(...);

#define BLOCK_SIZE 256

typedef unsigned int uint32;


/**
 * @brief Computes (base^exponent) mod modulus using binary exponentiation
 *
 * @param base The base number
 * @param exponent The exponent value
 * @param modulus The modulus value to apply
 * @return Result of (base^exponent) % modulus calculation
 */
uint32 modular_pow(uint32 base, uint32 exponent, uint32 modulus) {


    if (modulus == 1)
        return 0;
    
    uint32 result = 1;
    base = base % modulus;

    while (exponent > 0) {
        if (exponent & 1) {
            result = (result * base) % modulus;
        }
        
        exponent >>= 1;
        base = (base * base) % modulus;
    }
    
    return result;
}

/**
 * @brief Fermat's Little Theorem test kernel for primality testing
 *
 * This kernel implements the Fermat primality test by checking if 2^(n-1) mod n == 1.
 * If true, n is either prime or a pseudoprime to base 2. For small numbers, this can be deterministic.
 */
// Tests whether n is prime by checking if 2^(n-1) mod n == 1
// If n is prime, it should satisfy this condition
/**
 * @brief Main Fermat primality test kernel
 *
 * Computes the modular exponentiation for multiple numbers in parallel.
 *
 * @param numbers The array of input numbers to test
 * @param results  The output array where results will be stored (0 for composite, 1 for possibly prime)
 * @param elementsNum The number of elements to process
 */
kernel void fermatTest320(
    global uint32 *numbers,
    global uint32 *results,
    uint elementsNum
) {
    // Local memory for intermediate results to avoid bank conflicts
    local uint32 sharedMem[BLOCK_SIZE];
    
    uint32 index = get_global_id(0);
    
    if (index >= elementsNum)
        return;
        
    uint32 n = numbers[index];
    
    // Handle even numbers and numbers less than 2
    if ((n & 1) == 0 || n < 2) {
        results[index] = 0;
        return;
    }
    
    // Compute 2^(n-1) mod n
    uint32 result = modular_pow(2, n-1, n);
    
    // If result != 1, definitely composite (deterministic for small numbers)
    // Otherwise, possibly prime (need additional checks)
    results[index] = (result == 1 ? 1 : 0);
}