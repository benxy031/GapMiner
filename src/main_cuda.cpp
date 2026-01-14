#if !defined(USE_CUDA_BACKEND)
#if defined(__has_include)
#if __has_include(<cuda_runtime_api.h>)
#define USE_CUDA_BACKEND
#else
#pragma message("CUDA runtime headers not found, building without CUDA backend.")
#endif
#else
#define USE_CUDA_BACKEND
#endif
#endif

#ifdef USE_CUDA_BACKEND
#include "main.cpp"
#else
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
	std::cerr << "CUDA backend requested but cuda_runtime_api.h not available." << std::endl;
	return EXIT_FAILURE;
}
#endif
