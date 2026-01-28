// Focused CRT micro-benchmark harness
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include "ChineseRemainder.h"
#include "utils.h"

int main() {
  std::mt19937 rng(123456789);
  const int trials = 30;
  int sizes[] = {50, 100, 200};

  for (int s = 0; s < 3; ++s) {
    int n = sizes[s];
    double total_ms = 0.0;
    for (int t = 0; t < trials; ++t) {
      std::vector<sieve_t> nums(n);
      std::vector<sieve_t> rems(n);
      for (int i = 0; i < n; ++i) {
        nums[i] = first_primes[i];
        rems[i] = (sieve_t)(rng() % nums[i]);
      }

      auto start = std::chrono::high_resolution_clock::now();
      ChineseRemainder cr(nums.data(), rems.data(), (sieve_t) n);
      auto end = std::chrono::high_resolution_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    std::cout << "CRT bench size=" << n << " avg_ms=" << (total_ms / trials) << "\n";
  }

  return 0;
}
