/*
 * Simple tool to compute PoW difficulty for given prime_base, target and offsets.
 * Usage:
 *   ./offset_difficulty <target> <prime_base0> ... <prime_base9> <offset1> [offset2 ...]
 * Example:
 *   ./offset_difficulty 5739214316657489 a5ad2000 bb40c842 ... 172465143 80639769
 */

#include <gmp.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cstdlib>

#include "../src/PoWCore/src/PoW.h"

using namespace std;

int main(int argc, char **argv) {
  if (argc < 13) {
    cerr << "Usage: " << argv[0] << " <target> <prime_base0> ... <prime_base9> <offset1> [offset2 ...]" << endl;
    return 1;
  }

  // parse target
  uint64_t target = strtoull(argv[1], NULL, 10);

  // parse prime_base (10 words hex or dec)
  uint32_t prime_base[10];
  for (int i = 0; i < 10; ++i) {
    prime_base[i] = static_cast<uint32_t>(strtoul(argv[2 + i], NULL, 0));
  }

  // offsets start at argv[12]
  vector<uint32_t> offsets;
  for (int i = 12; i < argc; ++i) {
    offsets.push_back(static_cast<uint32_t>(strtoul(argv[i], NULL, 10)));
  }

  mpz_t mpz_hash;
  mpz_init(mpz_hash);

  // import prime_base the same way HybridSieve does
  mpz_import(mpz_hash, 10, -1, 4, 0, 0, prime_base);
  mpz_div_2exp(mpz_hash, mpz_hash, 45);

  for (uint32_t off : offsets) {
    mpz_t mpz_adder;
    mpz_init_set_ui(mpz_adder, off);

    PoW pow(mpz_hash, 45, mpz_adder, target, 0);
    uint64_t diff = pow.difficulty();

    char *hex_hash = mpz_get_str(NULL, 16, mpz_hash);
    char *hex_adder = mpz_get_str(NULL, 16, mpz_adder);

    cout << "offset=" << off
         << " difficulty=" << diff
         << " (0x" << hex << diff << dec << ")"
         << " ratio=" << fixed << setprecision(6);
    double ratio = 0.0;
    if (target != 0) ratio = ((double) diff) / ((double) target);
    cout << ratio;
    cout << " mpz_hash=0x" << hex_hash
         << " mpz_adder=0x" << hex_adder
         << endl;

    free(hex_hash);
    free(hex_adder);
    mpz_clear(mpz_adder);
  }

  mpz_clear(mpz_hash);
  return 0;
}
