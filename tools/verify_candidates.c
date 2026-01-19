#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <gmp.h>

// Usage: verify_candidates <base_words_hex_comma_separated> <offset1> [offset2 ...]
// base words: 10 hex words separated by commas, little-endian limbs (limb0 is least significant)

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s base10,base9,...,base0 offset1 [offset2 ...]\n", argv[0]);
        return 1;
    }
    // parse base words
    const char *base_arg = argv[1];
    uint32_t base[10];
    char *tmp = strdup(base_arg);
    char *tok = strtok(tmp, ",");
    int idx = 0;
    while (tok && idx < 10) {
        // allow hex with 0x or without
        unsigned long long val = 0;
        if (sscanf(tok, "%llx", &val) != 1) {
            if (sscanf(tok, "%llu", &val) != 1) {
                fprintf(stderr, "Failed to parse base word '%s'\n", tok);
                return 2;
            }
        }
        base[idx++] = (uint32_t)val;
        tok = strtok(NULL, ",");
    }
    free(tmp);
    if (idx != 10) {
        fprintf(stderr, "Expected 10 base words, got %d\n", idx);
        return 3;
    }

    for (int i = 2; i < argc; ++i) {
        unsigned long long off = 0;
        if (sscanf(argv[i], "%llu", &off) != 1) {
            fprintf(stderr, "Failed to parse offset '%s'\n", argv[i]);
            continue;
        }
        // Construct mpz from base + offset
        mpz_t n;
        mpz_init(n);
        // n = 0
        mpz_set_ui(n, 0);
        unsigned long long carry = off;
        for (int limb = 0; limb < 10; ++limb) {
            unsigned long long sum = (unsigned long long)base[limb] + carry;
            // add sum * (2^(32*limb))
            mpz_t term;
            mpz_init(term);
            mpz_set_ui(term, (unsigned long) (sum & 0xffffffffu));
            if (limb > 0) mpz_mul_2exp(term, term, 32 * limb);
            mpz_add(n, n, term);
            mpz_clear(term);
            carry = sum >> 32;
        }
        if (carry) {
            // add carry shifted by 32*10
            mpz_t term;
            mpz_init(term);
            mpz_set_ui(term, (unsigned long)carry);
            mpz_mul_2exp(term, term, 32*10);
            mpz_add(n, n, term);
            mpz_clear(term);
        }

        int reps = 25;
        int res = mpz_probab_prime_p(n, reps);
        // res: 0=composite, 1=probably prime, 2=definitely prime
        gmp_printf("offset=%llu res=%d\n", off, res);
        mpz_clear(n);
    }
    return 0;
}
