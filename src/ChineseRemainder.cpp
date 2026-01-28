/**
 * Implementation of a Chinese Remainder Theorem computation class
 *
 * Copyright (C)  2015  The Gapcoin developers  <info@gapcoin.org>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "PoWCore/src/Sieve.h"
#include "ChineseRemainder.h"
#include "utils.h"
#include <vector>
#include <gmp.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdint.h>

using namespace std;

/* run the test */
ChineseRemainder::ChineseRemainder(const sieve_t *numbers, 
                                   const sieve_t *reminders, 
                                   const sieve_t len) {
  this->numbers   = (sieve_t *) malloc(sizeof(sieve_t) * len);
  this->reminders = (sieve_t *) malloc(sizeof(sieve_t) * len);
  this->len       = len;

  memcpy(this->numbers,   numbers,   sizeof(sieve_t) * len);
  memcpy(this->reminders, reminders, sizeof(sieve_t) * len);

  /* primorial */
  mpz_init_set_ui(mpz_primorial, 1);
  for (sieve_t i = 0; i < len; i++)
    mpz_mul_ui(mpz_primorial, mpz_primorial, numbers[i]);

  /* Use Garner's algorithm with native integer modular inverses to avoid
     expensive big-int gcd/ext calls per modulus. We compute small mixed-radix
     coefficients and then assemble the final `mpz_target` using mpz operations. */

  /* helper: modular inverse for small integers */
  auto modinv64 = [](uint64_t a, uint64_t m) -> uint64_t {
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t) m, newr = (int64_t) (a % m);
    while (newr != 0) {
      int64_t q = r / newr;
      int64_t tmp = newt; newt = t - q * newt; t = tmp;
      tmp = newr; newr = r - q * newr; r = tmp;
    }
    if (r > 1)
      return 0; /* inverse doesn't exist, shouldn't happen for coprime moduli */
    if (t < 0) t += m;
    return (uint64_t) t;
  };

  /* x[i] will hold the reduced coefficient modulo numbers[i] */
  std::vector<uint64_t> x(len);
  for (sieve_t i = 0; i < len; i++)
    x[i] = (uint64_t) (reminders[i] % numbers[i]);

  for (sieve_t i = 1; i < len; i++) {
    uint64_t mi = (uint64_t) numbers[i];
    uint64_t t = x[i];
    for (sieve_t j = 0; j < i; j++) {
      uint64_t mj = (uint64_t) numbers[j];
      uint64_t inv = modinv64(mj % mi, mi);
      uint64_t xj_mod = x[j] % mi;
      uint64_t diff = (t + mi - xj_mod) % mi;
      t = (diff * inv) % mi;
    }
    x[i] = t;
  }

  /* assemble result into mpz_target: target = x[0] + x[1]*m0 + x[2]*m0*m1 + ... */
  mpz_init_set_ui(mpz_target, 0);
  mpz_t accum; mpz_init_set_ui(accum, 1); /* accum = product m0..m_{i-1} */

  for (sieve_t i = 0; i < len; i++) {
    if (i == 0) {
      mpz_add_ui(mpz_target, mpz_target, x[0]);
    } else {
      mpz_mul_ui(accum, accum, numbers[i-1]);
      mpz_t tmp; mpz_init(tmp);
      mpz_mul_ui(tmp, accum, (unsigned long) x[i]);
      mpz_add(mpz_target, mpz_target, tmp);
      mpz_clear(tmp);
    }
  }

  /* reduce into canonical range */
  mpz_fdiv_r(mpz_target, mpz_target, mpz_primorial);

  /* verify result quickly (cheap modulus checks) */
  for (sieve_t i = 0; i < len; i++) {
    if (mpz_tdiv_ui(mpz_target, numbers[i]) != reminders[i]) {
      log_str("ChineseRemainder Failed!!!", LOG_W);
    }
  }

  mpz_clear(accum);
}

ChineseRemainder::~ChineseRemainder() {
  if (this->numbers)
    free(this->numbers);
  if (this->reminders)
    free(this->reminders);
  mpz_clear(mpz_primorial);
  mpz_clear(mpz_target);
}

