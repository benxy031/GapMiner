/**
 * Implementation of a storage class of a chinese reminder theorem resut
 *
 * Copyright (C)  2014  Jonny Frey  <j0nn9.fr39@gmail.com>
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
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include "ChineseSet.h"
#include "utils.h"

using namespace std;

/* creats a new ChineseSet */
ChineseSet::ChineseSet(sieve_t n_primes, 
                       sieve_t size, 
                       sieve_t n_candidates, 
                       const char *offset) {

  log_str("creating ChineseSet with" + itoa(n_primes) + " primes", LOG_D);
  mpz_init_set_str(mpz_offset, offset, 0);
  mpz_init_set_ui(mpz_primorial, 1);

  this->n_primes     = n_primes;
  this->size         = size;
  this->n_candidates = n_candidates;
  
  init();
}

/* creats a new ChineseSet */
ChineseSet::ChineseSet(sieve_t n_primes, 
                       sieve_t size, 
                       sieve_t n_candidates, 
                       mpz_t mpz_offset) {

  log_str("creating ChineseSet with" + itoa(n_primes) + " primes", LOG_D);
  mpz_init_set(this->mpz_offset, mpz_offset);
  mpz_init_set_ui(mpz_primorial, 1);

  this->n_primes     = n_primes;
  this->size         = size;
  this->n_candidates = n_candidates;

  init();
}

/* creats a new ChineseSet */
ChineseSet::ChineseSet(const char *fname) {
  
  log_str("creating ChineseSet from file: " + fname, LOG_D);
  FILE *file = fopen(fname, "r");

  fseek(file, strlen("|== ChineseSet ==|\n"), SEEK_CUR);

  fscanf(file, "n_primes:     %" PRISIEVE "\n", &this->n_primes);
  fscanf(file, "size:         %" PRISIEVE "\n", &this->size);
  fscanf(file, "n_candidates: %" PRISIEVE "\n", &this->n_candidates);
  fseek(file, strlen("offset:       "), SEEK_CUR);

  mpz_init_set_ui(mpz_offset, 0);
  mpz_init_set_ui(mpz_primorial, 1);
  mpz_inp_str(mpz_offset, file, 10);

  init();
  fclose(file);
}

/* init this */
void ChineseSet::init() {

  /* generate the primorial */
  for (sieve_t i = 0; i < n_primes; i++)
    mpz_mul_ui(mpz_primorial, mpz_primorial, first_primes[i]);

  this->max_merit = size / (mpz_log(mpz_primorial) + log(2) * (256 + 20));
  this->byte_size = bound(size, sizeof(sieve_t) * 8) / 8;
  this->bit_size  = mpz_sizeinbase(mpz_primorial, 2);

  /* calculate the speed increase */
  sieve_t avg_count = 0;
  void *sieve_buf = nullptr;
  if (posix_memalign(&sieve_buf, 64, byte_size) != 0)
    sieve_buf = malloc(byte_size);
  sieve = (sieve_t *) sieve_buf;
  this->rand = new_rand128(time(NULL) ^ getpid() ^ n_primes ^ size ^ n_candidates);

  /** calculate the average candidates per sieve */
  for (sieve_t i = 0; i < 10000u; i++) {
    memset(sieve, 0, byte_size);

    /* applay the previous calculated layers */
    for (sieve_t x = 0; x < n_primes; x++) {
    
      const sieve_t index = rand128(this->rand) % first_primes[x];
      const sieve_t prime = first_primes[x];
      
      /* for each posible residue calss creat one sieve */
      for (sieve_t p = index; p < size; p += prime)
        set_composite(sieve, p);

    }

    /* count the candidates */
    sieve_t cur_count = 0;
    for (sieve_t s = 0; s < byte_size / sizeof(sieve_t); s++)
      cur_count += popcount(sieve[s]);

    avg_count += cur_count;
  }
  this->avg_candidates = size - (((double) avg_count) / 10000);
  memset(this->sieve, 0, byte_size);

  /* make offset divisible by two */
  if (mpz_get_ui(mpz_offset) & 1)
    mpz_sub_ui(mpz_offset, mpz_offset, 1);

  /* sieve all small primes */
  for (sieve_t i = 0; i < n_primes; i++) {

    /* calculates for each prime, the first index in the sieve
     * which is divisible by that prime */
    sieve_t start = first_primes[i] - mpz_tdiv_ui(mpz_offset, first_primes[i]);

    if (start == first_primes[i])
      start = 0;

    /**
     * sieve all odd multiplies of the current prime
     */
    for (sieve_t p = start; p < size; p += first_primes[i])
      set_composite(sieve, p);
  }

  /* check the candidates */
  sieve_t n = 0;
  for (sieve_t i = 0; i < size; i++)
    if (is_prime(sieve, i))
      n++;

  if (n > n_candidates) {
    cout << "[EE] ChineseSet failed to creat for " << n_primes << " primes";
    cout << endl << "     n_candidates: " << n_candidates;
    cout << endl << "     n:            " << n;
    cout << endl << "     sievesize:    " << size << endl;
  }
}

/* saves this to a file */
void ChineseSet::save(const char *fname) {

  log_str("saving ChineseSet to file: " + fname, LOG_D);
  FILE *file = fopen(fname, "w");
  fprintf(file, "|== ChineseSet ==|\n");
  save(file);
  fclose(file);
}

/* saves this to a file */
void ChineseSet::save(FILE *file) {

  log_str("saving ChineseSet to file", LOG_D);
  fprintf(file, "n_primes:     %" PRISIEVE "\n", n_primes);
  fprintf(file, "size:         %" PRISIEVE "\n", size);
  fprintf(file, "n_candidates: %" PRISIEVE "\n", n_candidates);
  fprintf(file, "offset:       ");
  mpz_out_str(file, 10, mpz_offset);
}

ChineseSet::~ChineseSet() {

  log_str("deleting ChineseSet", LOG_D);
  free(sieve);
  mpz_clear(mpz_offset);
  mpz_clear(mpz_primorial);
}

/* returns the theoreticaly speed increas factor for a given merit */
double ChineseSet::get_speed_factor(double merit) { 
    
  if (merit > max_merit)
    merit = max_merit;

  return exp((1.0 - (n_candidates / avg_candidates)) * merit);
}
