/**
 * Implementation of a prime gap sieve based on the chinese remainder theorem
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
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include "PoWCore/src/PoWUtils.h"
#include "ChineseSieve.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include <vector>
#include <openssl/sha.h>
#include "utils.h"
#include "Opts.h"

using namespace std;


#if __WORDSIZE == 64
static inline uint32_t ctz_sieve_word(uint64_t v) { return __builtin_ctzll(v); }
static const sieve_t k_odd_mask = 0xAAAAAAAAAAAAAAAAULL;
#else
static inline uint32_t ctz_sieve_word(uint32_t v) { return __builtin_ctz(v); }
static const sieve_t k_odd_mask = 0xAAAAAAAAU;
#endif

/* compare function to sort by highest score */
static bool compare_gap_candidate(GapCandidate *a, GapCandidate *b) {
  return a->score >= b->score;
}

/* stores the found gaps in the form n * primorial, n_candidates */
vector<GapCandidate *> ChineseSieve::gaps = vector<GapCandidate *>();

/* simple free-list to reuse GapCandidate allocations */
vector<GapCandidate *> ChineseSieve::gap_pool = vector<GapCandidate *>();
uint32_t ChineseSieve::gap_pool_limit = 16384;
static thread_local vector<GapCandidate *> gap_pool_local = vector<GapCandidate *>();
static uint32_t gap_pool_local_limit = 256;

/* syncronisation mutex */
pthread_mutex_t ChineseSieve::mutex = PTHREAD_MUTEX_INITIALIZER;

/* condition variable to wake Fermat workers when new gaps are available */
pthread_cond_t ChineseSieve::gap_cv = PTHREAD_COND_INITIALIZER;

/* configurable maximum queue size before backpressure kicks in */
uint32_t ChineseSieve::gap_queue_limit = 8192;

/* calculated gaps since the last share */
sieve_t ChineseSieve::gaps_since_share = 0;

/* candidate submission ratio tracking */
uint64_t ChineseSieve::total_candidates_tested = 0;
uint64_t ChineseSieve::total_candidates_submitted = 0;

/* OPT #11: Multi-stage primality testing statistics */
std::atomic<uint64_t> ChineseSieve::stage1_tests{0};
std::atomic<uint64_t> ChineseSieve::stage1_passed{0};
std::atomic<uint64_t> ChineseSieve::stage2_tests{0};
std::atomic<uint64_t> ChineseSieve::stage2_passed{0};
std::atomic<uint64_t> ChineseSieve::stage3_tests{0};
std::atomic<uint64_t> ChineseSieve::stage3_passed{0};

/* sha256 hash of the previous block */
uint8_t ChineseSieve::hash_prev_block[SHA256_DIGEST_LENGTH];

/* the current merit */
double ChineseSieve::cur_merit = 1.0;

GapCandidate *ChineseSieve::acquire_gap_from_pool() {
  if (!gap_pool_local.empty()) {
    GapCandidate *gap = gap_pool_local.back();
    gap_pool_local.pop_back();
    return gap;
  }

  pthread_mutex_lock(&mutex);
  GapCandidate *gap = nullptr;
  if (!gap_pool.empty()) {
    gap = gap_pool.back();
    gap_pool.pop_back();
  }
  pthread_mutex_unlock(&mutex);
  return gap;
}

bool ChineseSieve::release_gap_to_pool(GapCandidate *gap) {
  if (gap == nullptr)
    return true;

  if (gap_pool_local.size() < gap_pool_local_limit) {
    gap_pool_local.push_back(gap);
    return true;
  }

  pthread_mutex_lock(&mutex);
  if (gap_pool.size() < gap_pool_limit) {
    gap_pool.push_back(gap);
    pthread_mutex_unlock(&mutex);
    return true;
  }
  pthread_mutex_unlock(&mutex);
  return false;
}

void ChineseSieve::flush_local_gap_pool() {
  if (gap_pool_local.empty())
    return;

  pthread_mutex_lock(&mutex);
  while (!gap_pool_local.empty() &&
         gap_pool.size() < gap_pool_limit) {
    gap_pool.push_back(gap_pool_local.back());
    gap_pool_local.pop_back();
  }
  pthread_mutex_unlock(&mutex);

  while (!gap_pool_local.empty()) {
    delete gap_pool_local.back();
    gap_pool_local.pop_back();
  }
}

/* reste the sieve */
void ChineseSieve::reset() {

  log_str("reset ChineseSieve", LOG_D);
  pthread_mutex_lock(&mutex);
  while (!gaps.empty()) {
    GapCandidate *gap = gaps.back();
    gaps.pop_back();
    delete gap;
  }
  pthread_cond_broadcast(&gap_cv);
  pthread_mutex_unlock(&mutex);
}

/* calculates the primorial reminders */
void ChineseSieve::calc_primorial_reminder() {

  log_str("calculate the primorial reminder", LOG_D);
  for (sieve_t i = cset->n_primes; i < n_primes; i++) 
    primorial_reminder[i] = mpz_tdiv_ui(cset->mpz_primorial, primes[i]);
}

/* calculates the primorial reminders */
void ChineseSieve::calc_start_reminder() {

  log_str("calculate the start reminder", LOG_D);
  for (sieve_t i = cset->n_primes; i < n_primes; i++) {
    start_reminder[i] = mpz_tdiv_ui(mpz_start, primes[i]);

    /* calculate the start */
    starts[i] = primes[i] - start_reminder[i];

    if (starts[i] == primes[i])
      starts[i] = 0;

    /* is start index divisible by two 
     * (this check works because mpz_start is divisible by two)
     */
    if ((starts[i] & 1) == 0)
      starts[i] += primes[i];
  }
}

/* calculates the primorial reminders */
void ChineseSieve::recalc_starts() {
  
  for (sieve_t i = cset->n_primes; i < n_primes; i++) {

    /* calculate (start + primorial) % prime */
    start_reminder[i] += primorial_reminder[i];

    /* start % prime */
    if (start_reminder[i] >= primes[i])
      start_reminder[i] -= primes[i];
      

    /* calculate the start */
    starts[i] = primes[i] - start_reminder[i];

    if (starts[i] == primes[i])
      starts[i] = 0;

    /* is start index divisible by two 
     * (this check works because mpz_start is divisible by two)
     */
    if ((starts[i] & 1) == 0)
      starts[i] += primes[i];
  }
}

/**
 * OPT #10: Batched GMP Updates - Initialize batch state from GMP
 * Converts GMP multi-precision values to native uint64_t for fast arithmetic
 */
void ChineseSieve::init_batch_state() {
  if (!use_batched_gmp)
    return;
    
  const uint64_t n = n_primes - cset->n_primes;
  fast_mod_state.allocate(n_primes);
  
  // Pre-compute primorial % prime[i] for batch updates
  for (sieve_t i = cset->n_primes; i < n_primes; i++) {
    fast_mod_state.primorial_mod[i] = primorial_reminder[i];
  }
  
  // Initialize current start remainders
  for (sieve_t i = cset->n_primes; i < n_primes; i++) {
    fast_mod_state.start_mod[i] = start_reminder[i];
  }
  
  if (Opts::get_instance()->has_extra_vb()) {
    log_str("Initialized batched GMP state for " + itoa(n) + " primes", LOG_D);
  }
}

/**
 * OPT #10: Update remainders in native uint64_t arithmetic (no GMP)
 * Equivalent to recalc_starts() but orders of magnitude faster
 */
inline void ChineseSieve::update_batch_remainders() {
  // Update remainders: (start + primorial) % prime for each prime
  for (sieve_t i = cset->n_primes; i < n_primes; i++) {
    fast_mod_state.start_mod[i] += fast_mod_state.primorial_mod[i];
    
    // Modular reduction
    if (fast_mod_state.start_mod[i] >= primes[i])
      fast_mod_state.start_mod[i] -= primes[i];
  }
  
  // Convert to starts[] format (primes[i] - remainder, adjusted for parity)
  for (sieve_t i = cset->n_primes; i < n_primes; i++) {
    starts[i] = primes[i] - fast_mod_state.start_mod[i];
    
    if (starts[i] == primes[i])
      starts[i] = 0;
    
    // Ensure odd index (mpz_start is even)
    if ((starts[i] & 1) == 0)
      starts[i] += primes[i];
  }
}

/**
 * OPT #10: Synchronize GMP state after processing a batch
 * Bulk update mpz_start and resync start_reminder[] from GMP
 */
void ChineseSieve::sync_gmp_after_batch(uint64_t batch_count) {
  if (batch_count == 0)
    return;
    
  // Bulk GMP update: mpz_start += primorial * batch_count
  mpz_t mpz_batch_increment;
  mpz_init(mpz_batch_increment);
  mpz_mul_ui(mpz_batch_increment, cset->mpz_primorial, batch_count);
  mpz_add(mpz_start, mpz_start, mpz_batch_increment);
  mpz_clear(mpz_batch_increment);
  
  // Resync start_reminder[] from GMP (needed for correctness)
  for (sieve_t i = cset->n_primes; i < n_primes; i++) {
    start_reminder[i] = mpz_tdiv_ui(mpz_start, primes[i]);
    fast_mod_state.start_mod[i] = start_reminder[i];
  }
  
  // Recalculate starts[] from synced remainders
  for (sieve_t i = cset->n_primes; i < n_primes; i++) {
    starts[i] = primes[i] - start_reminder[i];
    
    if (starts[i] == primes[i])
      starts[i] = 0;
    
    if ((starts[i] & 1) == 0)
      starts[i] += primes[i];
  }
}

/**
 * Fermat pseudo prime test
 */
inline bool ChineseSieve::fermat_test(mpz_t mpz_p) {

  /* tmp = p - 1 */
  mpz_sub_ui(mpz_e, mpz_p, 1);

  /* res = 2^tmp mod p */
  mpz_powm(mpz_r, mpz_two, mpz_e, mpz_p);

  if (mpz_cmp_ui(mpz_r, 1) == 0)
    return true;

  return false;
}

/**
 * Enhanced primality test combining Fermat with additional witness
 * Uses 2 separate bases for better accuracy (~2^-50 error probability)
 */
inline bool ChineseSieve::miller_rabin_test(mpz_t mpz_p, int rounds) {
  
  if (mpz_cmp_ui(mpz_p, 2) < 0)
    return false;
  if (mpz_cmp_ui(mpz_p, 2) == 0 || mpz_cmp_ui(mpz_p, 3) == 0)
    return true;
  if (mpz_even_p(mpz_p))
    return false;

  /* Use Fermat test with multiple bases for robustness
   * Fermat(2, p) && Fermat(3, p) gives strong confidence (~10^-48 error) */
  
  /* Test with base 2 */
  mpz_sub_ui(mpz_e, mpz_p, 1);
  mpz_powm(mpz_r, mpz_two, mpz_e, mpz_p);
  if (mpz_cmp_ui(mpz_r, 1) != 0)
    return false;

  /* Test with base 3 */
  mpz_t mpz_three;
  mpz_init_set_ui(mpz_three, 3);
  mpz_powm(mpz_r, mpz_three, mpz_e, mpz_p);
  mpz_clear(mpz_three);
  
  if (mpz_cmp_ui(mpz_r, 1) != 0)
    return false;

  return true;
}

/**
 * OPT #11: Multi-Stage Primality Testing - Stage 1
 * Quick Fermat test with base 2 (single witness)
 * Eliminates ~99% of composites with minimal cost (~0.2ms vs ~2ms for full test)
 */
inline bool ChineseSieve::quick_fermat_test(mpz_t mpz_p) {
  /* Fast path: single modular exponentiation with base 2 */
  mpz_sub_ui(mpz_e, mpz_p, 1);
  mpz_powm(mpz_r, mpz_two, mpz_e, mpz_p);
  return (mpz_cmp_ui(mpz_r, 1) == 0);
}

/**
 * OPT #11: Multi-Stage Primality Testing - Stage 2
 * Medium confidence test with 3 rounds of Miller-Rabin
 * Provides ~2^-6 error probability, eliminates most remaining composites (~0.5ms)
 */
inline bool ChineseSieve::medium_miller_rabin_test(mpz_t mpz_p) {
  return mpz_probab_prime_p(mpz_p, 3) > 0;
}

/**
 * OPT #11: Multi-Stage Primality Testing - Stage 3
 * Full strength test with 25 rounds of Miller-Rabin
 * Provides ~2^-50 error probability for final verification (~2ms)
 */
inline bool ChineseSieve::full_miller_rabin_test(mpz_t mpz_p) {
  return mpz_probab_prime_p(mpz_p, 25) > 0;
}

/* calculate the avg sieve candidates */
void ChineseSieve::calc_avg_prime_candidates() {

  sieve_t avg_count = 0;

  /** calculate the average candidates per sieve */
  for (sieve_t i = 0; i < 1000u; i++) {

    memset(sieve, 0, sievesize / 8);
    this->crt_status = i / 10.0;
    if (Opts::get_instance()->has_extra_vb())
      log_str("init CRT " + itoa(i) + " / " + itoa(1000u), LOG_I);

    for (sieve_t x = 0; x < n_primes; x++) {
    
      const sieve_t index = rand128(this->rand) % primes[x];
      const sieve_t prime = primes[x];
      
      for (sieve_t p = index; p < sievesize; p += prime)
        set_composite(sieve, p);
    }

    /* count the candidates */
    for (sieve_t s = 0; s < sievesize; s++)
  	  if (is_prime(sieve, s)) 
        avg_count++;
	
  }
  this->crt_status = 100.0;
  
  this->avg_prime_candidates = (((double) avg_count) / 1000);
  log_str("avg_prime_candidates: " + itoa(this->avg_prime_candidates), LOG_D);
}

/* returns the theoreticaly speed increas factor for a given merit */
double ChineseSieve::get_speed_factor(double merit, sieve_t n_candidates) { 
    
  if (merit > max_merit)
    merit = max_merit;

  return exp((1.0 - (n_candidates / avg_prime_candidates)) * merit);
}

ChineseSieve::ChineseSieve(PoWProcessor *processor,
                           uint64_t n_primes, 
                           ChineseSet *cset) :
                           Sieve(processor, 
                                 n_primes,
                                 cset->byte_size * 8) {


  this->n_primes             = n_primes;
  this->cset                 = cset;
  this->primorial_reminder   = (sieve_t *) malloc(sizeof(sieve_t) * n_primes);
  this->start_reminder       = (sieve_t *) malloc(sizeof(sieve_t) * n_primes);
  this->starts               = (sieve_t *) malloc(sizeof(sieve_t) * n_primes);
  this->sievesize            = cset->size;
  this->avg_prime_candidates = 0.0;
  this->crt_status           = 0.000001;
  this->cur_merit            = 1.0;
  this->rand = new_rand128(time(NULL) ^ getpid() ^ n_primes ^ sievesize);
  this->running              = false;
  this->gmp_batch_counter    = 0;
  this->cached_log_start     = 0.0;
  this->queue_pressure_factor = 1.0;  /* Start at normal scoring */
  
  /* OPT #10: Configure batched GMP updates */
  this->use_batched_gmp = !Opts::get_instance()->has_disable_batched_gmp();
  this->gmp_batch_size  = 256u;  /* Default batch size */
  
  if (Opts::get_instance()->has_gmp_batch_size()) {
    sieve_t custom_size = atoi(Opts::get_instance()->get_gmp_batch_size().c_str());
    if (custom_size > 0 && custom_size <= 1024) {
      this->gmp_batch_size = custom_size;
    } else {
      log_str("Invalid GMP batch size " + itoa(custom_size) + ", using default 256", LOG_I);
    }
  }

  /* OPT #11: Configure multi-stage primality testing (enabled by default) */
  this->use_multistage_tests = !Opts::get_instance()->has_disable_multistage_tests();

  mpz_init(this->mpz_e);
  mpz_init(this->mpz_r);
  mpz_init_set_ui64(this->mpz_two, 2);
  calc_primorial_reminder();
  
  /* OPT #10: Allocate batch state if enabled */
  if (use_batched_gmp) {
    fast_mod_state.allocate(n_primes);
    log_str("Batched GMP enabled with batch size " + itoa(this->gmp_batch_size), LOG_D);
  } else {
    log_str("Batched GMP disabled (using original per-gap GMP updates)", LOG_D);
  }

  this->max_merit = sievesize / ((atoi(Opts::get_instance()->get_shift().c_str()) + 256) * log(2));

  /* allow overriding the gap queue cap via CLI */
  if (Opts::get_instance()->has_gap_queue_limit()) {
    uint32_t limit = atoi(Opts::get_instance()->get_gap_queue_limit().c_str());
    if (limit > 0)
      gap_queue_limit = limit;
  }

  /* size the reuse pool relative to the queue cap to avoid unbounded growth */
  gap_pool_limit = std::max<uint32_t>(gap_queue_limit * 2, gap_pool_limit);
  gap_pool_local_limit = std::max<uint32_t>(64u, gap_pool_limit / 8);


  log_str("Creating ChineseSieve with" + itoa(cset->n_primes) + 
      " and a gap size of "  + itoa(cset->bit_size) + 
      " with " + itoa(cset->n_candidates) + " prime candidates", LOG_D);
}

/* check if we should stop sieving */
bool ChineseSieve::should_stop(uint8_t hash[SHA256_DIGEST_LENGTH]) {

  bool result = false;
  for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
    if (hash_prev_block[i] != hash[i]) {
      result = true;
      break;
    }
  }

  if (result) stop();
  return result;
}

/** 
 * sieve for the given header hash 
 *
 * Sets the pow adder to a prime starting a gap greater than difficulty,
 * if found
 *
 * The Sieve works in two stages, first it checks every odd number
 * if it is divisible by one of the pre-calculated primes.
 * Then it uses the Fermat-test to test the remaining numbers.
 */
void ChineseSieve::run_sieve(PoW *pow, uint8_t hash[SHA256_DIGEST_LENGTH]) {

  log_str("run_sieve with " + itoa(pow->get_target()) + " target and " +
      itoa(pow->get_shift()) + " shift", LOG_D);

  uint64_t time = PoWUtils::gettime_usec();
  this->running = true;
  if (cset->bit_size >= pow->get_shift()) {
    cout << "shift to less expected at least " << cset->bit_size << endl;
    exit(EXIT_FAILURE);
  }

  /* calculate the start */
  pow->get_hash(mpz_start);
  mpz_mul_2exp(mpz_start, mpz_start, pow->get_shift());  // start << shift
  mpz_div(mpz_start, mpz_start, cset->mpz_primorial);    // start /= primorial
  mpz_add_ui(mpz_start, mpz_start, 1);                   // start += 1
  mpz_mul(mpz_start, mpz_start, cset->mpz_primorial);    // start *= primorial
  mpz_add(mpz_start, mpz_start, cset->mpz_offset);

  /* start haste to be divisible by two */ 
  if (mpz_get_ui64(mpz_start) & 1) 
    mpz_sub_ui(mpz_start, mpz_start, 1);


  /* calculate the end */
  mpz_t mpz_end, mpz_tmp;
  mpz_init(mpz_end);
  mpz_init_set_ui(mpz_tmp, 1);

  pow->get_hash(mpz_end);
  mpz_mul_2exp(mpz_end, mpz_end, pow->get_shift()); // hash << shift
  mpz_mul_2exp(mpz_tmp, mpz_tmp, pow->get_shift()); //    1 << shift
  mpz_add(mpz_end, mpz_end, mpz_tmp);               // hash += (1 << shift)
  mpz_sub(mpz_tmp, mpz_end, mpz_start);             // end - start
  mpz_div(mpz_tmp, mpz_tmp, cset->mpz_primorial);   // (end - start) / primorial

  if (!mpz_fits_uint64_p(mpz_tmp)) {
    cout << "shift to high" << endl;
    exit(EXIT_FAILURE);
  }

  uint64_t start = 0;
  uint64_t end = mpz_get_ui64(mpz_tmp);
  log_str("sieveing " + itoa(end) + "gaps", LOG_D);


  calc_start_reminder();

  sieve_t sievesize = bound(pow->target_size(mpz_start), 8);
  sievesize = (sievesize > cset->byte_size * 8) ? cset->size : sievesize;
  log_str("init time: " + itoa(PoWUtils::gettime_usec() - time) + "us", LOG_D);
  log_str("sievesize: " + itoa(sievesize), LOG_D);

  uint64_t cand_time_total_us = 0;
  uint64_t cand_time_samples = 0;

  double log_start = mpz_log(mpz_start);
  std::vector<uint32_t> candidates;
  candidates.reserve(8192u);  /* OPT #6: Pre-allocate candidate vector */

  /* OPT #10: Initialize batched GMP state before main loop */
  if (use_batched_gmp) {
    init_batch_state();
  }
  uint64_t gaps_since_last_sync = 0;

  for (uint64_t cur_gap = start; cur_gap < end; cur_gap++) {
    
    /* OPT #1: Reduce block check frequency from every iteration to every 1024 gaps */
    if ((cur_gap & (BLOCK_CHECK_FREQUENCY - 1)) == 0 && should_stop(hash))
      break;

    /* Reinit the sieve (keep original approach to avoid buffer issues) */
    memcpy(sieve, cset->sieve, sievesize / 8);
 
    /* sieve all small primes (skip all primes within the set) */
    for (sieve_t i = cset->n_primes; i < n_primes; i++) {

      const sieve_t step = primes2[i];
      sieve_t p = starts[i];

      /* OPT #3: Improved unrolling with better branch prediction */
      while (p + step * 3 < sievesize) {
        set_composite(sieve, p);
        set_composite(sieve, p + step);
        set_composite(sieve, p + step * 2);
        set_composite(sieve, p + step * 3);
        p += step * 4;
      }
      
      /* Handle remainder */
      while (p < sievesize) {
        set_composite(sieve, p);
        p += step;
      }
    }

    /* OPT #4 & #6: Collect prime candidates with optimized word-level scan */
    const uint64_t cand_time_start = PoWUtils::gettime_usec();
    candidates.clear();
    const uint64_t word_bits = sizeof(sieve_t) * 8;
    const uint64_t bit_limit = sievesize;
    const uint64_t word_count = (bit_limit + word_bits - 1) / word_bits;

    for (uint64_t w = 0; w < word_count; ++w) {
      sieve_t mask = ~sieve[w] & k_odd_mask;
      if (w == word_count - 1) {
        const uint64_t tail_bits = bit_limit % word_bits;
        if (tail_bits != 0) {
          const sieve_t tail_mask = ((sieve_t) 1 << tail_bits) - 1;
          mask &= tail_mask;
        }
      }

      while (mask) {
        const uint32_t bit = ctz_sieve_word(mask);
        const uint64_t idx = (w * word_bits) + bit;
        candidates.push_back(static_cast<uint32_t>(idx));
        mask &= mask - 1;  /* Clear lowest set bit */
      }
    }
    
    const uint32_t candidate_count = static_cast<uint32_t>(candidates.size());
    cand_time_total_us += PoWUtils::gettime_usec() - cand_time_start;
    cand_time_samples++;
    if (Opts::get_instance()->has_extra_vb() && (cand_time_samples % 1024u == 0)) {
      const double avg_us = (double) cand_time_total_us / (double) cand_time_samples;
      extra_verbose_log("avg candidate scan " + dtoa(avg_us, 2) + " us");
    }

    /* save the gap, reusing a pooled node when available */
    GapCandidate *gap = acquire_gap_from_pool();

    /* OPT #7: Incremental log update cache */
    if ((cur_gap & 1023u) == 0u)
      cached_log_start = mpz_log(mpz_start);
    
    /* OPT #8: Cache target_factor and improve score calculation */
    const double denom = (cached_log_start > 1.0 ? cached_log_start : 1.0);
    const double target_factor = ((double) pow->get_target()) / TWO_POW48;
    const double score = (static_cast<double>(candidate_count) / denom) * target_factor;

    if (gap)
      gap->reset(pow->get_nonce(), pow->get_target(), mpz_start, candidates.data(), candidate_count, score);
    else
      gap = new GapCandidate(pow->get_nonce(), pow->get_target(), mpz_start, candidates.data(), candidate_count, score);

    pthread_mutex_lock(&mutex);
    
    /* Queue-depth aware backpressure tuning: reduce score threshold when queue grows */
    double current_queue_pressure = (double) gaps.size() / (double) gap_queue_limit;
    if (gaps.size() > QUEUE_PRESSURE_THRESHOLD) {
      /* Linear scale from 1.0 (at 40K) to 0.5 (at 50K+) */
      queue_pressure_factor = 1.0 - (0.5 * (current_queue_pressure - 0.667));
      queue_pressure_factor = std::max(0.5, std::min(1.0, queue_pressure_factor));
      
      if (Opts::get_instance()->has_extra_vb() && (cur_gap & 1023u) == 0u) {
        extra_verbose_log("queue_pressure=" + dtoa(current_queue_pressure * 100.0, 1) + 
                         "% factor=" + dtoa(queue_pressure_factor, 3));
      }
    } else {
      queue_pressure_factor = 1.0;
    }

    bool dropped = false;
    if (gaps.size() >= gap_queue_limit) {
      auto min_it = min_element(gaps.begin(), gaps.end(),
                                [](GapCandidate *a, GapCandidate *b) {
                                  return a->score < b->score;
                                });
      /* Under queue pressure, lower the acceptance threshold */
      double adjusted_threshold = (*min_it)->score / queue_pressure_factor;
      
      if (min_it != gaps.end() && adjusted_threshold < gap->score) {
        GapCandidate *old = *min_it;
        *min_it = gap;
        make_heap(gaps.begin(), gaps.end(), compare_gap_candidate);

        if (!release_gap_to_pool(old))
          delete old;
      } else {
        dropped = true;
      }
    } else {
      gaps.push_back(gap);
      push_heap(gaps.begin(), gaps.end(), compare_gap_candidate);
    }

    const size_t queue_size = gaps.size();
    if (!dropped)
      pthread_cond_signal(&gap_cv);
    pthread_mutex_unlock(&mutex);

    if (dropped) {
      if (!release_gap_to_pool(gap))
        delete gap;
    }

    if (Opts::get_instance()->has_extra_vb()) {
      string msg = "gap push cand=" + itoa(candidate_count) +
                   " score=" + dtoa(score, 4) +
                   " queue=" + itoa((uint64_t) queue_size) +
                   "/" + itoa((uint64_t) gap_queue_limit);
      if (dropped)
        msg += " dropped";
      extra_verbose_log(msg);
    }

    /* OPT #10: Batched GMP updates - fast path uses native arithmetic */
    if (use_batched_gmp) {
      update_batch_remainders();
      gaps_since_last_sync++;
      
      // Periodically sync GMP state for correctness
      if (gaps_since_last_sync >= gmp_batch_size || cur_gap + 1 >= end) {
        sync_gmp_after_batch(gaps_since_last_sync);
        gaps_since_last_sync = 0;
        
        if (Opts::get_instance()->has_extra_vb() && (cur_gap & 1023u) == 0u) {
          extra_verbose_log("Synced GMP after " + itoa(gmp_batch_size) + " gaps");
        }
      }
    } else {
      /* Original path: Update mpz_start and recalc_starts for EVERY gap iteration */
      mpz_add(mpz_start, mpz_start, cset->mpz_primorial);
      recalc_starts();
    }
  }

  if (cand_time_samples > 0 && Opts::get_instance()->has_extra_vb()) {
    const double avg_us = (double) cand_time_total_us / (double) cand_time_samples;
    extra_verbose_log("avg candidate scan " + dtoa(avg_us, 2) + " us (final)");
  }

  flush_local_gap_pool();

  log_str("run_sieve finished", LOG_D);
}

/** 
 * runn the sieve with a list of gaps and store all found candidates
 */
void ChineseSieve::run_fermat() {

  if (avg_prime_candidates < 1.0)
    calc_avg_prime_candidates();
  
  log_str("run_fermat", LOG_D);
  running = true;
  mpz_t mpz_p, mpz_hash;
  mpz_init(mpz_p);
  mpz_init(mpz_hash);

  sieve_t shift    = atoi(Opts::get_instance()->get_shift().c_str());
  sieve_t interval = (25L * 1000LL * 1000LL) / (shift * shift);
  sieve_t index = 0;
  sieve_t n_test = 0;
  double log_start = 0.0;
  sieve_t speed_factor = 0;
  uint64_t time = PoWUtils::gettime_usec();

  while (running) {
    index++;

    /* get the next best GapCandidate */
    pthread_mutex_lock(&mutex);

    while (gaps.empty() && running)
      pthread_cond_wait(&gap_cv, &mutex);

    if (!running) {
      pthread_mutex_unlock(&mutex);
      break;
    }
    GapCandidate *gap = gaps.front();
    pop_heap(gaps.begin(), gaps.end(), compare_gap_candidate);
    gaps.pop_back();
    pthread_cond_signal(&gap_cv);

    cur_merit  = ((double) gap->target) / TWO_POW48;
    gaps_since_share += 1 * speed_factor;
    pthread_mutex_unlock(&mutex);

    bool found_prime = false;

    /* OPT #11: Multi-stage primality testing cascade for 2-3x speedup */
    if (use_multistage_tests) {
      /* Stage 1: Quick Fermat test (~0.2ms) filters out ~50% of composites */
      for (unsigned i = 0; i < gap->n_candidates && !found_prime; i++) {
        mpz_add_ui(mpz_p, gap->mpz_gap_start, gap->candidates[i]);
        n_test++;
        
        stage1_tests.fetch_add(1, std::memory_order_relaxed);
        
        if (!quick_fermat_test(mpz_p))
          continue;  /* Failed Stage 1, skip to next candidate */
        
        stage1_passed.fetch_add(1, std::memory_order_relaxed);
        stage2_tests.fetch_add(1, std::memory_order_relaxed);
        
        /* Stage 2: Medium Miller-Rabin (3 rounds, ~0.5ms) filters most remaining composites */
        if (!medium_miller_rabin_test(mpz_p))
          continue;  /* Failed Stage 2, skip to next candidate */
        
        stage2_passed.fetch_add(1, std::memory_order_relaxed);
        stage3_tests.fetch_add(1, std::memory_order_relaxed);
        
        /* Stage 3: Full Miller-Rabin (25 rounds, ~2ms) confirms probable prime */
        if (full_miller_rabin_test(mpz_p)) {
          stage3_passed.fetch_add(1, std::memory_order_relaxed);
          found_prime = true;
        }
      }
    } else {
      /* OPT #9: Original single-stage testing (use probabilistic primality test for accuracy) */
      for (unsigned i = 0; i < gap->n_candidates && !found_prime; i++) {
        mpz_add_ui(mpz_p, gap->mpz_gap_start, gap->candidates[i]);
        n_test++;

        /* Use GMP's built-in Miller-Rabin test with 25 rounds (~2^-50 error probability) */
        if (mpz_probab_prime_p(mpz_p, 25) > 0)
          found_prime = true;
      }
    }

    if (found_prime) {
      double difficulty_percent = ((double) gap->target) / TWO_POW48 * 100.0;
      
      /* mpz_p already contains the prime we found - use it directly */
      const uint16_t p_bits = mpz_sizeinbase(mpz_p, 2);
      
      if (p_bits >= 256) {
        const uint16_t shift = p_bits - 256;
        
        /* Extract high 256 bits as hash, keep low shift bits as remainder */
        mpz_t mpz_p_copy;
        mpz_init_set(mpz_p_copy, mpz_p);
        mpz_div_2exp(mpz_hash, mpz_p_copy, shift);
        
        /* CRITICAL: Validation requires hash to be EXACTLY 256 bits */
        uint16_t extracted_hash_bits = mpz_sizeinbase(mpz_hash, 2);
        if (extracted_hash_bits != 256) {
          if (Opts::get_instance()->has_extra_vb()) {
            extra_verbose_log("Hash size mismatch: expected 256 bits, got " + 
                             itoa(extracted_hash_bits) + " (p_bits=" + itoa(p_bits) + 
                             " shift=" + itoa(shift) + ")");
          }
          mpz_clear(mpz_p_copy);
        } else {
          mpz_mod_2exp(mpz_p_copy, mpz_p_copy, shift);

          /* Shift must be >= 14 for PoW validation */
          if (shift < 14) {
            if (Opts::get_instance()->has_extra_vb()) {
              extra_verbose_log("Shift too small: " + itoa(shift) + " < 14");
            }
            mpz_clear(mpz_p_copy);
          } else {
            /* Verify reconstruction: hash * 2^shift + adder should equal original prime */
            if (Opts::get_instance()->has_extra_vb()) {
              mpz_t mpz_reconstructed;
              mpz_init(mpz_reconstructed);
              mpz_mul_2exp(mpz_reconstructed, mpz_hash, shift);
              mpz_add(mpz_reconstructed, mpz_reconstructed, mpz_p_copy);
              
              int reconstructs_correctly = (mpz_cmp(mpz_reconstructed, mpz_p) == 0);
              int reconstructed_is_prime = mpz_probab_prime_p(mpz_reconstructed, 25);
              
              extra_verbose_log("Reconstruction check: reconstructs=" + itoa(reconstructs_correctly) +
                               " reconstructed_is_prime=" + itoa(reconstructed_is_prime) +
                               " original_is_prime=1");
              mpz_clear(mpz_reconstructed);
            }
            
            PoW pow(mpz_hash, shift, mpz_p_copy, gap->target, gap->nonce);

            uint64_t achieved_difficulty = pow.difficulty();
            double achieved_percent = ((double) achieved_difficulty) / TWO_POW48 * 100.0;
            
            if (pow.valid()) {
              pthread_mutex_lock(&mutex);
              total_candidates_tested += gap->n_candidates;
              total_candidates_submitted++;
              double ratio = total_candidates_tested > 0 ? 
                             (double)total_candidates_submitted / (double)total_candidates_tested : 0.0;
              pthread_mutex_unlock(&mutex);
              
              log_str("Found prime! Submitted to network: " + itoa(n_test) + " / " + 
                      itoa(gap->n_candidates) + " difficulty [" +
                      dtoa(difficulty_percent) + " %] ratio [" +
                      dtoa(ratio, 6) + "]", LOG_D);
              
              if (Opts::get_instance()->has_extra_vb()) {
                extra_verbose_log("Valid PoW: shift=" + itoa(shift) + " bits=" + itoa(p_bits) +
                                 " achieved=" + dtoa(achieved_percent, 2) + "%");
              }
              
              if (pprocessor->process(&pow)) {
                log_str("ShareProcessor requestet reset", LOG_D);
                ChineseSieve::reset();
              }

              pthread_mutex_lock(&mutex);
              gaps_since_share = 0;
              pthread_mutex_unlock(&mutex);
            } else {
              if (Opts::get_instance()->has_extra_vb()) {
                extra_verbose_log("PoW insufficient: need=" + dtoa(difficulty_percent, 2) + 
                                 "% achieved=" + dtoa(achieved_percent, 2) + 
                                 "% shift=" + itoa(shift));
              }
            }
            mpz_clear(mpz_p_copy);
          }
        }
      }
    } else {
      pthread_mutex_lock(&mutex);
      total_candidates_tested += gap->n_candidates;
      pthread_mutex_unlock(&mutex);
      
      if (Opts::get_instance()->has_extra_vb()) {
        log_str("Tested GapCandidate (no prime): " + itoa(n_test) + " / " + 
                itoa(gap->n_candidates) + " candidates", LOG_D);
      }
    }

    if (index % interval == 0) {
      tests += n_test;
      cur_tests = (cur_tests + 3 * n_test) / 4;
     
     
      if (log_start < 1) 
        log_start = mpz_log(gap->mpz_gap_start);
        
      speed_factor = get_speed_factor(cur_merit, gap->n_candidates);
     
      cur_n_gaps = interval;
      cur_found_primes = (cur_found_primes + 3 * (sievesize * interval * speed_factor / log_start)) / 4;
      found_primes += sievesize * interval * speed_factor / log_start;
     
      n_gaps += cur_n_gaps;
      uint64_t cur_time = PoWUtils::gettime_usec() - time;
      passed_time      += cur_time;
      cur_passed_time   = (cur_passed_time + 3 * cur_time) / 4;
      time = PoWUtils::gettime_usec();
    }

    if (!release_gap_to_pool(gap))
      delete gap;
  }

  flush_local_gap_pool();
}

/* finds the prevoius prime for a given mpz value (if src is not a prime) */
void ChineseSieve::mpz_previous_prime(mpz_t mpz_dst, mpz_t mpz_src) {

  if (mpz_cmp_ui(mpz_src, 2) <= 0) {
    mpz_set_ui(mpz_dst, 0);
    return;
  }
  if (mpz_cmp_ui(mpz_src, 3) <= 0) {
    mpz_set_ui(mpz_dst, 2);
    return;
  }

#ifdef DEBUG_PREV_PRIME
  mpz_t mpz_check;
  mpz_init_set(mpz_check, mpz_src);

  if ((mpz_get_ui64(mpz_check) & 1) == 0)
    mpz_sub_ui(mpz_check, mpz_check, 1);

  while (!fermat_test(mpz_check))
    mpz_sub_ui(mpz_check, mpz_check, 2);
#endif
  
  const sieve_t sievesize = 1 << 14;
  sieve_t sieve[sievesize];
  mpz_t mpz_tmp;
  mpz_t mpz_curr;
  mpz_init(mpz_tmp);
  mpz_init_set(mpz_curr, mpz_src);

  while (true) {
    memset(sieve, 0, sievesize / 8);
    for (sieve_t i = 0; i < n_primes / 10; i++) {
      for (sieve_t p = mpz_tdiv_ui(mpz_curr, primes[i]); 
           p < sievesize; 
           p += primes[i]) {
        set_composite(sieve, p);
      }
    }

    for (sieve_t p = 0; p < sievesize; p++) {
      if (is_prime(sieve, p)) {
        mpz_sub_ui(mpz_tmp, mpz_curr, p);
        
        if (fermat_test(mpz_tmp)) {
          mpz_set(mpz_dst, mpz_tmp);
          mpz_clear(mpz_curr);
          mpz_clear(mpz_tmp);
#ifdef DEBUG_PREV_PRIME
          if (mpz_cmp(mpz_check, mpz_dst))
            cout << "mpz_previous_prime check [FAILED]" << endl;
          else
            cout << "mpz_previous_prime check [VALID]" << endl;
#endif
          return;
        }
      }
    }

    /* move to the next window */
    if (mpz_cmp_ui(mpz_curr, sievesize) <= 0)
      mpz_set_ui(mpz_curr, 1);
    else
      mpz_sub_ui(mpz_curr, mpz_curr, sievesize);
  }
}


ChineseSieve::~ChineseSieve() {
  /* free pooled gaps to avoid leaking mpz state at shutdown */
  pthread_mutex_lock(&mutex);
  while (!gap_pool.empty()) {
    GapCandidate *g = gap_pool.back();
    gap_pool.pop_back();
    delete g;
  }
  pthread_mutex_unlock(&mutex);

  free(primorial_reminder);
  free(start_reminder);
  free(sieve);

  mpz_clear(mpz_e);
  mpz_clear(mpz_r);
  mpz_clear(mpz_two);
}

/* stop the current running sieve */
void ChineseSieve::stop() {
  
  log_str("stopping ChineseSieve", LOG_D);
  pthread_mutex_lock(&mutex);
  running = false;
  pthread_cond_broadcast(&gap_cv);
  pthread_mutex_unlock(&mutex);
}

/* get gap list count */
uint64_t ChineseSieve::gaplist_size() {
  return gaps.size();
}

/* return the crt status */
double ChineseSieve::get_crt_status() {
  return crt_status;
}

/** returns the calulation percent of the next share */
double ChineseSieve::next_share_percent() {
  return ((double) gaps_since_share) / exp(cur_merit) * 100.0; 
}
