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

#define MAX_CANDIDATES 65536

/* compare function to sort by highest score */
static bool compare_gap_candidate(GapCandidate *a, GapCandidate *b) {
  return a->score >= b->score;
}

/* stores the found gaps in the form n * primorial, n_candidates */
vector<GapCandidate *> ChineseSieve::gaps = vector<GapCandidate *>();

/* simple free-list to reuse GapCandidate allocations */
vector<GapCandidate *> ChineseSieve::gap_pool = vector<GapCandidate *>();
uint32_t ChineseSieve::gap_pool_limit = 16384;

/* syncronisation mutex */
pthread_mutex_t ChineseSieve::mutex = PTHREAD_MUTEX_INITIALIZER;

/* condition variable to wake Fermat workers when new gaps are available */
pthread_cond_t ChineseSieve::gap_cv = PTHREAD_COND_INITIALIZER;

/* configurable maximum queue size before backpressure kicks in */
uint32_t ChineseSieve::gap_queue_limit = 8192;

/* calculated gaps since the last share */
sieve_t ChineseSieve::gaps_since_share = 0;

/* sha256 hash of the previous block */
uint8_t ChineseSieve::hash_prev_block[SHA256_DIGEST_LENGTH];

/* the current merit */
double ChineseSieve::cur_merit = 1.0;

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

/* calculate the avg sieve candidates */
void ChineseSieve::calc_avg_prime_candidates() {

  sieve_t avg_count = 0;

  /** calculate the average candidates per sieve */
  for (sieve_t i = 0; i < 1000u; i++) {

    memset(sieve, 0, sievesize / 8);
    this->crt_status = i / 10.0;
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

  mpz_init(this->mpz_e);
  mpz_init(this->mpz_r);
  mpz_init_set_ui64(this->mpz_two, 2);
  calc_primorial_reminder();

  this->max_merit = sievesize / ((atoi(Opts::get_instance()->get_shift().c_str()) + 256) * log(2));

  /* allow overriding the gap queue cap via CLI */
  if (Opts::get_instance()->has_gap_queue_limit()) {
    uint32_t limit = atoi(Opts::get_instance()->get_gap_queue_limit().c_str());
    if (limit > 0)
      gap_queue_limit = limit;
  }

  /* size the reuse pool relative to the queue cap to avoid unbounded growth */
  gap_pool_limit = std::max<uint32_t>(gap_queue_limit * 2, gap_pool_limit);


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


  for (uint64_t cur_gap = start; cur_gap < end && !should_stop(hash); cur_gap++) {

   /* reinit the sieve */
    memcpy(sieve, cset->sieve, sievesize / 8);
 
    /* sieve all small primes (skip all primes within the set) */
    for (sieve_t i = cset->n_primes; i < n_primes; i++) {

      const sieve_t step = primes2[i];
      sieve_t p = starts[i];

      /* unroll marking to reduce branch/loop overhead */
      while (p < sievesize) {
        set_composite(sieve, p);
        p += step;
        if (p >= sievesize) break;
        set_composite(sieve, p);
        p += step;
        if (p >= sievesize) break;
        set_composite(sieve, p);
        p += step;
        if (p >= sievesize) break;
        set_composite(sieve, p);
        p += step;
      }
    }

    /* collect the prime candidates */
    uint32_t candidates[MAX_CANDIDATES];
    uint32_t candidate_count = 0;
    for (uint32_t i = 1; i < sievesize && candidate_count < MAX_CANDIDATES; i += 2)
      if (is_prime(sieve, i))
        candidates[candidate_count++] = i;

    /* optional backpressure: keep the gap queue bounded to avoid lock churn */
    pthread_mutex_lock(&mutex);
    while (running && gaps.size() >= gap_queue_limit)
      pthread_cond_wait(&gap_cv, &mutex);
    pthread_mutex_unlock(&mutex);

    const double log_start = mpz_log(mpz_start);
    const double denom = (log_start > 1.0 ? log_start : 1.0);
    const double target_factor = ((double) pow->get_target()) / TWO_POW48;
    const double score = (static_cast<double>(candidate_count) / denom) * target_factor;

    /* save the gap, reusing a pooled node when available */
    GapCandidate *gap = nullptr;
    pthread_mutex_lock(&mutex);
    if (!gap_pool.empty()) {
      gap = gap_pool.back();
      gap_pool.pop_back();
    }
    pthread_mutex_unlock(&mutex);

    if (gap)
      gap->reset(pow->get_nonce(), pow->get_target(), mpz_start, candidates, candidate_count, score);
    else
      gap = new GapCandidate(pow->get_nonce(), pow->get_target(), mpz_start, candidates, candidate_count, score);

    pthread_mutex_lock(&mutex);

    gaps.push_back(gap);
    push_heap(gaps.begin(), gaps.end(), compare_gap_candidate);
    const size_t queue_size = gaps.size();
    pthread_cond_signal(&gap_cv);
    pthread_mutex_unlock(&mutex);

    if (Opts::get_instance()->has_extra_vb()) {
      string msg = "gap push cand=" + itoa(candidate_count) +
                   " score=" + dtoa(score, 4) +
                   " queue=" + itoa((uint64_t) queue_size) +
                   "/" + itoa((uint64_t) gap_queue_limit);
      extra_verbose_log(msg);
    }

    mpz_add(mpz_start, mpz_start, cset->mpz_primorial);

    /* recalculate the start for the given gap */
    recalc_starts();
  }

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

    /* check all prime candidates for the current GapCandidate */
    for (unsigned i = 0; i < gap->n_candidates && !found_prime; i++) {
      mpz_add_ui(mpz_p, gap->mpz_gap_start, gap->candidates[i]);
      n_test++;

      if (fermat_test(mpz_p))
        found_prime = true;
    }

    if (!found_prime) {
      log_str("Found GapCandidate: " + itoa(n_test) + " / " + 
              itoa(gap->n_candidates) + " share [" +
              dtoa(next_share_percent()) + " %]", LOG_D);
      
      /* calculate the adder */
      mpz_previous_prime(mpz_p, gap->mpz_gap_start);
      const uint16_t shift = mpz_sizeinbase(mpz_p, 2) - 256;
      mpz_div_2exp(mpz_hash, mpz_p, shift);
      mpz_mod_2exp(mpz_p, mpz_p, shift);

      PoW pow(mpz_hash, shift, mpz_p, gap->target, gap->nonce);

      if (pow.valid()) {
        if (pprocessor->process(&pow)) {
          log_str("ShareProcessor requestet reset", LOG_D);
          ChineseSieve::reset();
        }

        pthread_mutex_lock(&mutex);
        gaps_since_share = 0;
        pthread_mutex_unlock(&mutex);
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

    bool recycled = false;
    pthread_mutex_lock(&mutex);
    if (gap_pool.size() < gap_pool_limit) {
      gap_pool.push_back(gap);
      recycled = true;
    }
    pthread_mutex_unlock(&mutex);

    if (!recycled)
      delete gap;
  }
}

/* finds the prevoius prime for a given mpz value (if src is not a prime) */
void ChineseSieve::mpz_previous_prime(mpz_t mpz_dst, mpz_t mpz_src) {

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
  mpz_init(mpz_tmp);

  memset(sieve, 0, sievesize / 8);
  for (sieve_t i = 0; i < n_primes / 10; i++) {
    for (sieve_t p = mpz_tdiv_ui(mpz_src, primes[i]); 
         p < sievesize; 
         p += primes[i]) {
      
      set_composite(sieve, p);
    }
  }

  for (sieve_t p = 0; p < sievesize; p++) {
    if (is_prime(sieve, p)) {
      mpz_sub_ui(mpz_tmp, mpz_src, p);
      
      if (fermat_test(mpz_tmp)) {
        mpz_set(mpz_dst, mpz_tmp);
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

  /* start again */
  mpz_sub_ui(mpz_tmp, mpz_src, sievesize);
  mpz_previous_prime(mpz_dst, mpz_tmp);
  mpz_clear(mpz_tmp);
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
