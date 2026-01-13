/**
 * Implementation of Gapcoins Proof of Work calculation unit.
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
#ifndef CPU_ONLY
#ifndef __STDC_FORMAT_MACROS 
#define __STDC_FORMAT_MACROS 
#endif
#ifndef __STDC_LIMIT_MACROS  
#define __STDC_LIMIT_MACROS  
#endif
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>
#include <gmp.h>
#include <mpfr.h>
#include <pthread.h>
#include <iostream>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "utils.h"
#include "HybridSieve.h"
#include "Opts.h"

bool reset_stats = false;

#if __WORDSIZE == 64
/**
 * Sets the given bit-position in a 64-bit array
 */
#define set_bit(ary, i) (ary[(i) >> 6] |= (1L << ((i) & 0x3f)))
    
/**
 * Unset the given bit-position in a 64-bit array
 */
#define unset_bit(ary, i) (ary[(i) >> 6] &= ~(1L << ((i) & 0x3f)))

/**
 * returns whether the given bit-position in a 64-bit array is set or not
 */
#define bit_at(ary, i) (ary[(i) >> 6] & (1L << ((i) & 0x3f)))
#else
/**
 * Sets the given bit-position in a 32-bit array
 */
#define set_bit(ary, i) (ary[(i) >> 5] |= (1 << ((i) & 0x1f)))
    
/**
 * Unset the given bit-position in a 32-bit array
 */
#define unset_bit(ary, i) (ary[(i) >> 5] &= ~(1 << ((i) & 0x1f)))

/**
 * returns whether the given bit-position in a 32-bit array is set or not
 */
#define bit_at(ary, i) (ary[(i) >> 5] & (1 << ((i) & 0x1f)))
#endif

/**
 * returns whether the given index is a prime or not
 */
#define is_prime(ary, i) !bit_at(ary, i)

/**
 * marks the given index in the given array as composite
 */
#define set_composite(ary, i) set_bit(ary, i)

/**
 * sets x to the next greater number divisible by y
 */
#define bound(x, y) ((((x) + (y) - 1) / (y)) * (y))

/**
 * returns the sieve limit for an simple sieve of Eratosthenes
 */
#define sieve_limit(x) ((uint64_t) (sqrt((double) (x)) + 1))

/**
 * generate x^2
 */
#define POW(X) ((X) * (X))

/* gpu group size */
#define gpu_groub_size 256

/* gpu operand size */
#define gpu_op_size 10

using namespace std;

class MutexGuard {
public:
    explicit MutexGuard(pthread_mutex_t& mtx) : mutex_(mtx) {
        pthread_mutex_lock(&mutex_);
    }
    ~MutexGuard() {
        pthread_mutex_unlock(&mutex_);
    }
    MutexGuard(const MutexGuard&) = delete;
    MutexGuard& operator=(const MutexGuard&) = delete;

private:
    pthread_mutex_t& mutex_;
};



/**
 * create a new HybridSieve
 */
HybridSieve::HybridSieve(PoWProcessor *pprocessor, 
                         uint64_t n_primes, 
                         uint64_t sievesize,
                         uint64_t work_items,
                         uint64_t n_tests,
                         uint64_t queue_size) : Sieve(pprocessor, 
                                                      n_primes,
                                                      sievesize) { 

  log_str("creating HybridSieve", LOG_D);
  this->n_primes         = n_primes;
  this->work_items       = work_items;
  this->passed_time      = 1;
  this->cur_passed_time  = 1;
  this->gpu_list = new GPUWorkList(work_items * gpu_groub_size / n_tests, 
                                   n_tests, 
                                   pprocessor, 
                                   this, 
                                   GPUFermat::get_instance()->get_prime_base_buffer(),
                                   GPUFermat::get_instance()->get_candidates_buffer(),
                                   &tests,
                                   &cur_tests);

  this->sieve_queue = new SieveQueue(queue_size, 
                                     this, 
                                     gpu_list, 
                                     &cur_found_primes, 
                                     &found_primes);


  pthread_create(&gpu_thread, NULL, gpu_results_thread, (void *) gpu_list);
  pthread_create(&gpu_thread, NULL, gpu_work_thread, (void *) sieve_queue);
}


HybridSieve::~HybridSieve() { 
  
  log_str("deleting HybridSieve", LOG_D);
  stop();
  
  gpu_list->running    = false;
  sieve_queue->running = false;
  
  pthread_join(gpu_thread, NULL);
  pthread_join(results_thread, NULL);

  delete sieve_queue;
  delete gpu_list;

  free(candidates_template);
}

/** 
 * sieve for the given header hash 
 *
 * Sets the pow adder to a prime starting a gap greater than difficulty,
 * if found
 *
 * The HybridSieve works in two stages, first it checks every odd number
 * if it is divisible by one of the pre-calculated primes.
 * Then it uses the Fermat-test to test the remaining numbers.
 */
void HybridSieve::run_sieve(PoW *pow, 
                            vector<uint8_t> *offset, 
                            uint8_t hash[SHA256_DIGEST_LENGTH]) {
  
  log_str("run_sieve with " + itoa(pow->get_target()) + " target and " +
      itoa(pow->get_shift()) + " shift", LOG_D);

  if (Opts::get_instance()->has_extra_vb()) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "Starting new sieve" << endl;
    pthread_mutex_unlock(&io_mutex);
  }
  running = true;
  sieve_queue->clear();
  
  /* just to be sure */
  pow->set_shift(45);

  mpz_t mpz_offset;
  mpz_init_set_ui64(mpz_offset, 0);

  if (offset != NULL)
    ary_to_mpz(mpz_offset, offset->data(), offset->size());

  /* make sure offset (and later start) is divisible by two */
  if (mpz_get_ui64(mpz_offset) & 0x1)
    mpz_add_ui(mpz_offset, mpz_offset, 1L);

  mpz_t mpz_adder;
  mpz_init(mpz_adder);

  pow->get_hash(mpz_start);
  mpz_mul_2exp(mpz_start, mpz_start, pow->get_shift());
  mpz_add(mpz_start, mpz_start, mpz_offset);

  /* calculates for each prime, the first index in the sieve
   * which is divisible by that prime */
  calc_muls();

  /* run the sieve till stop signal arrives */
  for (sieve_t sieve_round = 0; 
       running && !should_stop(hash) && sieve_round * sievesize < UINT32_MAX - sievesize; 
       sieve_round++) {
  
    /* speed measurement */
    uint64_t start_time = PoWUtils::gettime_usec();
    
    if (reset_stats) {
      reset_stats      = false;
      cur_found_primes = 0;
      cur_tests        = 0;
      cur_passed_time  = 1;
    }
    

    /* clear the sieve */
    memset(sieve, 0, sievesize / 8);

    /* sieve all small primes (skip 2) */
    for (sieve_t i = 1; i < n_primes; i++) {

      /**
       * sieve all odd multiplies of the current prime
       */
      sieve_t p;
      for (p = starts[i]; p < sievesize; p += primes2[i])
        set_composite(sieve, p);

      starts[i] = p - sievesize;
    }

    if (!should_stop(hash)) 
      sieve_queue->push(new SieveItem(sieve, sievesize, sieve_round, hash, mpz_start, pow));

    passed_time     += PoWUtils::gettime_usec() - start_time;
    cur_passed_time += PoWUtils::gettime_usec() - start_time;
 
  }
  
  mpz_clear(mpz_offset);
  mpz_clear(mpz_adder);
}

/* create a new SieveItem */
HybridSieve::SieveItem::SieveItem(sieve_t *sieve, 
                                  sieve_t sievesize, 
                                  sieve_t sieve_round,
                                  uint8_t hash[SHA256_DIGEST_LENGTH],
                                  mpz_t mpz_start,
                                  PoW *pow) {

  this->sieve       = (sieve_t *) malloc(sievesize / 8);
  this->sievesize   = sievesize;
  this->sieve_round = sieve_round;
  this->pow         = pow;
  mpz_init_set(this->mpz_start, mpz_start);
  memcpy(this->sieve, sieve, sievesize / 8);
  memcpy(this->hash,  hash,  SHA256_DIGEST_LENGTH);
}

/* destroys a SieveItem */
HybridSieve::SieveItem::~SieveItem() {
  free(sieve);
  mpz_clear(mpz_start);
}

/* creates a new SieveQueue */
HybridSieve::SieveQueue::SieveQueue(unsigned capacity,
                                    HybridSieve *hsieve,
                                    GPUWorkList *gpu_list,
                                    uint64_t *cur_found_primes,
                                    uint64_t *found_primes) {

  this->capacity         = capacity;
  this->running          = true;
  this->hsieve           = hsieve;
  this->gpu_list         = gpu_list;
  this->cur_found_primes = cur_found_primes;
  this->found_primes     = found_primes;

  pthread_mutex_init(&access_mutex, NULL);
  pthread_cond_init(&notfull_cond, NULL);
  pthread_cond_init(&full_cond, NULL);
}

/* destroys a SieveQueue */
HybridSieve::SieveQueue::~SieveQueue() {
  pthread_mutex_destroy(&access_mutex);
  pthread_cond_destroy(&notfull_cond);
  pthread_cond_destroy(&full_cond);
}

/* remove the oldest gpu work */
HybridSieve::SieveItem *HybridSieve::SieveQueue::pull() {
 
  pthread_mutex_lock(&access_mutex);

  while (q.empty())
    pthread_cond_wait(&notfull_cond, &access_mutex);
   
  SieveItem *work = q.front();
  q.pop();

  pthread_cond_signal(&full_cond);
  pthread_mutex_unlock(&access_mutex);

  return work;
}

/* add an new SieveItem */
void HybridSieve::SieveQueue::push(HybridSieve::SieveItem *work) {

  pthread_mutex_lock(&access_mutex);

  while (q.size() >= capacity)
    pthread_cond_wait(&full_cond, &access_mutex);

  q.push(work);

  pthread_cond_signal(&notfull_cond);
  pthread_mutex_unlock(&access_mutex);
}

/* clear this */
void HybridSieve::SieveQueue::clear() {
 
  pthread_mutex_lock(&access_mutex);

  while (!q.empty()) {
    SieveItem *work = q.front();
    q.pop();
    delete work;
  }

  pthread_cond_signal(&full_cond);
  pthread_mutex_unlock(&access_mutex);
}

/* get the size of this */
size_t HybridSieve::SieveQueue::size() {
 
  return q.size();
}

/* indicates that this queue is full */
bool HybridSieve::SieveQueue::full() {
  return (q.size() >= capacity);
}

/* the gpu thread */
void *HybridSieve::gpu_work_thread(void *args) {
  log_str("starting gpu_work_thread", LOG_D);
  if (Opts::get_instance()->has_extra_vb()) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "GPU work thread started\n";
    pthread_mutex_unlock(&io_mutex);
  }

#ifndef WINDOWS
  struct sched_param param;
  param.sched_priority = sched_get_priority_min(SCHED_IDLE);
  sched_setscheduler(0, SCHED_IDLE, &param);
#endif
  
  SieveQueue *queue                  = (SieveQueue *) args;
  HybridSieve *hsieve                = queue->hsieve;
  HybridSieve::GPUWorkList *gpu_list = queue->gpu_list;
  vector<uint32_t> offset_buffer;
  offset_buffer.reserve(1024);

  mpz_t mpz_p, mpz_e, mpz_r, mpz_two, mpz_start;
  mpz_init_set_ui64(mpz_p, 0);
  mpz_init_set_ui64(mpz_e, 0);
  mpz_init_set_ui64(mpz_r, 0);
  mpz_init_set_ui64(mpz_two, 2);
  mpz_init_set_ui64(mpz_start, 0);

  while (queue->running) {
    unique_ptr<SieveItem> sitem(queue->pull());
    if (!sitem)
      continue;

    PoW *pow            = sitem->pow;
    sieve_t *sieve      = sitem->sieve;
    sieve_t sievesize   = sitem->sievesize;
    sieve_t sieve_round = sitem->sieve_round;
    mpz_set(mpz_start, sitem->mpz_start);

    double  d_difficulty = ((double) pow->get_target()) / TWO_POW48;
    sieve_t min_len      = log(mpz_get_d(mpz_start)) * d_difficulty;
    sieve_t start        = 0;
    sieve_t i            = 1;

    min_len &= ~((sieve_t) 1);
    if (min_len < 2)
      min_len = 2;
    if (min_len >= sievesize)
      min_len = (sievesize > 2) ? ((sievesize & ~((sieve_t) 1)) - 2) : 2;

    bool stop_requested = hsieve->should_stop(sitem->hash);

    if (sieve_round == 0 && !stop_requested) {
      size_t exported_size;
      uint32_t prime_base[10] = {0};
      mpz_export(prime_base, &exported_size, -1, 4, 0, 0, mpz_start);
      gpu_list->reinit(prime_base, pow->get_target(), pow->get_nonce());
    }

    if (!stop_requested) {
      for (; i < sievesize; i += 2) {
        if (!is_prime(sieve, i))
          continue;

        mpz_add_ui(mpz_p, mpz_start, i + sievesize * sieve_round);
        mpz_sub_ui(mpz_e, mpz_p, 1);
        mpz_powm(mpz_r, mpz_two, mpz_e, mpz_p);

        if (mpz_cmp_ui(mpz_r, 1) == 0) {
          i += 2;
          start = i + sievesize * sieve_round;
          break;
        }
      }
    }

    while (!stop_requested && i < sievesize - min_len) {
      const size_t needed_capacity = static_cast<size_t>(min_len / 2) + 1;
      if (offset_buffer.capacity() < needed_capacity)
        offset_buffer.reserve(needed_capacity);
      offset_buffer.clear();

      for (sieve_t n = 0; n < min_len; n += 2) {
        if (is_prime(sieve, i + n))
          offset_buffer.push_back(i + n + sievesize * sieve_round);
      }

      if (!offset_buffer.empty()) {
        size_t emitted = 0;
        while (emitted < offset_buffer.size()) {
          size_t chunk = min(offset_buffer.size() - emitted,
                             static_cast<size_t>(numeric_limits<uint16_t>::max()));
          uint32_t chunk_start = (emitted == 0) ? start : 0;
          gpu_list->add(new GPUWorkItem(offset_buffer.data() + emitted,
                                        static_cast<uint16_t>(chunk),
                                        min_len,
                                        chunk_start));
          start = 0;
          emitted += chunk;
        }
      }

      i += min_len;
      stop_requested = hsieve->should_stop(sitem->hash);
    }

    *queue->cur_found_primes += i / 198;
    *queue->found_primes     += i / 198;

    if (stop_requested) {
      queue->clear();
    }
  }

  mpz_clear(mpz_p);
  mpz_clear(mpz_e);
  mpz_clear(mpz_r);
  mpz_clear(mpz_two);
  mpz_clear(mpz_start);
  return NULL;
}


/* the gpu results processing thread */
void *HybridSieve::gpu_results_thread(void *args) {

  log_str("starting gpu_results_thread", LOG_D);
  if (Opts::get_instance()->has_extra_vb()) {
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "GPU result processing thread started\n";
    pthread_mutex_unlock(&io_mutex);
  }

#ifndef WINDOWS
  /* use idle CPU cycles for mining */
  struct sched_param param;
  param.sched_priority = sched_get_priority_min(SCHED_IDLE);
  sched_setscheduler(0, SCHED_IDLE, &param);
#endif

  GPUWorkList *list = (GPUWorkList *) args;
  GPUFermat *fermat = GPUFermat::get_instance();
  uint32_t *result  = fermat->get_results_buffer();
  const bool extra_verbose = Opts::get_instance()->has_extra_vb();

  while (list->running) {
    const uint64_t cycle_start = PoWUtils::gettime_usec();

    list->create_candidates();
    if (!list->running)
      break;

    fermat->fermat_gpu();
    if (!list->running)
      break;

    list->parse_results(result);
    if (!list->running)
      break;

    const uint32_t processed = list->n_candidates();
    *list->tests     += processed;
    *list->cur_tests += processed;

    if (extra_verbose) {
      const uint64_t cycle_end = PoWUtils::gettime_usec();
      pthread_mutex_lock(&io_mutex);
      cout << get_time() << "GPU cycle processed " << processed
           << " candidates in " << (cycle_end - cycle_start) / 1000.0 << " ms\n";
      pthread_mutex_unlock(&io_mutex);
    }
  }

  return NULL;
}

/* stop the current running sieve */
void HybridSieve::stop() {
  
  log_str("stopping HybridSieve", LOG_D);
  running = false;
}

/* check if we should stop sieving */
bool HybridSieve::should_stop(uint8_t hash[SHA256_DIGEST_LENGTH]) {

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

/* create new work item */
HybridSieve::GPUWorkItem::GPUWorkItem(uint32_t *offsets, uint16_t len, uint16_t min_len, uint32_t start) {

  this->offsets = (uint32_t *) malloc(sizeof(uint32_t) * len);
  memcpy(this->offsets, offsets, sizeof(uint32_t) * len);
  this->end       = 0;
  this->len       = len;
  this->index     = len - 1;
  this->min_len   = min_len;
  this->start     = start;
  this->next      = NULL;
  this->first_end = 0;

#ifdef DEBUG_FAST
  /* check offsets remove*/
  for (uint32_t i = 0; i < len; i++) {
    if (offsets[i] == 0) {
      cout << "[DD] offsets[" << i << "] = 0,  len = " << len << endl;
    }
  }
#endif
}

HybridSieve::GPUWorkItem::~GPUWorkItem() {
  free(offsets);
}

/* get the next candidate offset */
uint32_t HybridSieve::GPUWorkItem::pop() {
  return (index < 0) ? (((uint32_t) --index) & 1u) | 1u : offsets[index--];
}

/* set a number to be prime (i relative to index) 
 * returns true if this can be skipped */
#ifndef DEBUG_BASIC
void HybridSieve::GPUWorkItem::set_prime(int16_t i) {
#else
void HybridSieve::GPUWorkItem::set_prime(int16_t i, uint32_t prime_base[10]) {
#endif
  
  if (index + i >= 0) {
    if (end == 0) {

      if (first_end == 0 && next != NULL /* && next->get_start() == 0 TODO*/ ) {
      
#ifdef DEBUG_SLOW
        if (prime_base != NULL) {
          mpz_t mpz;
          mpz_init(mpz);
          mpz_import(mpz, 10, -1, 4, 0, 0, prime_base);
          mpz_add_ui(mpz, mpz, next->offsets[0] - 2);
         
          for (uint32_t n = 0; !mpz_probab_prime_p(mpz, 25) && n < 1000; n++)
            mpz_sub_ui(mpz, mpz, 2);
         
          uint32_t real_start = mpz_get_ui(mpz) & 0xFFFFFFFF;
          if (real_start != offsets[index + i]) {
            cout << "[DD] setting start to wrong offset:" << endl;
            cout << "     real_start              = " << real_start << endl;
            cout << "     start                   = " << offsets[index + i] << endl;
            cout << "     next->start - this->end = " << next->offsets[0] -  offsets[len - 1] << endl;
            cout << "     real_start - start      = " << real_start -  offsets[index + i] << endl;
            cout << "     next->offsets[0]        = " << next->offsets[0] << endl;
            cout << "     offsets[len - 1]        = " << offsets[len - 1] << endl;
            cout << "     end-index               = " << index + i << endl;
            cout << "     len                     = " << len << endl;
         
         
            mpz_import(mpz, 10, -1, 4, 0, 0, prime_base);
            mpz_add_ui(mpz, mpz, offsets[len - 1]);
           
            while (!mpz_probab_prime_p(mpz, 25))
              mpz_sub_ui(mpz, mpz, 2);
           
            uint32_t real_end = mpz_get_ui(mpz) & 0xFFFFFFFF;
           
            cout << "     real_end                = " << real_end << endl;
            cout << "     end                     = " << offsets[index + i] << endl;
            cout << "     real_end - end          = " << real_end -  offsets[index + i] << endl;

            int32_t ind = -1;
            for (int32_t x = 0; x < len; x++)
              if (offsets[x] == real_end)
                ind = x;

            if (ind != -1)
              cout << "     real-end-index          = " << ind << endl;
            else
              cout << "     real-end-index          = NOT FOUND" << endl;

            ind = -1;
            for (int32_t x = 0; x < len; x++)
              if (offsets[x] == real_start)
                ind = x;

            if (ind != -1)
              cout << "     real-start-index (cur)  = " << ind << endl;
            else
              cout << "     real-start-index (cur)  = NOT FOUND" << endl;


            ind = -1;
            for (int32_t x = 0; x < next->get_len(); x++)
              if (next->offsets[x] == real_start)
                ind = x;

            if (ind != -1)
              cout << "     real-start-index (next) = " << ind << endl;
            else
              cout << "     real-start-index (next) = NOT FOUND" << endl;

            if (real_start > offsets[len - 1])
              cout << "     real_start - this->end  = " << real_start - offsets[len - 1] << endl;
            else
              cout << "     this->end - real_start  = " << offsets[len - 1] - real_start << endl;

            if (next->offsets[0] > real_start)
              cout << "     next->start-real_start  = " << next->offsets[0] - real_start << endl;
            else
              cout << "     real_start-next->start  = " << real_start - next->offsets[0] << endl;
           
            print(prime_base);
            next->print(prime_base);
          }
          mpz_clear(mpz);
        }
#endif

        next->set_start(offsets[index + i]);
      }

      first_end = offsets[index + i];
    }

    end = offsets[index + i];

#ifdef DEBUG_FAST
    if (end == 0) {
      cout << "[DD] end setted to 0 in: " << __FILE__ << ":" << __LINE__ << endl;
    }
#endif
  }
}

/* sets the gapstart of this */
void HybridSieve::GPUWorkItem::set_start(uint32_t start) {

#ifdef DEBUG_FAST
  if (this->start != 0)
    cout << "[DD] setting this->start from " << this->start << " to " << start << endl;
#endif

  this->start = start;
}

/* returns whether this gap can be skipped */
bool HybridSieve::GPUWorkItem::skip() {
  return (start != 0 && end != 0 && next != NULL && 
          (end <= start || end - start < min_len));
}

/* returns whether this is a valid gap */
bool HybridSieve::GPUWorkItem::valid() {
  if (start == 0 || index >= 0)
    return false;
  
  if (end == 0) {
    if (next != NULL)
      next->mark_skipable();
    return true;
  }
  
  return end > start && end - start >= min_len;
}

/* returns the start offset */
uint32_t HybridSieve::GPUWorkItem::get_start() { return start; }

/* returns the number of offsets of this */
uint16_t HybridSieve::GPUWorkItem::get_len() { return len; }

/* returns the number of current offsets of this */
uint16_t HybridSieve::GPUWorkItem::get_cur_len() { 
  return (index >= 0) ? index + 1 : 0; 
}


/* create a new gpu work list */
HybridSieve::GPUWorkList::GPUWorkList(uint32_t len, 
                    uint32_t n_tests,
                    PoWProcessor *pprocessor,
                    HybridSieve *sieve,
                    uint32_t *prime_base,
                    uint32_t *candidates,
                    uint64_t *tests,
                    uint64_t *cur_tests) {
  
  log_str("creating new GPUWorkList", LOG_D);
  
  pthread_mutex_init(&access_mutex, NULL);
  pthread_cond_init(&notfull_cond, NULL);
  pthread_cond_init(&full_cond, NULL);

  this->len           = len;
  this->cur_len       = 0;
  this->n_tests       = n_tests;
  this->pprocessor    = pprocessor;
  this->sieve         = sieve;
  this->prime_base    = prime_base;
  this->candidates    = candidates;
  this->tests         = tests;
  this->cur_tests     = cur_tests;
  this->start         = NULL;
  this->end           = NULL;
  this->running       = true;
  this->extra_verbose = Opts::get_instance()->has_extra_vb();

  memset(prime_base, 0, sizeof(uint32_t) * 10);
  
  mpz_init_set_ui(mpz_hash, 0);
  mpz_init_set_ui(mpz_adder, 0);
}

HybridSieve::GPUWorkList::~GPUWorkList() {
  log_str("deleting GPUWorkList", LOG_D);
  
  pthread_mutex_destroy(&access_mutex);
  pthread_cond_destroy(&notfull_cond);
  pthread_cond_destroy(&full_cond);
  
  mpz_clear(mpz_hash);
  mpz_clear(mpz_adder);
}

/* returns the size of this */
size_t HybridSieve::GPUWorkList::size() {
  MutexGuard guard(access_mutex);

  while (cur_len < len)
    pthread_cond_wait(&full_cond, &access_mutex);

  size_t size = sizeof(GPUWorkList) + sizeof(uint32_t) * 10 * len * n_tests;

  for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next) 
    size += sizeof(GPUWorkItem) + sizeof(uint32_t) * cur->get_len();

  return size;
}

/* returns the average length*/
uint16_t HybridSieve::GPUWorkList::avg_len() {
  MutexGuard guard(access_mutex);

  while (cur_len < len)
    pthread_cond_wait(&full_cond, &access_mutex);

  uint64_t total_len = 0;
  uint64_t count = 0;

  for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next, count++) 
    total_len += cur->get_len();

  return (count > 0) ? total_len / count : 0;
}

/* returns the average length*/
uint16_t HybridSieve::GPUWorkList::avg_cur_len() {
  MutexGuard guard(access_mutex);

  while (cur_len < len)
    pthread_cond_wait(&full_cond, &access_mutex);

  uint64_t total_len = 0;
  uint64_t count = 0;

  for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next, count++) 
    total_len += cur->get_cur_len();

  return (count > 0) ? total_len / count : 0;
}

/* returns the min length*/
uint16_t HybridSieve::GPUWorkList::min_cur_len() {
  MutexGuard guard(access_mutex);

  while (cur_len < len)
    pthread_cond_wait(&full_cond, &access_mutex);

  uint16_t min = UINT16_MAX;
  for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next) {
    uint16_t cur_len_val = cur->get_cur_len();
    if (cur_len_val < min)
      min = cur_len_val;
    if (min == 0) break;  // Early exit optimization
  }

  return min;
}

/* reinits this */
void HybridSieve::GPUWorkList::reinit(uint32_t prime_base[10], uint64_t target, uint32_t nonce) {

  MutexGuard guard(access_mutex);

  clear();
  this->target = target;
  this->nonce = nonce;
  memcpy(this->prime_base, prime_base, sizeof(uint32_t) * 10);

  pthread_cond_signal(&notfull_cond);
}

/* returns the number of candidates */
uint32_t HybridSieve::GPUWorkList::n_candidates() { 
  return len * n_tests; 
}

/* add a item to the list */
void HybridSieve::GPUWorkList::add(GPUWorkItem *item) {
  
  MutexGuard guard(access_mutex);

  while (cur_len >= len)
  pthread_cond_wait(&notfull_cond, &access_mutex);

  if (start == NULL) {
  start = end = item;
  } else {
  // Link gap boundaries between consecutive items
  if (end->get_end() != 0 && item->get_start() == 0)
    item->set_start(end->get_end());
  else if (end->get_end() == 0 && item->get_start() != 0)
    end->set_end();

  end->next = item;
  end = item;
  }

  cur_len++;
  
  if (cur_len >= len)
  pthread_cond_signal(&full_cond);
}

/* creates the candidate array to process */
void HybridSieve::GPUWorkList::create_candidates() {

  MutexGuard guard(access_mutex);

  while (cur_len < len)
    pthread_cond_wait(&full_cond, &access_mutex);

#ifdef DEBUG_FAST
  this->check = get_xor(); 
#endif
  
  uint32_t i = 0;
  for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next) {
    for (uint32_t n = 0; n < n_tests; n++)
      candidates[i + n] = cur->pop();

    i += n_tests;
  }
}

/* parse the gpu results */
void HybridSieve::GPUWorkList::parse_results(uint32_t *results) {

#ifdef DEBUG_FAST
  if (check != get_xor()) 
    cout << "[DD] GPUWorkItems CHANGED!!!!  " << check << " == " << get_xor() << endl;
#endif
  
  uint32_t i = 0;
  GPUWorkItem *del = NULL;
  for (GPUWorkItem *cur = start, *prev = NULL; cur != NULL; cur = cur->next) {

    if (del != NULL)
      delete del;

    for (uint32_t n = 0; n < n_tests; n++) {
      if (results[i + n]) {
#ifndef DEBUG_BASIC
        cur->set_prime(n_tests - n);    
#else
        cur->set_prime(n_tests - n, prime_base);    
#endif
          
#ifdef DEBUG_SLOW
        mpz_import(mpz_hash, 10, -1, 4, 0, 0, prime_base);
        mpz_add_ui(mpz_hash, mpz_hash, cur->get_prime(n_tests - n));
   
        if (!mpz_probab_prime_p(mpz_hash, 25)) {
          cout << "[DD] in parse_results: prime test failed!!! i: " << i; 
          cout << " n: " << n << " n_tests: " << n_tests << endl;
        }
#endif
      } 
#ifdef DEBUG_SLOW
      else {
        mpz_import(mpz_hash, 10, -1, 4, 0, 0, prime_base);
        mpz_add_ui(mpz_hash, mpz_hash, cur->get_prime(n_tests - n));
   
        if (mpz_probab_prime_p(mpz_hash, 25)) {
          cout << "[DD] in parse_results: composite test failed!!! i: ";
          cout << i << " n: " << n << " n_tests: " << n_tests << endl;
        } 
      }

      if (candidates[i + n] != cur->get_prime(n_tests - n)) {
        cout << "[DD] candidates[" << i + n << "] = " << candidates[i + n];
        cout << " != " <<  cur->get_prime(n_tests - n) << endl;
      }
#endif
    }

    if (cur->skip() || cur->valid()) {
       
       if (cur->valid()) {
#ifdef DEBUG_FAST
         if (prev != NULL)  
           prev->print(prime_base);   
         cur->print(prime_base);     
#endif
         submit(cur->get_start());
       }

       if (prev == NULL && cur->next == NULL) {
         start = NULL;
         end = NULL;
       } else if (prev == NULL && cur->next != NULL) {
         start = cur->next;
       } else if (prev != NULL && cur->next == NULL) {
         end = prev;
         end->next = NULL;
       } else
         prev->next = cur->next;

       del = cur;
       cur_len--;
       
    } else {
      prev = cur;
      del = NULL;
    }

    i += n_tests;
  }

  if (del != NULL)
    delete del;

  pthread_cond_signal(&notfull_cond);
  pthread_mutex_unlock(&access_mutex);

  if (extra_verbose) {
    stringstream ss;
    ss << get_time() << "GPU-Items: " << setprecision(3) << size() / 1048576.0;
    ss << " MB  avg: " << setw(3) << avg_cur_len() << " tests  min: ";
    ss << setw(3) << min_cur_len() << " tests" << endl;

    pthread_mutex_lock(&io_mutex);
    cout << ss.str();
    pthread_mutex_unlock(&io_mutex);
  }
}

/**
 * calculate for every prime the first
 * index in the sieve which is divisible by that prime
 * (and not divisible by two)
 */
void HybridSieve::calc_muls() {
  mpz_t remainder, diff;
  mpz_init(remainder);
  mpz_init(diff);

  for (sieve_t i = 0; i < n_primes; ++i) {
    const mpz_class& prime = primes[i];

    mpz_mod(remainder, mpz_start, prime.get_mpz_t());
    if (mpz_sgn(remainder) == 0) {
      starts[i] = 0;
      continue;
    }

    mpz_sub(diff, prime.get_mpz_t(), remainder);

    // Ensure diff fits into sieve_t without overflow.
    sieve_t start_value;
    if (mpz_fits_ulong_p(diff)) {
      unsigned long tmp = mpz_get_ui(diff);
      const unsigned long max_sieve_t =
          static_cast<unsigned long>(std::numeric_limits<sieve_t>::max());
      if (tmp > max_sieve_t) {
        tmp = max_sieve_t;
      }
      start_value = static_cast<sieve_t>(tmp);
    } else {
      // If diff does not fit in unsigned long, clamp to maximum sieve_t value.
      start_value = std::numeric_limits<sieve_t>::max();
    }

    starts[i] = start_value;
    if (prime != 2 && (starts[i] & 1u) == 0)
      starts[i] += prime.get_ui();
  }

  mpz_clear(diff);
  mpz_clear(remainder);
}

/* submits a given offset */
bool HybridSieve::GPUWorkList::submit(uint32_t offset) {
  
  mpz_import(mpz_hash, 10, -1, 4, 0, 0, prime_base);
  mpz_div_2exp(mpz_hash, mpz_hash, 45);
  mpz_set_ui(mpz_adder, offset);

#ifdef DEBUG_FAST
  static unsigned valid = 0, invalid = 0;
  mpz_t mpz, next;
  mpz_init_set(mpz, mpz_hash);
  mpz_mul_2exp(mpz, mpz, 32);
  mpz_add_ui(mpz, mpz, offset);
  mpz_init_set(next, mpz);
  mpz_nextprime(next, mpz);
  mpz_sub(next, next, mpz);

  if (!mpz_probab_prime_p(mpz, 25)) {
    cout << "[DD] submit: prime test failed!!!\n";
  } else
    cout << "[DD] submit: prime test OK len: " << mpz_get_ui64(next) << endl;
  
  mpz_nextprime(next, mpz);
  cout << "[DD] end offset: " << (mpz_get_ui64(next) & 0xFFFFFFFF) << endl;
  mpz_clear(mpz);
  mpz_clear(next);
#endif

  PoW pow(mpz_hash, 45, mpz_adder, target, nonce);

  if (pow.valid()) {
#ifdef DEBUG_FAST
    valid++;
    cout << "[DD] PoW valid (" << valid << ")\n";
#endif 

    /* stop calculating if processor said so */
    if (pprocessor->process(&pow)) {
      sieve->stop();
      return true;
    }
  } 
#ifdef DEBUG_FAST
  else {
    invalid++;
    cout << "[DD] PoW invalid!!! (" << invalid << ")\n";
  }
#endif

  return false;
}

/* clears the list */
void HybridSieve::GPUWorkList::clear() {

  GPUWorkItem *prev = NULL;
  for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next) {
    if (prev != NULL)
      delete prev;

    prev = cur;
  }

  if (prev != NULL)
    delete prev;

  start   = NULL;
  end     = NULL;
  cur_len = 0;
}

/* tells this that it should be skipped anyway */
void HybridSieve::GPUWorkItem::mark_skipable() {
  
  start = offsets[len - 1];

#ifdef DEBUG_FAST
  if (start == 0) {
    cout << "[DD] last offset = 0 in " << __FILE__ << ":" << __LINE__ << endl;
  }
#endif
}

/* returns the end offset */
uint32_t HybridSieve::GPUWorkItem::get_end() { return first_end; }

/* sets the end of this so that 
 * it don't sets the start of the next item */
void HybridSieve::GPUWorkItem::set_end() { first_end = (uint32_t) -1; }

/* debugging related functions */
#ifdef DEBUG_BASIC

/* returns the prime at a given index offset i */
uint32_t HybridSieve::GPUWorkItem::get_prime(int32_t i) {

  if (index + i >= 0) {
    return offsets[index + i];
  }

  return 1;
}

/* simple xor check to validate the items */
uint32_t HybridSieve::GPUWorkItem::get_xor() { 

  uint32_t x = 0;

  for (int32_t i = 0; i < len; i++)
    x ^= offsets[i];

  return x;
  
}

/* prints this */
void HybridSieve::GPUWorkItem::print(uint32_t prime_base[10]) {
  cout << "GPUWorkItem(start=" << start << ", end=" << end;
  cout << ", min_len=" << min_len << ", len=";
  cout << len << ", index=" << index << ")\n";
  cout << "            end - start:  " << end - start << endl;
  cout << "            offsets[0]:   " << offsets[0] << endl;
  cout << "            offsets[len]: " << offsets[len - 1] << endl;

  mpz_t mpz;
  mpz_init(mpz);
  mpz_import(mpz, 10, -1, 4, 0, 0, prime_base);
  mpz_add_ui(mpz, mpz, offsets[0] - 2);

  while (!mpz_probab_prime_p(mpz, 25))
    mpz_sub_ui(mpz, mpz, 2);

  cout << "            real_start:   " << (mpz_get_ui(mpz) & 0xFFFFFFFF) << endl;
  mpz_clear(mpz);
}

/* simple xor check to validate the items */
uint32_t HybridSieve::GPUWorkList::get_xor() { 

  uint32_t x = 0;

  for (GPUWorkItem *cur = start; cur->next != NULL; cur = cur->next) 
    x ^= cur->get_xor();

  return x;
  
}

/* returns the current prime_base of this */
uint32_t *HybridSieve::GPUWorkList::get_prime_base() { return prime_base; }
#endif /* DEBUG_BASIC */

#endif /* CPU_ONLY */
