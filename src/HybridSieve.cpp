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
#include <string>
#include <math.h>
#include <gmp.h>
#include <mpfr.h>
#include <pthread.h>
#include <unistd.h>
#include <cstdio>
#include <time.h>
#include <errno.h>
#include <iostream>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
#include <sstream>
#include <iomanip>

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
#define gpu_group_size GPUFermat::get_group_size()

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

HybridSieve::BitmapBufferPool::BitmapBufferPool()
  : buffer_bytes(0),
    primary(nullptr),
    initialized(false) {
  pthread_mutex_init(&pool_mutex, NULL);
  pthread_cond_init(&pool_cond, NULL);
}

HybridSieve::BitmapBufferPool::~BitmapBufferPool() {
  shutdown();
  pthread_mutex_destroy(&pool_mutex);
  pthread_cond_destroy(&pool_cond);
}

void HybridSieve::BitmapBufferPool::init(size_t bytes,
                     size_t min_buffers,
                     sieve_t *primary_buffer) {
  if (initialized)
    return;

  buffer_bytes = bytes;
  primary = primary_buffer;
  free_buffers.clear();
  free_buffers.reserve(min_buffers);
  free_buffers.push_back(primary);

  const size_t total = std::max<size_t>(2u, min_buffers);
  extra_buffers.reserve(total - 1);
  for (size_t i = 1; i < total; ++i) {
    std::unique_ptr<uint8_t[]> storage(new uint8_t[buffer_bytes]);
    memset(storage.get(), 0, buffer_bytes);
    free_buffers.push_back(reinterpret_cast<sieve_t *>(storage.get()));
    extra_buffers.push_back(std::move(storage));
  }

  initialized = true;
}

sieve_t *HybridSieve::BitmapBufferPool::acquire() {
  MutexGuard guard(pool_mutex);
  while (free_buffers.empty())
    pthread_cond_wait(&pool_cond, &pool_mutex);
  sieve_t *buffer = free_buffers.back();
  free_buffers.pop_back();
  return buffer;
}

void HybridSieve::BitmapBufferPool::release(sieve_t *buffer) {
  if (!buffer)
    return;
  MutexGuard guard(pool_mutex);
  free_buffers.push_back(buffer);
  pthread_cond_signal(&pool_cond);
}

sieve_t *HybridSieve::BitmapBufferPool::shutdown() {
  if (!initialized)
    return primary;

  {
    MutexGuard guard(pool_mutex);
    free_buffers.clear();
    free_buffers.push_back(primary);
    initialized = false;
    pthread_cond_broadcast(&pool_cond);
  }

  extra_buffers.clear();
  return primary;
}

HybridSieve::PrimeStartPool::PrimeStartPool()
    : value_count(0), storage(), free_buffers(), is_initialized(false) {
  pthread_mutex_init(&pool_mutex, NULL);
  pthread_cond_init(&pool_cond, NULL);
}

HybridSieve::PrimeStartPool::~PrimeStartPool() {
  shutdown();
  pthread_mutex_destroy(&pool_mutex);
  pthread_cond_destroy(&pool_cond);
}

void HybridSieve::PrimeStartPool::init(size_t count, size_t buffer_count) {
  if (is_initialized)
    return;
  if (count == 0 || buffer_count == 0)
    return;
  value_count = count;
  storage.clear();
  free_buffers.clear();
  storage.reserve(buffer_count);
  free_buffers.reserve(buffer_count);
  for (size_t i = 0; i < buffer_count; ++i) {
    std::unique_ptr<uint32_t[]> buffer(new uint32_t[value_count]);
    memset(buffer.get(), 0, value_count * sizeof(uint32_t));
    free_buffers.push_back(buffer.get());
    storage.push_back(std::move(buffer));
  }
  is_initialized = true;
}

uint32_t *HybridSieve::PrimeStartPool::acquire() {
  if (!is_initialized)
    return nullptr;
  pthread_mutex_lock(&pool_mutex);
  while (free_buffers.empty() && is_initialized)
    pthread_cond_wait(&pool_cond, &pool_mutex);
  uint32_t *buffer = nullptr;
  if (is_initialized && !free_buffers.empty()) {
    buffer = free_buffers.back();
    free_buffers.pop_back();
  }
  pthread_mutex_unlock(&pool_mutex);
  return buffer;
}

void HybridSieve::PrimeStartPool::release(uint32_t *buffer) {
  if (buffer == nullptr)
    return;
  pthread_mutex_lock(&pool_mutex);
  const bool notify = is_initialized;
  if (is_initialized)
    free_buffers.push_back(buffer);
  pthread_mutex_unlock(&pool_mutex);
  if (notify)
    pthread_cond_signal(&pool_cond);
}

void HybridSieve::PrimeStartPool::shutdown() {
  if (!is_initialized)
    return;
  {
    MutexGuard guard(pool_mutex);
    free_buffers.clear();
    storage.clear();
    is_initialized = false;
    value_count = 0;
  }
  pthread_cond_broadcast(&pool_cond);
}



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
  Opts *opts = Opts::get_instance();
  this->n_primes         = n_primes;
  this->work_items       = work_items;
  this->passed_time      = 1;
  this->cur_passed_time  = 1;
  Opts *opts_local = opts;
  const bool queue_override = opts_local && opts_local->has_queue_size();
  const uint32_t group_size_raw = GPUFermat::get_group_size();
  const uint64_t safe_group_size =
      static_cast<uint64_t>(std::max<uint32_t>(1u, group_size_raw));
  const uint64_t safe_n_tests = std::max<uint64_t>(1u, n_tests);
  uint64_t safe_work_items = std::max<uint64_t>(1u, work_items);
  const uint64_t mul_limit = std::numeric_limits<uint64_t>::max() / safe_group_size;
  if (safe_work_items > mul_limit) {
    log_str("work-items value too large for GPU queue auto-sizing; clamping product", LOG_W);
    safe_work_items = mul_limit;
  }
  uint64_t computed_gpu_queue = (safe_work_items * safe_group_size) / safe_n_tests;
  if (computed_gpu_queue == 0)
    computed_gpu_queue = 1u;
  uint64_t requested_gpu_queue = queue_override
                                     ? std::max<uint64_t>(1u, queue_size)
                                     : computed_gpu_queue;
  const uint64_t gpu_queue_limit = std::numeric_limits<uint32_t>::max();
  if (requested_gpu_queue > gpu_queue_limit) {
    log_str("GPU queue override exceeds 32-bit limit; clamping to max", LOG_W);
    requested_gpu_queue = gpu_queue_limit;
  }
  const uint32_t gpu_queue_len = static_cast<uint32_t>(requested_gpu_queue);
  if (queue_override) {
    std::ostringstream ss;
    ss << "GPU queue capacity overridden to " << gpu_queue_len
       << " items via --queue-size";
    log_str(ss.str(), LOG_I);
  }

  this->gpu_list = new GPUWorkList(gpu_queue_len,
                                   n_tests,
                                   pprocessor,
                                   this,
                                   GPUFermat::get_instance()->get_prime_base_buffer(),
                                   GPUFermat::get_instance()->get_candidates_buffer(),
                                   &tests,
                                   &cur_tests);

  uint64_t sieve_queue_capacity = std::max<uint64_t>(1u, queue_size);
  const uint64_t sieve_queue_limit = std::numeric_limits<unsigned>::max();
  if (sieve_queue_capacity > sieve_queue_limit) {
    log_str("Sieve queue capacity exceeds unsigned range; clamping", LOG_W);
    sieve_queue_capacity = sieve_queue_limit;
  }

  this->sieve_queue = new SieveQueue(static_cast<unsigned>(sieve_queue_capacity),
                                     this,
                                     gpu_list,
                                     &cur_found_primes,
                                     &found_primes);

  auto parse_pool_override = [&](const char *flag,
                                const std::string &value,
                                size_t fallback) -> size_t {
    if (value.empty())
      return fallback;
    errno = 0;
    char *endptr = nullptr;
    unsigned long long parsed = strtoull(value.c_str(), &endptr, 10);
    const bool bad_format = (endptr == value.c_str()) || (endptr && *endptr != '\0');
    if (errno == ERANGE || bad_format) {
      log_str(std::string("Invalid value '") + value + "' for " + flag + 
                  ", using " + itoa(static_cast<uint64_t>(fallback)),
              LOG_W);
      return fallback;
    }
    if (parsed < 2ull)
      parsed = 2ull;
    const unsigned long long max_allowed =
        static_cast<unsigned long long>(std::numeric_limits<size_t>::max());
    if (parsed > max_allowed) {
      log_str(std::string("Clamping ") + flag + " to " + itoa(max_allowed), LOG_W);
      parsed = max_allowed;
    }
    return static_cast<size_t>(parsed);
  };

  const uint64_t desired_buffers = queue_size + 2;
  const uint64_t capped_buffers = std::max<uint64_t>(static_cast<uint64_t>(2u), desired_buffers);
  const uint64_t max_buffers = std::numeric_limits<size_t>::max();
  size_t pool_size = static_cast<size_t>(std::min<uint64_t>(capped_buffers, max_buffers));
  if (opts && opts->has_bitmap_pool_buffers())
    pool_size = parse_pool_override("--bitmap-pool-buffers",
                                    opts->get_bitmap_pool_buffers(),
                                    pool_size);
  const size_t bitmap_bytes = static_cast<size_t>(this->sievesize / 8);
  bitmap_pool.init(bitmap_bytes, pool_size, sieve);

#ifdef USE_CUDA_BACKEND
  size_t snapshot_buffers = pool_size;
  if (opts && opts->has_snapshot_pool_buffers())
    snapshot_buffers = parse_pool_override("--snapshot-pool-buffers",
                                           opts->get_snapshot_pool_buffers(),
                                           snapshot_buffers);

  if (opts && opts->use_cuda_sieve_proto()) {
    prime_start_pool.init(static_cast<size_t>(this->n_primes), snapshot_buffers);
    const sieve_t *prime_src = get_primes();
    std::vector<uint32_t> gpu_primes;
    gpu_primes.reserve(static_cast<size_t>(this->n_primes));
    for (sieve_t idx = 0; idx < this->n_primes; ++idx)
      gpu_primes.push_back(static_cast<uint32_t>(prime_src[idx]));
    GPUFermat::get_instance()->configure_sieve_primes(gpu_primes.data(),
                                                      gpu_primes.size());
  }
#endif


  pthread_create(&results_thread, NULL, gpu_results_thread, (void *) gpu_list);
  pthread_create(&gpu_thread, NULL, gpu_work_thread, (void *) sieve_queue);
}


HybridSieve::~HybridSieve() { 
  
  log_str("deleting HybridSieve", LOG_D);
  stop();
  
  gpu_list->stop();
  sieve_queue->running = false;
  
  pthread_join(gpu_thread, NULL);
  pthread_join(results_thread, NULL);

  delete sieve_queue;
  delete gpu_list;

  sieve = bitmap_pool.shutdown();
  prime_start_pool.shutdown();
  free(candidates_template);
}

void HybridSieve::increment_gap_counters(uint64_t count) {
  n_gaps += count;
  cur_n_gaps += count;
  if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
    std::ostringstream ss;
    ss << get_time() << " gap_counter += " << count
       << " (n_gaps=" << n_gaps << ", cur_n_gaps=" << cur_n_gaps << ")";
    extra_verbose_log(ss.str());
  }
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
    std::ostringstream ss;
    ss << get_time() << "Starting new sieve";
    extra_verbose_log(ss.str());
  }
  running = true;
  sieve_queue->clear();
  
  /* set shift, default to 45 for GPU */
  if (Opts::get_instance()->has_shift()) {
    pow->set_shift(atoi(Opts::get_instance()->get_shift().c_str()));
  } else {
    pow->set_shift(45);
  }

  if (gpu_list)
    gpu_list->set_shift(pow->get_shift());

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

#ifdef USE_CUDA_BACKEND
  const bool capture_gpu_starts =
      Opts::get_instance()->use_cuda_sieve_proto() && prime_start_pool.initialized();
#else
  const bool capture_gpu_starts = false;
#endif

  /* run the sieve till stop signal arrives */
  for (sieve_t sieve_round = 0; 
       running && !should_stop(hash) && sieve_round * sievesize < UINT32_MAX - sievesize; 
       sieve_round++) {
  
      const uint32_t queue_cap = gpu_list->capacity();
      const uint32_t high_water = std::max<uint32_t>(1u, (queue_cap * 4u) / 5u);
      while (running && !should_stop(hash) && gpu_list->queued_items() > high_water)
        usleep(1000);

    /* speed measurement */
    uint64_t start_time = PoWUtils::gettime_usec();
    
    if (reset_stats) {
      reset_stats      = false;
      cur_found_primes = 0;
      cur_tests        = 0;
      cur_passed_time  = 1;
    }
    

    sieve_t *active_bitmap = bitmap_pool.acquire();
    this->sieve = active_bitmap;

    /* clear the sieve */
    memset(sieve, 0, sievesize / 8);

#ifdef USE_CUDA_BACKEND
    const size_t prime_snapshot_capacity = static_cast<size_t>(n_primes);
    uint32_t *prime_start_snapshot = nullptr;
    bool offload_window = false;
    if (capture_gpu_starts) {
      prime_start_snapshot = prime_start_pool.acquire();
      if (prime_start_snapshot != nullptr) {
        offload_window = true;
        for (size_t idx = 0; idx < prime_snapshot_capacity; ++idx) {
          sieve_t residue = starts[idx];
          if (residue >= sievesize) {
            const sieve_t step = primes2[idx];
            if (step != 0 && residue >= step)
              residue %= step;
          }
          prime_start_snapshot[idx] = static_cast<uint32_t>(
              std::min<sieve_t>(residue,
                                static_cast<sieve_t>(std::numeric_limits<uint32_t>::max())));
        }
      }
    }
#else
    uint32_t *prime_start_snapshot = nullptr;
    bool offload_window = false;
#endif

    if (offload_window) {
      const sieve_t window = sievesize;
      for (sieve_t idx = 1; idx < n_primes; ++idx) {
        const sieve_t step = primes2[idx];
        if (step == 0)
          continue;
        sieve_t residue = starts[idx];
        if (residue >= window)
          residue %= step;
        const sieve_t remaining = (residue < window) ? (window - residue) : 0;
        const sieve_t steps =
            (remaining == 0) ? 0 : ((remaining + step - 1) / step);
        const sieve_t advanced = residue + steps * step;
        starts[idx] = advanced - window;
      }
    } else {
      /* sieve all small primes (skip 2) */
      for (sieve_t i = 1; i < n_primes; i++) {
        sieve_t p;
        for (p = starts[i]; p < sievesize; p += primes2[i])
          set_composite(sieve, p);
        starts[i] = p - sievesize;
      }
    }

        bool handed_off = false;
        if (!should_stop(hash)) {
    #ifdef USE_CUDA_BACKEND
          PrimeStartPool *start_pool_ptr =
          (offload_window && prime_start_snapshot != nullptr) ? &prime_start_pool
                          : nullptr;
      const uint32_t snapshot_count = (offload_window && prime_start_snapshot != nullptr)
                   ? static_cast<uint32_t>(std::min<sieve_t>(
                     n_primes,
                     static_cast<sieve_t>(std::numeric_limits<uint32_t>::max())))
                   : 0u;
    #else
      PrimeStartPool *start_pool_ptr = nullptr;
      const uint32_t snapshot_count = 0u;
    #endif
      sieve_queue->push(new SieveItem(sieve,
                  sievesize,
                  sieve_round,
                  hash,
                  mpz_start,
                  pow,
                  &bitmap_pool,
                  start_pool_ptr,
                  prime_start_snapshot,
                  snapshot_count));
    #ifdef USE_CUDA_BACKEND
      prime_start_snapshot = nullptr;
    #endif
      handed_off = true;
        }

    if (!handed_off) {
#ifdef USE_CUDA_BACKEND
      if (prime_start_snapshot != nullptr && capture_gpu_starts)
        prime_start_pool.release(prime_start_snapshot);
#endif
      bitmap_pool.release(sieve);
    }

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
                                  PoW *pow,
                                  BitmapBufferPool *pool,
                                  PrimeStartPool *start_pool,
                                  uint32_t *prime_starts,
                                  uint32_t prime_start_count) {

  this->sieve       = sieve;
  this->sievesize   = sievesize;
  this->sieve_round = sieve_round;
  this->pow         = pow;
  this->buffer_pool = pool;
  this->start_pool  = start_pool;
  this->prime_starts = prime_starts;
  this->prime_start_count = prime_start_count;
  mpz_init_set(this->mpz_start, mpz_start);
  memcpy(this->hash,  hash,  SHA256_DIGEST_LENGTH);
}

/* destroys a SieveItem */
HybridSieve::SieveItem::~SieveItem() {
  if (buffer_pool && sieve)
    buffer_pool->release(sieve);
  if (start_pool && prime_starts)
    start_pool->release(prime_starts);
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

  while (q.empty()) {
    if (Opts::get_instance()->has_extra_vb()) {
      std::ostringstream ss;
      ss << get_time() << "GPU work thread waiting for queue";
      extra_verbose_log(ss.str());
    }
    pthread_cond_wait(&notfull_cond, &access_mutex);
  }
   
  SieveItem *work = q.front();
  q.pop();

  pthread_cond_signal(&full_cond);
  pthread_mutex_unlock(&access_mutex);

  return work;
}

HybridSieve::SieveItem *HybridSieve::SieveQueue::try_pull() {
  pthread_mutex_lock(&access_mutex);
  if (q.empty()) {
    pthread_mutex_unlock(&access_mutex);
    return NULL;
  }
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
  Opts *opts = Opts::get_instance();
  if (opts->has_extra_vb()) {
    std::ostringstream ss;
    ss << get_time() << "GPU work thread started";
    extra_verbose_log(ss.str());
  }

#if !defined(USE_CUDA_BACKEND)
  if (opts->use_cuda_sieve_proto()) {
    log_str("CUDA sieve prototype requested but CUDA backend is disabled", LOG_W);
  }
#endif

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

#ifdef USE_CUDA_BACKEND
  GPUFermat *prototype_fermat = opts->use_cuda_sieve_proto() ? GPUFermat::get_instance() : NULL;
#else
  GPUFermat *prototype_fermat = NULL;
#endif

  mpz_t mpz_p, mpz_e, mpz_r, mpz_two, mpz_start;
  mpz_init_set_ui64(mpz_p, 0);
  mpz_init_set_ui64(mpz_e, 0);
  mpz_init_set_ui64(mpz_r, 0);
  mpz_init_set_ui64(mpz_two, 2);
  mpz_init_set_ui64(mpz_start, 0);

  #ifdef USE_CUDA_BACKEND
  // Increased from 4 to 16 for better GPU utilization and amortization of launch overhead
  constexpr uint32_t kMaxPrototypeBatch = 16u;
  struct PrototypeSlice {
    const uint32_t *offsets = nullptr;
    size_t count = 0;
    bool absolute = false;
  };
  auto compute_batch_goal = [&](HybridSieve::GPUWorkList *list) -> uint32_t {
    if (!list)
      return 1u;
    const uint32_t capacity = list->capacity();
    if (capacity == 0)
      return 1u;
    const uint32_t queued = list->queued_items();
    const double fill =
        static_cast<double>(queued) / static_cast<double>(capacity);
    if (fill < 0.15)
      return kMaxPrototypeBatch;
    if (fill < 0.35)
      return std::min<uint32_t>(3u, kMaxPrototypeBatch);
    if (fill < 0.6)
      return std::min<uint32_t>(2u, kMaxPrototypeBatch);
    return 1u;
  };
  unique_ptr<SieveItem> deferred_item;
  #endif

  while (queue->running) {
    unique_ptr<SieveItem> first_item;
#ifdef USE_CUDA_BACKEND
    if (deferred_item) {
      first_item = std::move(deferred_item);
    } else {
      first_item.reset(queue->pull());
    }
#else
    first_item.reset(queue->pull());
#endif
    if (!first_item)
      continue;

#ifdef USE_CUDA_BACKEND
    const bool use_gpu_sieve = (prototype_fermat != NULL);
#else
    const bool use_gpu_sieve = false;
#endif

    vector<unique_ptr<SieveItem>> batch;
    batch.push_back(std::move(first_item));

#ifdef USE_CUDA_BACKEND
    const bool seed_has_snapshot =
        batch[0]->get_prime_starts() != nullptr &&
        batch[0]->get_prime_start_count() > 0;
    if (use_gpu_sieve && !seed_has_snapshot) {
      const uint32_t batch_goal = compute_batch_goal(gpu_list);
      while (batch.size() < batch_goal) {
        SieveItem *extra = queue->try_pull();
        if (!extra)
          break;
        unique_ptr<SieveItem> next(extra);
        const uint32_t *next_starts = next->get_prime_starts();
        const uint32_t next_count = next->get_prime_start_count();
        if (next_starts != nullptr && next_count > 0) {
          deferred_item = std::move(next);
          break;
        }
        batch.push_back(std::move(next));
      }
    }
#endif

    vector<sieve_t> batch_min_len(batch.size(), 0);
    vector<uint8_t> batch_cpu_bitmap_available;
#ifdef USE_CUDA_BACKEND
    vector<GPUFermat::SievePrototypeParams> batch_params;
    vector<size_t> gpu_param_to_batch;
    if (use_gpu_sieve) {
      batch_params.reserve(batch.size());
      gpu_param_to_batch.reserve(batch.size());
      batch_cpu_bitmap_available.assign(batch.size(), 1u);
    }
#endif

    for (size_t idx = 0; idx < batch.size(); ++idx) {
      SieveItem *item = batch[idx].get();
      if (!item)
        continue;

      mpz_set(mpz_start, item->mpz_start);
      double d_difficulty = ((double) item->pow->get_target()) / TWO_POW48;
      sieve_t computed_min_len = log(mpz_get_d(mpz_start)) * d_difficulty;
      computed_min_len &= ~((sieve_t) 1);
      if (computed_min_len < 2)
        computed_min_len = 2;
      if (computed_min_len >= item->sievesize)
        computed_min_len = (item->sievesize > 2)
                               ? ((item->sievesize & ~((sieve_t) 1)) - 2)
                               : 2;

      sieve_t min_len = computed_min_len;

      if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
        std::ostringstream ss;
        ss << get_time() << " effective min_len=" << min_len
           << " computed=" << computed_min_len;
        ss << " sievesize=" << item->sievesize
           << " round=" << item->sieve_round;
        extra_verbose_log(ss.str());
      }

      batch_min_len[idx] = min_len;

#ifdef USE_CUDA_BACKEND
      if (use_gpu_sieve) {
        const size_t word_shift = (sizeof(sieve_t) == 8) ? 6u : 5u;
        const size_t sieve_word_count =
            static_cast<size_t>((item->sievesize >> word_shift) + 1);
        const size_t sieve_bytes = sieve_word_count * sizeof(sieve_t);
        GPUFermat::SievePrototypeParams params{};
        const uint32_t *prime_start_residues = item->get_prime_starts();
        const uint32_t prime_start_count = item->get_prime_start_count();
        const bool build_bitmap_from_residues =
          prime_start_residues != nullptr && prime_start_count > 0;
        const bool has_host_bitmap = !build_bitmap_from_residues;
        if (idx < batch_cpu_bitmap_available.size())
          batch_cpu_bitmap_available[idx] = has_host_bitmap ? 1u : 0u;
        if (build_bitmap_from_residues) {
          params.sieve_bytes = nullptr;
          params.sieve_byte_len = 0u;
          params.prime_starts = prime_start_residues;
          params.prime_count = prime_start_count;
        } else {
          params.sieve_bytes = reinterpret_cast<const uint8_t *>(item->sieve);
          params.sieve_byte_len = static_cast<uint32_t>(
              std::min<size_t>(sieve_bytes,
                                static_cast<size_t>(numeric_limits<uint32_t>::max())));
          params.prime_starts = nullptr;
          params.prime_count = 0u;
        }
        params.sieve_base = mpz_get_ui64(mpz_start);
        params.window_size = static_cast<uint32_t>(
            std::min<sieve_t>(item->sievesize,
                              static_cast<sieve_t>(numeric_limits<uint32_t>::max())));
        params.sieve_round = static_cast<uint32_t>(
            std::min<sieve_t>(item->sieve_round,
                              static_cast<sieve_t>(numeric_limits<uint32_t>::max())));
        params.min_gap = static_cast<uint32_t>(
            std::min<sieve_t>(min_len,
                              static_cast<sieve_t>(numeric_limits<uint32_t>::max())));
        if (params.window_size == 0)
          continue;
        batch_params.push_back(params);
        gpu_param_to_batch.push_back(idx);
      }
#endif
    }

#ifdef USE_CUDA_BACKEND
    vector<PrototypeSlice> gpu_window_slices;
    vector<uint32_t> gpu_absolute_offsets;
    if (use_gpu_sieve)
      gpu_window_slices.assign(batch.size(), PrototypeSlice{});
    if (use_gpu_sieve && !batch_params.empty()) {
      prototype_fermat->prototype_sieve_batch(batch_params.data(),
                                              static_cast<uint32_t>(batch_params.size()));
        const uint32_t *gpu_offsets = prototype_fermat->prototype_offsets_data();
      const uint32_t *window_offsets = prototype_fermat->prototype_window_offsets();
      const uint32_t window_count = prototype_fermat->prototype_window_count();
      if (gpu_offsets && window_offsets && window_count == batch_params.size()) {
          const uint32_t total_offsets = prototype_fermat->prototype_offsets_count();
          gpu_absolute_offsets.clear();
          gpu_absolute_offsets.reserve(total_offsets);
        for (uint32_t w = 0; w < window_count; ++w) {
          if (w >= gpu_param_to_batch.size())
            break;
          const size_t batch_index = gpu_param_to_batch[w];
          if (batch_index >= batch.size())
            continue;
          SieveItem *slice_item = batch[batch_index].get();
          if (!slice_item)
            continue;

          const uint32_t start = window_offsets[w];
          const uint32_t end = window_offsets[w + 1];
          const size_t slice_count = (end > start)
                                         ? static_cast<size_t>(end - start)
                                         : 0u;
          if (slice_count == 0)
            continue;

          const size_t slice_offset = gpu_absolute_offsets.size();
          const uint64_t round_base =
              static_cast<uint64_t>(slice_item->sieve_round) *
              static_cast<uint64_t>(slice_item->sievesize);
          size_t pushed = 0;
            // collect local absolute offsets for this slice then sort to
            // produce deterministic ascending ordering matching OpenCL
            std::vector<uint32_t> local_abs;
            for (uint32_t idx = start; idx < end; ++idx) {
              const uint32_t local_offset = gpu_offsets[idx];
              const uint64_t absolute =
                  round_base + static_cast<uint64_t>(local_offset);
              if (absolute > std::numeric_limits<uint32_t>::max())
                continue;
              local_abs.push_back(static_cast<uint32_t>(absolute));
            }
            if (!local_abs.empty()) {
              std::sort(local_abs.begin(), local_abs.end());
              for (uint32_t v : local_abs) {
                gpu_absolute_offsets.push_back(v);
                pushed++;
              }
            }
          gpu_window_slices[batch_index].count = pushed;
          if (pushed > 0) {
            gpu_window_slices[batch_index].offsets =
                gpu_absolute_offsets.data() + slice_offset;
            gpu_window_slices[batch_index].absolute = true;
          }
        }
      } else {
        if (use_gpu_sieve) {
          gpu_window_slices.assign(batch.size(), PrototypeSlice{});
          gpu_absolute_offsets.clear();
        }
      }
    }
#endif

    for (size_t idx = 0; idx < batch.size(); ++idx) {
      unique_ptr<SieveItem> &sitem_ptr = batch[idx];
      if (!sitem_ptr)
        continue;
      SieveItem *sitem = sitem_ptr.get();

      PoW *pow            = sitem->pow;
      sieve_t *sieve      = sitem->sieve;
      sieve_t sievesize   = sitem->sievesize;
      sieve_t sieve_round = sitem->sieve_round;
      mpz_set(mpz_start, sitem->mpz_start);

      sieve_t min_len = batch_min_len[idx];
      sieve_t start        = 0;
      sieve_t i            = 1;

      bool stop_requested = hsieve->should_stop(sitem->hash);
      if (stop_requested && opts->has_extra_vb()) {
        std::ostringstream ss;
        ss << get_time() << "GPU work thread received stop signal at round "
           << sieve_round << " (queue depth " << queue->size() << ")";
        extra_verbose_log(ss.str());
      }

  #ifdef USE_CUDA_BACKEND
      PrototypeSlice slice{};
      if (idx < gpu_window_slices.size())
        slice = gpu_window_slices[idx];
      bool window_use_gpu_offsets =
          use_gpu_sieve && slice.count > 0 && slice.offsets != nullptr;
      const uint32_t *gpu_sieve_offsets = window_use_gpu_offsets ? slice.offsets : NULL;
      size_t gpu_sieve_count = window_use_gpu_offsets ? slice.count : 0;
      size_t gpu_sieve_index = 0;
  #else
      bool window_use_gpu_offsets = false;
      const uint32_t *gpu_sieve_offsets = NULL;
      size_t gpu_sieve_count = 0;
      size_t gpu_sieve_index = 0;
  #endif

      if (sieve_round == 0 && !stop_requested) {
        size_t exported_size;
        uint32_t prime_base[10] = {0};
        mpz_export(prime_base, &exported_size, -1, 4, 0, 0, mpz_start);
        gpu_list->reinit(prime_base, pow->get_target(), pow->get_nonce());
      }

      if (!stop_requested) {
        bool gpu_prime_found = false;
#ifdef USE_CUDA_BACKEND
        if (window_use_gpu_offsets) {
          const bool extra_verbose = Opts::get_instance()->has_extra_vb();
          const uint64_t sieve_base =
              static_cast<uint64_t>(sieve_round) * static_cast<uint64_t>(sievesize);
          const bool offsets_absolute = slice.absolute;
          size_t probe = 0;
          bool logged_mismatch = false;
          for (; probe < gpu_sieve_count; ++probe) {
            const uint32_t raw = gpu_sieve_offsets[probe];
            if (!offsets_absolute && raw >= sievesize)
              continue;

            uint64_t relative = 0;
            sieve_t local_offset = 0;
            if (offsets_absolute) {
              relative = static_cast<uint64_t>(raw);
              if (relative < sieve_base)
                continue;
              const uint64_t delta = relative - sieve_base;
              if (delta >= static_cast<uint64_t>(sievesize))
                continue;
              local_offset = static_cast<sieve_t>(delta);
            } else {
              local_offset = static_cast<sieve_t>(raw);
              relative = sieve_base + static_cast<uint64_t>(local_offset);
            }

            mpz_add_ui(mpz_p, mpz_start, relative);
            mpz_sub_ui(mpz_e, mpz_p, 1);
            mpz_powm(mpz_r, mpz_two, mpz_e, mpz_p);
            if (mpz_cmp_ui(mpz_r, 1) == 0) {
              local_offset += 2;
              start = local_offset + sievesize * sieve_round;
              i = local_offset;
              gpu_prime_found = true;
              break;
            }

            if (extra_verbose && !logged_mismatch) {
              const bool bit_set = is_prime(sieve, static_cast<sieve_t>(local_offset));
              if (!bit_set) {
                std::ostringstream diag;
                diag << get_time() << "GPU slice mismatch: offset " << local_offset
                     << " not marked prime in CPU sieve (round_base=" << sieve_base
                     << ")";
                extra_verbose_log(diag.str());
                logged_mismatch = true;
              }
            }
          }
          if (!gpu_prime_found && Opts::get_instance()->has_extra_vb()) {
            std::ostringstream warn;
            warn << get_time()
                 << "CUDA sieve prototype did not find a Fermat seed in GPU offsets"
                 << " (" << gpu_sieve_count << " candidates)"
                 << " round_base=" << sieve_base;
            const size_t preview = std::min<size_t>(gpu_sieve_count, 8u);
            if (preview > 0) {
              warn << " sample" << (slice.absolute ? " abs" : " rel") << ":";
              for (size_t sample = 0; sample < preview; ++sample) {
                const uint32_t raw_sample = gpu_sieve_offsets[sample];
                const uint64_t abs_sample = slice.absolute
                                                ? static_cast<uint64_t>(raw_sample)
                                                : sieve_base + static_cast<uint64_t>(raw_sample);
                warn << " " << abs_sample;
              }
            }
            extra_verbose_log(warn.str());
          }
        }
        if (!gpu_prime_found && window_use_gpu_offsets) {
          window_use_gpu_offsets = false;
          gpu_sieve_offsets = NULL;
          gpu_sieve_count = 0;
        }
#endif
        if (!gpu_prime_found) {
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
      }

      while (!stop_requested && i < sievesize - min_len) {
        const size_t needed_capacity = static_cast<size_t>(min_len / 2) + 1;
        if (offset_buffer.capacity() < needed_capacity) {
          const size_t new_capacity = std::max(offset_buffer.capacity() * 2,
                                               needed_capacity);
          offset_buffer.reserve(new_capacity);
        }
        offset_buffer.clear();

#ifdef USE_CUDA_BACKEND
        if (window_use_gpu_offsets) {
          const uint64_t sieve_base =
              static_cast<uint64_t>(sieve_round) * static_cast<uint64_t>(sievesize);
          const bool offsets_absolute = slice.absolute;
          const uint64_t window_local_base = i;
          const uint64_t window_local_limit = window_local_base + min_len;

          if (offsets_absolute) {
            const uint64_t window_abs_base = sieve_base + window_local_base;
            const uint64_t window_abs_limit = window_abs_base + min_len;

            while (gpu_sieve_index < gpu_sieve_count &&
                   static_cast<uint64_t>(gpu_sieve_offsets[gpu_sieve_index]) <
                       window_abs_base)
              ++gpu_sieve_index;

            size_t cursor = gpu_sieve_index;
            while (cursor < gpu_sieve_count &&
                   static_cast<uint64_t>(gpu_sieve_offsets[cursor]) < window_abs_limit) {
              const uint64_t relative = static_cast<uint64_t>(gpu_sieve_offsets[cursor]);
              offset_buffer.push_back(static_cast<uint32_t>(relative));
              ++cursor;
            }
            gpu_sieve_index = cursor;
          } else {
            while (gpu_sieve_index < gpu_sieve_count &&
                   gpu_sieve_offsets[gpu_sieve_index] < window_local_base)
              ++gpu_sieve_index;

            size_t cursor = gpu_sieve_index;
            while (cursor < gpu_sieve_count &&
                   gpu_sieve_offsets[cursor] < window_local_limit) {
              const uint64_t relative =
                  sieve_base + static_cast<uint64_t>(gpu_sieve_offsets[cursor]);
              offset_buffer.push_back(static_cast<uint32_t>(relative));
              ++cursor;
            }
            gpu_sieve_index = cursor;
          }
        } else
#endif
        {
          for (sieve_t n = 0; n < min_len; n += 2) {
            if (is_prime(sieve, i + n))
              offset_buffer.push_back(i + n + sievesize * sieve_round);
          }
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
        if (opts->has_extra_vb()) {
          std::ostringstream ss;
          ss << get_time()
             << "GPU work thread clearing queue after stop signal";
          extra_verbose_log(ss.str());
        }
#ifdef USE_CUDA_BACKEND
        deferred_item.reset();
#endif
        queue->clear();
      }
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
    std::ostringstream ss;
    ss << get_time() << "GPU result processing thread started";
    extra_verbose_log(ss.str());
  }

#ifndef WINDOWS
  /* use idle CPU cycles for mining */
  struct sched_param param;
  param.sched_priority = sched_get_priority_min(SCHED_IDLE);
  sched_setscheduler(0, SCHED_IDLE, &param);
#endif

  GPUWorkList *list = (GPUWorkList *) args;
  GPUFermat *fermat = GPUFermat::get_instance();
  GPUFermat::ResultWord *result = fermat->get_results_buffer();
  const bool extra_verbose = Opts::get_instance()->has_extra_vb();

  while (list->running) {
    const uint64_t cycle_start = PoWUtils::gettime_usec();

    uint32_t candidate_count = list->create_candidates();
    if (!candidate_count) {
      if (extra_verbose && list->running) {
        std::ostringstream ss;
        ss << get_time()
           << "GPU result thread idle; no candidates produced this cycle";
        extra_verbose_log(ss.str());
      }
      continue;
    }
    if (!list->running) {
      if (extra_verbose) {
        std::ostringstream ss;
        ss << get_time()
           << "GPU result thread stopping after candidate collection";
        extra_verbose_log(ss.str());
      }
      break;
    }

    fermat->fermat_gpu(candidate_count);
    if (!list->running) {
      if (extra_verbose) {
        std::ostringstream ss;
        ss << get_time()
           << "GPU result thread stopping after fermat_gpu call";
        extra_verbose_log(ss.str());
      }
      break;
    }

    list->parse_results(result);
    if (!list->running) {
      if (extra_verbose) {
        std::ostringstream ss;
        ss << get_time()
           << "GPU result thread stopping after parse_results";
        extra_verbose_log(ss.str());
      }
      break;
    }

    const uint32_t processed = candidate_count;
    *list->tests     += processed;
    *list->cur_tests += processed;

    if (extra_verbose) {
      const uint64_t cycle_end = PoWUtils::gettime_usec();
      const double duration_ms = (cycle_end - cycle_start) / 1000.0;
      std::ostringstream ss;
      ss << get_time() << "GPU cycle processed " << processed
         << " candidates in " << duration_ms << " ms";
      extra_verbose_log(ss.str());
    }
  }

  if (extra_verbose) {
    std::ostringstream ss;
    ss << get_time() << "GPU result processing thread exiting";
    extra_verbose_log(ss.str());
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

  if (result) {
    if (running && Opts::get_instance()->has_extra_vb()) {
      std::ostringstream ss;
      ss << get_time() << "HybridSieve stop requested due to new block hash";
      extra_verbose_log(ss.str());
    }
    stop();
  }
  return result;
}

/* create new work item */
HybridSieve::GPUWorkItem::GPUWorkItem(uint32_t *offsets, uint16_t len, uint16_t min_len, uint32_t start) {

  this->offsets = (uint32_t *) malloc(sizeof(uint32_t) * len);
  memcpy(this->offsets, offsets, sizeof(uint32_t) * len);
  this->end       = 0;
  this->len       = len;
  this->index     = static_cast<int32_t>(len) - 1;
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

void HybridSieve::GPUWorkItem::copy_candidates(uint32_t *dest, uint32_t count) {
  if (count == 0)
    return;

  /* Fast path: copy directly if we have enough items in buffer */
  if (index >= 0 && static_cast<uint32_t>(index + 1) >= count) {
    int32_t local_index = index;
    for (uint32_t n = 0; n < count; ++n)
      dest[n] = offsets[local_index - n];
    index = local_index - static_cast<int32_t>(count);
    return;
  }

  /* Slow path: use pop() which handles negative index */
  for (uint32_t n = 0; n < count; ++n)
    dest[n] = pop();
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
  if (index < 0)
    return 0;
  const int32_t remaining = index + 1;
  return static_cast<uint16_t>(std::min<int32_t>(remaining, std::numeric_limits<uint16_t>::max()));
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
  this->last_cycle_tests = n_tests;
  this->last_cycle_items = 0;
  this->pprocessor    = pprocessor;
  this->sieve         = sieve;
  this->prime_base    = prime_base;
  this->candidates    = candidates;
  this->last_candidate_count = len * n_tests;
  this->tests         = tests;
  this->cur_tests     = cur_tests;
  this->start         = NULL;
  this->end           = NULL;
  this->running       = true;
  Opts *opts_local = Opts::get_instance();
  this->shift = 45; // default
  auto parse_u32_option = [&](const char *flag,
                              const std::string &value,
                              uint32_t min_allowed,
                              uint32_t max_allowed,
                              uint32_t fallback) -> uint32_t {
    if (value.empty())
      return fallback;
    errno = 0;
    char *endptr = nullptr;
    unsigned long long parsed = strtoull(value.c_str(), &endptr, 10);
    const bool bad_format = (endptr == value.c_str()) ||
                            (endptr && *endptr != '\0');
    if (errno == ERANGE || bad_format) {
      log_str(std::string("Invalid value '") + value + "' for " + flag +
                  ", using " + itoa(fallback),
              LOG_W);
      return fallback;
    }
    if (parsed < min_allowed) {
      log_str(std::string("Value '") + value + "' for " + flag +
                  " below minimum " + itoa(min_allowed) + ", clamping",
              LOG_W);
      parsed = min_allowed;
    } else if (parsed > max_allowed) {
      log_str(std::string("Value '") + value + "' for " + flag +
                  " above maximum " + itoa(max_allowed) + ", clamping",
              LOG_W);
      parsed = max_allowed;
    }
    return static_cast<uint32_t>(parsed);
  };

  const uint32_t default_divisor = 6u;
  preferred_launch_divisor = default_divisor;
  if (opts_local && opts_local->has_gpu_launch_divisor())
    preferred_launch_divisor =
        parse_u32_option("--gpu-launch-divisor",
                         opts_local->get_gpu_launch_divisor(),
                         1u,
                         std::numeric_limits<uint32_t>::max(),
                         default_divisor);

  const uint32_t default_wait_ms = 50u;
  preferred_launch_max_wait_ms = default_wait_ms;
  if (opts_local && opts_local->has_gpu_launch_wait_ms())
    preferred_launch_max_wait_ms =
        parse_u32_option("--gpu-launch-wait-ms",
                         opts_local->get_gpu_launch_wait_ms(),
                         0u,
                         std::numeric_limits<uint32_t>::max(),
                         default_wait_ms);
  this->last_batch_megabytes = 0.0;
  this->last_batch_avg_tests = 0;
  this->last_batch_min_tests = 0;
  this->last_batch_stats_valid = false;

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

void HybridSieve::GPUWorkList::rebalance_locked() {
  if (!start || !start->next)
    return;

  GPUWorkItem *best_prev = NULL;
  GPUWorkItem *best_item = start;
  uint16_t best_remaining = start->get_cur_len();
  GPUWorkItem *prev = start;

  for (GPUWorkItem *cur = start->next; cur != NULL; cur = cur->next) {
    const uint16_t cur_remaining = cur->get_cur_len();
    if (cur_remaining < best_remaining ||
        (cur_remaining == best_remaining && cur->get_len() > best_item->get_len())) {
      best_remaining = cur_remaining;
      best_prev = prev;
      best_item = cur;
    }
    prev = cur;
  }

  if (best_item == start)
    return;

  best_prev->next = best_item->next;
  if (best_prev == end)
    end = best_prev;

  best_item->next = start;
  start = best_item;
}

uint32_t HybridSieve::GPUWorkList::preferred_launch_items() const {
  if (len == 0)
    return 1u;
  const uint32_t divisor = std::max<uint32_t>(1u, preferred_launch_divisor);
  uint32_t preferred = len / divisor;
  if (preferred == 0)
    preferred = 1u;
  /* Cap the preferred depth so we don't stall waiting for very deep queues;
   * launch once we have roughly 1/8th of capacity, but never less than 256. */
  const uint32_t cap = std::max<uint32_t>(1u, len / 8u);
  preferred = std::min(preferred, cap);
  preferred = std::max<uint32_t>(256u, preferred);
  preferred = std::min(preferred, len);
  return preferred;
}

/* returns the size of this */
size_t HybridSieve::GPUWorkList::size() {
  MutexGuard guard(access_mutex);

  size_t size = sizeof(GPUWorkList) + sizeof(uint32_t) * 10 * len * n_tests;

  for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next) 
    size += sizeof(GPUWorkItem) + sizeof(uint32_t) * cur->get_len();

  return size;
}

/* returns the average length*/
uint16_t HybridSieve::GPUWorkList::avg_len() {
  MutexGuard guard(access_mutex);

  uint64_t total_len = 0;
  uint64_t count = 0;

  for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next, count++) 
    total_len += cur->get_len();

  return (count > 0) ? total_len / count : 0;
}

/* returns the average length*/
uint16_t HybridSieve::GPUWorkList::avg_cur_len() {
  MutexGuard guard(access_mutex);

  uint64_t total_len = 0;
  uint64_t count = 0;

  for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next, count++) 
    total_len += cur->get_cur_len();

  return (count > 0) ? total_len / count : 0;
}

/* returns the min length*/
uint16_t HybridSieve::GPUWorkList::min_cur_len() {
  MutexGuard guard(access_mutex);

  uint16_t min = UINT16_MAX;
  bool has_positive = false;
  for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next) {
    const uint16_t cur_len_val = cur->get_cur_len();
    if (cur_len_val == 0)
      continue;
    has_positive = true;
    if (cur_len_val < min)
      min = cur_len_val;
  }

  if (!has_positive)
    return 0;
  return min;
}

/* reinits this */
void HybridSieve::GPUWorkList::reinit(uint32_t prime_base[10], uint64_t target, uint32_t nonce) {

  MutexGuard guard(access_mutex);

  clear();
  this->target = target;
  this->nonce = nonce;
  this->last_cycle_tests = n_tests;
  this->last_candidate_count = len * n_tests;
  memcpy(this->prime_base, prime_base, sizeof(uint32_t) * 10);
  /* extra-verbose: print host prime_base words for comparison with device dump */
  if (Opts::get_instance() && Opts::get_instance()->has_extra_vb()) {
    std::ostringstream ss;
    ss << get_time() << " host-prime_base:";
    for (int i = 0; i < 10; ++i) {
      ss << " " << std::hex << std::setw(8) << std::setfill('0') << prime_base[i];
    }
    ss << std::dec;
    extra_verbose_log(ss.str());
  }

  pthread_cond_signal(&notfull_cond);
}

/* returns the number of candidates */
uint32_t HybridSieve::GPUWorkList::n_candidates() { 
  return last_candidate_count; 
}

uint32_t HybridSieve::GPUWorkList::queued_items() {
  MutexGuard guard(access_mutex);
  return cur_len;
}

uint32_t HybridSieve::GPUWorkList::capacity() const {
  return len;
}

void HybridSieve::GPUWorkList::stop() {
  MutexGuard guard(access_mutex);
  running = false;
  pthread_cond_broadcast(&full_cond);
  pthread_cond_broadcast(&notfull_cond);
}

/* add a item to the list */
void HybridSieve::GPUWorkList::add(GPUWorkItem *item) {
  
  MutexGuard guard(access_mutex);

  bool logged_full_wait = false;
  while (cur_len >= len) {
    if (extra_verbose && !logged_full_wait) {
      std::ostringstream ss;
      ss << get_time() << "GPU queue full " << cur_len << "/" << len
         << " items; waiting for space";
      extra_verbose_log(ss.str());
      logged_full_wait = true;
    }
    pthread_cond_wait(&notfull_cond, &access_mutex);
  }

  const bool was_empty = (cur_len == 0);

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
  
  if (was_empty)
  pthread_cond_signal(&full_cond);
}

/* creates the candidate array to process */
uint32_t HybridSieve::GPUWorkList::create_candidates() {

  MutexGuard guard(access_mutex);
  bool waited_for_work = false;
  bool logged_wait = false;
  while (running && cur_len == 0) {
    if (extra_verbose && !logged_wait) {
      std::ostringstream ss;
      ss << get_time()
         << "GPU results thread waiting for work (queue empty)";
      extra_verbose_log(ss.str());
      logged_wait = true;
    }
    waited_for_work = true;
    pthread_cond_wait(&full_cond, &access_mutex);
  }

  if (waited_for_work && extra_verbose && running && cur_len > 0) {
    std::ostringstream ss;
    ss << get_time() << "GPU results thread resumed with " << cur_len
       << " queued items";
    extra_verbose_log(ss.str());
  }

  if (!running || cur_len == 0) {
    if (extra_verbose) {
      std::ostringstream ss;
      ss << get_time()
         << "GPU results thread exiting wait because "
         << (running ? "queue stayed empty" : "running flag cleared");
      extra_verbose_log(ss.str());
    }
    return 0;
  }

  const uint32_t desired_items = preferred_launch_items();
  bool logged_batch_wait = false;
  const uint64_t wait_timeout_us = 5000;
  const uint64_t max_wait_us =
      static_cast<uint64_t>(preferred_launch_max_wait_ms) * 1000ULL;
  const bool allow_batch_wait = (max_wait_us > 0);
  if (cur_len > 0 && cur_len < desired_items) {
    if (allow_batch_wait) {
      const uint64_t wait_start_us = PoWUtils::gettime_usec();
      const uint64_t wait_deadline_us = wait_start_us + max_wait_us;
      while (running && cur_len > 0 && cur_len < desired_items) {
        if (extra_verbose && !logged_batch_wait) {
          std::ostringstream ss;
          ss << get_time() << "GPU queue depth " << cur_len << "/" << len
             << " below preferred " << desired_items
             << "; waiting to batch";
          extra_verbose_log(ss.str());
          logged_batch_wait = true;
        }

        const uint64_t now_us = PoWUtils::gettime_usec();
        if (now_us >= wait_deadline_us)
          break;
        const uint64_t remaining_us = wait_deadline_us - now_us;
        const uint64_t slice_us = std::min(wait_timeout_us, remaining_us);
        if (slice_us == 0)
          break;

        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        uint64_t nsec = ts.tv_nsec + slice_us * 1000ULL;
        ts.tv_sec += nsec / 1000000000ULL;
        ts.tv_nsec = nsec % 1000000000ULL;
        pthread_cond_timedwait(&full_cond, &access_mutex, &ts);
      }

      if (extra_verbose && logged_batch_wait && cur_len > 0) {
        std::ostringstream ss;
        if (cur_len >= desired_items) {
          ss << get_time() << "GPU queue depth reached " << cur_len
             << " items; resuming";
        } else {
          const double waited_ms =
              static_cast<double>(PoWUtils::gettime_usec() - wait_start_us) /
              1000.0;
          ss << get_time() << "GPU batch wait timed out after "
             << std::fixed << std::setprecision(3) << waited_ms
             << " ms with depth " << cur_len << "/" << len
             << "; running partial batch";
        }
        extra_verbose_log(ss.str());
      }
    } else if (extra_verbose) {
      std::ostringstream ss;
      ss << get_time() << "GPU queue depth " << cur_len << "/" << len
         << " below preferred " << desired_items
         << "; launch wait disabled";
      extra_verbose_log(ss.str());
    }
  }

  if (!running || cur_len == 0) {
    if (extra_verbose && logged_batch_wait) {
      std::ostringstream ss;
      ss << get_time()
         << "GPU results thread abandoning batch wait because "
         << (running ? "queue drained" : "running flag cleared");
      extra_verbose_log(ss.str());
    }
    return 0;
  }

  const uint32_t cycle_items = cur_len;

  if (cycle_items > 1)
    rebalance_locked();

#ifdef DEBUG_FAST
  this->check = get_xor(); 
#endif

  const uint32_t current_tests = n_tests;
  last_cycle_tests = current_tests;
  last_cycle_items = cycle_items;
  uint32_t i = 0;
  if (extra_verbose) {
    size_t bytes = sizeof(GPUWorkList) + sizeof(uint32_t) * 10 * len * n_tests;
    uint64_t sum_remaining = 0;
    uint32_t active_count = 0;
    uint16_t min_remaining = UINT16_MAX;
    for (GPUWorkItem *cur = start; cur != NULL; cur = cur->next) {
      bytes += sizeof(GPUWorkItem) + sizeof(uint32_t) * cur->get_len();
      const uint16_t remaining = cur->get_cur_len();
      if (remaining > 0) {
        sum_remaining += remaining;
        active_count++;
        if (remaining < min_remaining)
          min_remaining = remaining;
      }
    }
    if (active_count > 0) {
      const uint16_t avg_remaining = static_cast<uint16_t>(sum_remaining / active_count);
      last_batch_megabytes = bytes / 1048576.0;
      last_batch_avg_tests = avg_remaining;
      last_batch_min_tests = min_remaining;
      last_batch_stats_valid = true;
    } else {
      last_batch_stats_valid = false;
    }
  } else {
    last_batch_stats_valid = false;
  }

  /* Select items (allow variable per-item counts). We'll try to fill up to
   * cycle_items but accept items with fewer than current_tests remaining.
   * Build per-item counts and a packed candidates buffer. */
  last_cycle_item_list.clear();
  last_cycle_item_counts.clear();
  last_cycle_item_list.reserve(cycle_items);
  last_cycle_item_counts.reserve(cycle_items);

  uint32_t packed = 0;
  GPUFermat *fermat_ptr = GPUFermat::get_instance();
  const uint32_t max_candidates = fermat_ptr ? fermat_ptr->get_elements_num() : 512u;
  for (GPUWorkItem *cur = start; cur != NULL && last_cycle_item_list.size() < cycle_items; cur = cur->next) {
    uint32_t avail = cur->get_cur_len();
    if (avail == 0)
      continue;
    uint32_t take = std::min<uint32_t>(avail, current_tests);
    if (packed + take > max_candidates)
      break; /* respect GPU buffer */
    last_cycle_item_list.push_back(cur);
    last_cycle_item_counts.push_back(take);
    packed += take;
  }

  /* Align packed total to the device/kernel block size when possible to
   * improve efficiency (e.g. CUDA block size). We trim from the end so
   * earlier queued items remain prioritized. If the block size is larger
   * than the packed total, keep the partial batch (avoid dropping everything). */
  if (fermat_ptr) {
    const uint32_t block = fermat_ptr->get_block_size();
    if (block > 1) {
      uint32_t aligned = (packed / block) * block;
      if (aligned > 0 && aligned < packed) {
        uint32_t reduce = packed - aligned;
        while (reduce > 0 && !last_cycle_item_counts.empty()) {
          uint32_t &back = last_cycle_item_counts.back();
          if (back > reduce) {
            back -= reduce;
            packed -= reduce;
            reduce = 0;
          } else {
            reduce -= back;
            packed -= back;
            last_cycle_item_counts.pop_back();
            last_cycle_item_list.pop_back();
          }
        }
      }
    }
  }

  const uint32_t actual_items = static_cast<uint32_t>(last_cycle_item_list.size());
  last_cycle_items = actual_items;
  last_candidate_count = packed;

  /* copy candidates packed (variable counts per item) */
  i = 0;
  for (uint32_t idx = 0; idx < actual_items; ++idx) {
    uint32_t take = last_cycle_item_counts[idx];
    last_cycle_item_list[idx]->copy_candidates(candidates + i, take);
    i += take;
  }

  if (extra_verbose) {
    std::ostringstream ss;
    ss << get_time() << "GPU queue depth " << cycle_items << "/" << len
       << " items using " << current_tests << " tests";
    if (waited_for_work)
      ss << " after wait";
    if (cycle_items < len)
      ss << " (partial batch)";
    extra_verbose_log(ss.str());
  }

  /* Extra-verbose diagnostic: print first few packed candidate offsets and
   * the host-side reconstructed limbs (LSW and top limb) to aid debugging. */
  if (extra_verbose && packed > 0) {
    const uint32_t *prime_base_words = prime_base;
    const uint32_t sample_count = std::min<uint32_t>(8u, packed);
    std::ostringstream dss;
    dss << get_time() << " GPU-packed-samples:";
    for (uint32_t s = 0; s < sample_count; ++s) {
      uint32_t off = candidates[s];
      uint64_t lsw_sum = static_cast<uint64_t>(prime_base_words[0]) + off;
      uint32_t lsw = static_cast<uint32_t>(lsw_sum & 0xffffffffu);
      /* reconstruct top limb by propagating carry up */
      uint64_t carry = off;
      uint32_t top = 0;
      for (unsigned li = 0; li < 10u; ++li) {
        uint64_t sum = static_cast<uint64_t>(prime_base_words[li]) + carry;
        top = static_cast<uint32_t>(sum & 0xffffffffu);
        carry = sum >> 32;
      }
      dss << " (off=" << off << ",lsw=0x" << std::hex << lsw << std::dec
          << ",top=0x" << std::hex << top << std::dec << ")";
    }
    extra_verbose_log(dss.str());
  }

  return last_candidate_count;
}

/* parse the gpu results */
void HybridSieve::GPUWorkList::parse_results(
  const GPUFermat::ResultWord *results) {
  pthread_mutex_lock(&access_mutex);

#ifdef DEBUG_FAST
  if (check != get_xor()) 
    cout << "[DD] GPUWorkItems CHANGED!!!!  " << check << " == " << get_xor() << endl;
#endif
  
  uint32_t i = 0;
  const uint32_t cycle_tests = last_cycle_tests;
  const uint32_t cycle_items = last_cycle_items;
  const bool extra_verbose = Opts::get_instance()->has_extra_vb();
  uint32_t processed_items = 0;
  if (cycle_items == 0) {
    pthread_mutex_unlock(&access_mutex);
    return;
  }
  GPUWorkItem *del = NULL;
  if (extra_verbose) {
    GPUFermat *fermat_ptr = GPUFermat::get_instance();
    size_t rw_size = sizeof(GPUFermat::ResultWord);
    unsigned elements = fermat_ptr ? fermat_ptr->get_elements_num() : 0u;
    unsigned block = fermat_ptr ? fermat_ptr->get_block_size() : 0u;
    unsigned reported_rw = fermat_ptr ? fermat_ptr->get_result_word_size() : 0u;
    if (reported_rw != static_cast<unsigned>(rw_size)) {
      std::ostringstream ss_err;
      ss_err << get_time() << " ERROR: ResultWord size mismatch: compile-time=" << rw_size
             << " runtime_buffer_element_size=" << reported_rw
             << " -- aborting parse_results to avoid mis-interpretation";
      extra_verbose_log(ss_err.str());
      pthread_mutex_unlock(&access_mutex);
      return;
    }
    std::ostringstream ss0;
    ss0 << get_time() << " resultWordSize=" << rw_size
      << " elementsNum=" << elements
      << " block_size=" << block
      << " last_candidate_count=" << last_candidate_count;
    extra_verbose_log(ss0.str());
    const uint32_t sample = std::min<uint32_t>(32u, cycle_tests * cycle_items);
    std::ostringstream ss;
    ss << get_time() << " parse_results sample: tests=" << cycle_tests
       << " items=" << cycle_items << " sample=" << sample;
    extra_verbose_log(ss.str());
    std::ostringstream ss2;
    ss2 << get_time() << " results[0.." << (sample ? sample - 1 : 0) << "]: ";
    for (uint32_t k = 0; k < sample; ++k) {
      ss2 << static_cast<unsigned int>(results[k]) << ' ';
    }
    extra_verbose_log(ss2.str());
    std::ostringstream ss3;
    ss3 << get_time() << " candidates[0.." << (sample ? sample - 1 : 0) << "]: ";
    for (uint32_t k = 0; k < sample; ++k) {
      ss3 << candidates[k] << ' ';
    }
    extra_verbose_log(ss3.str());

    /* Count non-zero results across the full returned batch and sample some */
    uint32_t total_nonzero = 0;
    std::vector<uint32_t> nonzero_indices;
    const uint32_t full_count = last_candidate_count;
    for (uint32_t k = 0; k < full_count; ++k) {
      if (results[k]) {
        ++total_nonzero;
        if (nonzero_indices.size() < 8)
          nonzero_indices.push_back(k);
      }
    }
    std::ostringstream ss4;
    ss4 << get_time() << " results_nonzero_count=" << total_nonzero;
    if (!nonzero_indices.empty()) {
      ss4 << " samples:";
      for (uint32_t idx : nonzero_indices)
        ss4 << " (i=" << idx << ",v=" << static_cast<unsigned int>(results[idx]) << ",c=" << candidates[idx] << ")";
    }
    extra_verbose_log(ss4.str());
    if (extra_verbose && last_candidate_count > 0) {
      GPUFermat *fermat_ptr = GPUFermat::get_instance();
      if (fermat_ptr) {
        if (total_nonzero == 0) {
          const uint32_t sample_count = std::min<uint32_t>(8u, last_candidate_count);
          std::vector<uint32_t> sample_indices(sample_count);
          for (uint32_t s = 0; s < sample_count; ++s)
            sample_indices[s] = s; /* inspect first N packed indices */
          fermat_ptr->dump_device_samples(sample_indices.data(), sample_count);
        } else {
          /* dump device-side limbs for some positive results to compare */
          const uint32_t sample_count = std::min<uint32_t>(static_cast<uint32_t>(nonzero_indices.size()), 8u);
          fermat_ptr->dump_device_samples(nonzero_indices.data(), sample_count);
        }
      }
    }
  }
  size_t sel_idx = 0;
  uint32_t verify_checked = 0;
  for (GPUWorkItem *cur = start, *prev = NULL; cur != NULL; cur = cur->next) {

    if (processed_items >= cycle_items)
      break;

    if (del != NULL)
      delete del;

    uint32_t this_count = 0;
    if (sel_idx < last_cycle_item_list.size() && cur == last_cycle_item_list[sel_idx]) {
      this_count = last_cycle_item_counts[sel_idx];
      /* process this item's results */
      for (uint32_t n = 0; n < this_count; n++) {
        if (results[i + n]) {
          /* optional CPU verification of GPU-positive (limited samples) */
          if (extra_verbose && verify_checked < 8) {
            mpz_import(mpz_hash, 10, -1, 4, 0, 0, prime_base);
            mpz_div_2exp(mpz_hash, mpz_hash, this->shift);
            mpz_set_ui(mpz_adder, candidates[i + n]);
            mpz_add(mpz_hash, mpz_hash, mpz_adder);
            int cpu_res = mpz_probab_prime_p(mpz_hash, 25);
            std::ostringstream vss;
            vss << get_time() << " GPU-pos verify: idx=" << (i + n)
                << " candidate=" << candidates[i + n]
                << " cpu_probab_prime_p=" << cpu_res;
            extra_verbose_log(vss.str());
              if (cpu_res == 0) {
                GPUFermat *fermat_ptr = GPUFermat::get_instance();
                if (fermat_ptr) {
                  uint32_t sample_idx = i + n;
                  fermat_ptr->dump_device_samples(&sample_idx, 1);
                }
              }
            verify_checked++;
          }
#ifndef DEBUG_BASIC
          cur->set_prime(this_count - n);
#else
          cur->set_prime(this_count - n, prime_base);
#endif

#ifdef DEBUG_SLOW
          mpz_import(mpz_hash, 10, -1, 4, 0, 0, prime_base);
          mpz_add_ui(mpz_hash, mpz_hash, cur->get_prime(this_count - n));
          if (!mpz_probab_prime_p(mpz_hash, 25)) {
            cout << "[DD] in parse_results: prime test failed!!! i: " << i;
            cout << " n: " << n << " n_tests: " << this_count << endl;
          }
#endif
        }
#ifdef DEBUG_SLOW
        else {
          mpz_import(mpz_hash, 10, -1, 4, 0, 0, prime_base);
          mpz_add_ui(mpz_hash, mpz_hash, cur->get_prime(this_count - n));
          if (mpz_probab_prime_p(mpz_hash, 25)) {
            cout << "[DD] in parse_results: composite test failed!!! i: ";
            cout << i << " n: " << n << " n_tests: " << this_count << endl;
          }
        }

        if (candidates[i + n] != cur->get_prime(this_count - n)) {
          cout << "[DD] candidates[" << i + n << "] = " << candidates[i + n];
          cout << " != " << cur->get_prime(this_count - n) << endl;
        }
#endif
      }

      i += this_count;
      ++sel_idx;

    } /* end if selected item */

    /* Check if item should be removed: skip/valid OR fully exhausted.
     * Exhausted items (get_cur_len() == 0) with unset start field can't
     * be removed via skip/valid checks, so we explicitly cleanup zombie items. */
    const bool should_remove = cur->skip() || cur->valid() || cur->get_cur_len() == 0;

    if (should_remove) {
       
       if (cur->valid()) {
#ifdef DEBUG_FAST
         if (prev != NULL)  
           prev->print(prime_base);   
         cur->print(prime_base);     
#endif
         /* Count accepted gaps for GPU stats so gaps/s is non-zero on CUDA. */
         if (sieve)
           sieve->increment_gap_counters(1);
         /* For CUDA builds, optionally perform an extra-verbose CPU check to avoid
            submitting obvious non-primes caused by device/host mismatches. This
            runs only when extra_verbose is enabled and only on CUDA backend. */
#ifdef USE_CUDA_BACKEND
         if (extra_verbose) {
           mpz_import(mpz_hash, 10, -1, 4, 0, 0, prime_base);
           mpz_div_2exp(mpz_hash, mpz_hash, this->shift);
           mpz_set_ui(mpz_adder, cur->get_start());
           mpz_add(mpz_hash, mpz_hash, mpz_adder);
           int cpu_res = mpz_probab_prime_p(mpz_hash, 25);
           std::ostringstream ss_verify;
           ss_verify << get_time() << " submit-cpu-verify (extra-verbose, CUDA): offset=" << cur->get_start()
                     << " cpu_probab_prime_p=" << cpu_res;
           extra_verbose_log(ss_verify.str());
         }
         /* Always submit CUDA positives; extra-verbose only logs the CPU check. */
         submit(cur->get_start());
#else
         submit(cur->get_start());
#endif
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

    /* only count processed items (selected ones) */
    if (sel_idx > 0 && sel_idx <= last_cycle_item_list.size() && cur == last_cycle_item_list[sel_idx-1])
      processed_items++;
  }

  if (del != NULL)
    delete del;

  last_cycle_items = 0;

  const double logged_mb = last_batch_megabytes;
  const uint16_t logged_avg = last_batch_avg_tests;
  const uint16_t logged_min = last_batch_min_tests;
  const bool logged_valid = last_batch_stats_valid;

  pthread_cond_signal(&notfull_cond);
  pthread_mutex_unlock(&access_mutex);
  
  /* Yield CPU after releasing queue items to give work thread immediate opportunity
   * to populate queue. With -e flag, file I/O below naturally provides this delay.
   * Without -e, explicit yield prevents results thread from monopolizing CPU. */
  sched_yield();

  if (extra_verbose && logged_valid) {
    stringstream ss;
    ss << get_time() << "GPU-Items: " << setprecision(3) << logged_mb;
    ss << " MB  avg: " << setw(3) << logged_avg << " tests  min: ";
    ss << setw(3) << logged_min << " tests" << endl;

    std::string out = ss.str();
    if (isatty(fileno(stdout))) {
      const char *green = "\x1b[32m";
      const char *reset = "\x1b[0m";
      pthread_mutex_lock(&io_mutex);
      std::cout << green << out << reset;
      pthread_mutex_unlock(&io_mutex);
    } else {
      pthread_mutex_lock(&io_mutex);
      std::cout << out;
      pthread_mutex_unlock(&io_mutex);
    }
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
  mpz_div_2exp(mpz_hash, mpz_hash, this->shift);
  mpz_set_ui(mpz_adder, offset);

  static uint64_t submit_attempts = 0;
  ++submit_attempts;
  if (extra_verbose) {
    const unsigned long hash_low = static_cast<unsigned long>(mpz_get_ui(mpz_hash) & 0xFFFFFFFFu);
    const unsigned long adder_low = static_cast<unsigned long>(mpz_get_ui(mpz_adder) & 0xFFFFFFFFu);
    const uint64_t fingerprint = (static_cast<uint64_t>(hash_low) << 32) ^ static_cast<uint64_t>(adder_low);
    std::ostringstream ss;
    ss << get_time() << " submit_attempt#" << submit_attempts << " offset=" << offset;
    ss << " mpz_hash_low=" << hash_low;
    ss << " mpz_adder_low=" << adder_low;
    ss << " fingerprint=0x" << std::hex << fingerprint << std::dec;
    extra_verbose_log(ss.str());
  }

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

  PoW share_pow(mpz_hash, this->shift, mpz_adder, target, nonce);

  if (extra_verbose) {
    uint64_t share_diff = share_pow.difficulty();
    std::ostringstream ss_submit;
    double ratio = 0.0;
    if (target != 0) ratio = ((double) share_diff) / ((double) target);
    /* export full mpz values as hex strings for exact comparison */
    char *hex_hash = mpz_get_str(NULL, 16, mpz_hash);
    char *hex_adder = mpz_get_str(NULL, 16, mpz_adder);
    ss_submit << get_time() << " submit-candidate: offset=" << offset
              << " share_difficulty=" << share_diff
              << " (0x" << std::hex << share_diff << std::dec << ")"
              << " target=" << target
              << " (0x" << std::hex << target << std::dec << ")"
              << " ratio=" << std::fixed << std::setprecision(6) << ratio
              << " mpz_hash=0x" << hex_hash
              << " mpz_adder=0x" << hex_adder;
    free(hex_hash);
    free(hex_adder);
    extra_verbose_log(ss_submit.str());
  }

  if (share_pow.valid()) {
#ifdef DEBUG_FAST
    valid++;
    cout << "[DD] PoW valid (" << valid << ")\n";
#endif 

    /* stop calculating if processor said so */
    if (pprocessor->process(&share_pow)) {
      if (extra_verbose) {
        std::ostringstream ss;
        ss << get_time() << " submit accepted: offset=" << offset;
        extra_verbose_log(ss.str());
      }
      sieve->stop();
      return true;
    }
  } 
#ifdef DEBUG_BASIC
  else if (extra_verbose) {
    std::ostringstream ss;
    ss << get_time() << " submit rejected by PoW check: offset=" << offset;
    extra_verbose_log(ss.str());
  }
#endif
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
  last_cycle_items = 0;
  last_candidate_count = 0;
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
