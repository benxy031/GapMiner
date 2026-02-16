/**
 * Implementation of a Gapcoin miner
 *
 * Copyright (C)  2014  The Gapcoin developers  <info@gapcoin.org>
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
#ifndef __STDC_FORMAT_MACROS 
#define __STDC_FORMAT_MACROS 
#endif
#ifndef __STDC_LIMIT_MACROS  
#define __STDC_LIMIT_MACROS  
#endif
#include <pthread.h>
#include <stdlib.h>
#include <sched.h>
#include "Miner.h"
#include "BlockHeader.h"
#include "PoWCore/src/PoWUtils.h"
#include "PoWCore/src/PoW.h"
#include "PoWCore/src/PoWProcessor.h"
#include "ShareProcessor.h"
#include "PoWCore/src/Sieve.h"
#include "Opts.h"
#include "utils.h"
#include "HybridSieve.h"

/* synchronization mutex */
pthread_mutex_t Miner::mutex = PTHREAD_MUTEX_INITIALIZER;

/* create a new miner */
Miner::Miner(uint64_t sieve_size, 
             uint64_t sieve_primes, 
             int n_threads) {

  log_str("creating Miner with " + itoa(n_threads) + " threads", LOG_D);
  this->sieve_size     = sieve_size;
  this->sieve_primes   = sieve_primes;
  this->n_threads      = n_threads;
  this->running        = false;
  this->is_started     = false;
  this->use_chinese    = Opts::get_instance()->has_cset(); 
  this->fermat_threads = 1;
  if (Opts::get_instance()->has_fermat_threads())
    this->fermat_threads = atoi(Opts::get_instance()->get_fermat_threads().c_str());
#ifndef CPU_ONLY  
  this->use_gpu        = Opts::get_instance()->has_use_gpu(); 
#endif  

  threads      = (pthread_t *)   calloc(n_threads, sizeof(pthread_t));
  args         = (ThreadArgs **) calloc(n_threads, sizeof(ThreadArgs *));

#ifndef CPU_ONLY  
  if (use_gpu)
    this->n_threads = n_threads;
#endif    
}              

/* start processing */
void Miner::start(BlockHeader *header) {

  log_str("starting Miner", LOG_D);
  running = true;
  Opts *opts = Opts::get_instance();
#ifndef CPU_ONLY
  bool use_gpu = opts->has_use_gpu(); 

  uint64_t n_tests    = (opts->has_n_tests() ? atoi(opts->get_n_tests().c_str()) : 8);
  uint64_t work_items = (opts->has_work_items() ? atoi(opts->get_work_items().c_str()) : 512);
  uint64_t queue_size = (opts->has_queue_size() ? atoi(opts->get_queue_size().c_str()) : 10);
#endif

  ShareProcessor *share_processor = ShareProcessor::get_processor();
  share_processor->update_header(header);

  if (use_chinese)
    memcpy(ChineseSieve::hash_prev_block, 
           header->hash_prev_block,
           SHA256_DIGEST_LENGTH);


  for (int i = 0; i < n_threads; i++) {

    args[i] = new ThreadArgs(i, 
                             n_threads, 
                             &running, 
                             header);
    
#ifndef CPU_ONLY
    if (use_gpu) {
      args[i]->hsieve = new HybridSieve((PoWProcessor *) share_processor, 
                                        sieve_primes, 
                                        sieve_size,
                                        work_items,
                                        n_tests,
                                        queue_size);

      memcpy(args[i]->hsieve->hash_prev_block, 
             header->hash_prev_block,
             SHA256_DIGEST_LENGTH);
             
    } else {
#endif
      if (use_chinese) {
        
        ChineseSet *cset = new ChineseSet(opts->get_cset().c_str());
        args[i]->csieve = new ChineseSieve((PoWProcessor *) share_processor, 
                                           sieve_primes, 
                                           cset);

      } else {
        args[i]->sieve = new Sieve((PoWProcessor *) share_processor, 
                                   sieve_primes, 
                                   sieve_size);
      }
#ifndef CPU_ONLY
    }
#endif
    
    pthread_create(&threads[i], NULL, miner, (void *) args[i]);

    if (Opts::get_instance()->has_extra_vb()) {
      extra_verbose_log(get_time() + "Worker " + itoa((uint64_t) i) + " started");
    }
  }

  is_started = true;
}

/* delete a miner */
Miner::~Miner() {
  log_str("deleting Miner", LOG_D);
  stop();
}

/* stops all threads and waits until they are finished */
void Miner::stop() {

  log_str("stopping Miner", LOG_D);
#ifndef CPU_ONLY
  bool use_gpu = Opts::get_instance()->has_use_gpu(); 
#endif
  bool use_chinese = Opts::get_instance()->has_cset(); 

  if (running) {
    running = false;
    
    for (int i = 0; i < n_threads; i++) {

#ifndef CPU_ONLY
      if (use_gpu)
        args[i]->hsieve->stop();
#endif

      if (use_chinese)
        args[i]->csieve->stop();

      pthread_join(threads[i], NULL);
      delete args[i]->header;

#ifndef CPU_ONLY
      if (use_gpu)
        delete args[i]->hsieve;
      else {
#endif
        if (use_chinese)
          delete args[i]->csieve;
        else
          delete args[i]->sieve;
#ifndef CPU_ONLY
      }
#endif
    }
  }
}

/* updates the BlockHeader for all threads */
bool Miner::update_header(BlockHeader *header) {
  
  if (!is_started) return false;
  log_str("updating BlockHeader", LOG_D);

#ifndef CPU_ONLY
  bool use_gpu = Opts::get_instance()->has_use_gpu(); 
#endif
  bool use_chinese = Opts::get_instance()->has_cset(); 
  
  if (!running)
    return false;

#ifndef CPU_ONLY
  /* restart sieve  with new header */
  for (int i = 0; use_gpu && i < n_threads; i++) {
      memcpy(args[i]->hsieve->hash_prev_block, 
             header->hash_prev_block,
             SHA256_DIGEST_LENGTH);

      args[i]->hsieve->stop();
  }
#endif

  /* restart sieve  with new header */
  if (use_chinese) {
    memcpy(ChineseSieve::hash_prev_block, 
           header->hash_prev_block,
           SHA256_DIGEST_LENGTH);
    ChineseSieve::reset();
  }

  pthread_mutex_lock(&mutex);
  for (int i = 0; i < n_threads; i++) {

    delete args[i]->header; 
    args[i]->header = header->clone();
    args[i]->header->nonce = i;
  }
  pthread_mutex_unlock(&mutex);

  /* update header of ShareProcessor */
  ShareProcessor::get_processor()->update_header(header);

  return true;
}


/* create a new ThreadArgs */
Miner::ThreadArgs::ThreadArgs(int id, 
                              int n_threads, 
                              bool *running, 
                              BlockHeader *header) {

  this->id            = id;
  this->n_threads     = n_threads;
  this->running       = running;
  this->header        = header->clone();
  this->header->nonce = id;
}

/* a single mining thread */
void *Miner::miner(void *args) {

#ifndef WINDOWS
  /* use idle CPU cycles for mining */
  struct sched_param param;
  param.sched_priority = sched_get_priority_min(SCHED_IDLE);
  sched_setscheduler(0, SCHED_IDLE, &param);
#endif

  
  ThreadArgs *targs = (ThreadArgs *) args;
  log_str("Miner thread " + itoa(targs->id) + " started", LOG_D);
#ifndef CPU_ONLY
  bool use_gpu = Opts::get_instance()->has_use_gpu(); 
#endif
  bool use_chinese = Opts::get_instance()->has_cset(); 
  int fermat_threads = 1;
  if (Opts::get_instance()->has_fermat_threads())
    fermat_threads = atoi(Opts::get_instance()->get_fermat_threads().c_str());

  if (use_chinese && targs->id < fermat_threads) {
    targs->csieve->run_fermat();
    return NULL;
  }
    

  mpz_t mpz_hash;
  mpz_init(mpz_hash);
  uint8_t hash_prev_block[SHA256_DIGEST_LENGTH];

  while (*targs->running) {
    
    pthread_mutex_lock(&mutex);
    targs->header->get_hash(mpz_hash);
    memcpy(hash_prev_block,
           targs->header->hash_prev_block,
           SHA256_DIGEST_LENGTH);
    
    /* hash has to be in range (2^255, 2^256) */
    while (mpz_sizeinbase(mpz_hash, 2) != 256) {
      targs->header->nonce += targs->n_threads;
      targs->header->get_hash(mpz_hash);
    }

    pthread_mutex_unlock(&mutex);

    /* run the sieve */
    PoW pow(mpz_hash, 
            targs->header->shift, 
            NULL, 
            targs->header->target, 
            targs->header->nonce);

#ifndef CPU_ONLY
    if (use_gpu)
      targs->hsieve->run_sieve(&pow, NULL, hash_prev_block);
    else {
#endif
      if (use_chinese)
        targs->csieve->run_sieve(&pow, hash_prev_block);
      else        
        targs->sieve->run_sieve(&pow, NULL);
#ifndef CPU_ONLY
    }
#endif

    pthread_mutex_lock(&mutex);
    targs->header->nonce += targs->n_threads;
    pthread_mutex_unlock(&mutex);
  }
  
  mpz_clear(mpz_hash);

#ifndef CPU_ONLY
  if (use_gpu)
    delete targs->hsieve;
  else {
#endif
    if (use_chinese)
      delete targs->csieve;
    else
      delete targs->sieve;
#ifndef CPU_ONLY
  }
#endif

  log_str("Miner thread " + itoa(targs->id) + " stopped", LOG_D);
  return NULL;
}

/**
 * returns the average primes per seconds
 */
double Miner::avg_primes_per_sec() {
  
  if (!is_started) return 0;
  
  double apps = 0;
  for (int i = 0; i < n_threads; i++)
#ifndef CPU_ONLY
    if (use_gpu)
      apps += args[i]->hsieve->avg_primes_per_sec();
    else {
#endif
      if (use_chinese) 
        apps += args[i]->csieve->avg_primes_per_sec();
      else 
        apps  += args[i]->sieve->avg_primes_per_sec();
#ifndef CPU_ONLY
    }
#endif

  return apps;
}

/**
 * returns the primes per seconds
 */
double Miner::primes_per_sec() {
  
  if (!is_started) return 0;
  
  double pps = 0;
  for (int i = 0; i < n_threads; i++)
#ifndef CPU_ONLY
    if (use_gpu)
      pps += args[i]->hsieve->primes_per_sec();
    else {
#endif    
      if (use_chinese) 
        pps += args[i]->csieve->primes_per_sec();
      else 
        pps += args[i]->sieve->primes_per_sec();
#ifndef CPU_ONLY
    }
#endif

  return pps;
}

/**
 * returns the average primes per seconds
 */
double Miner::avg_gaps_per_second() {
  
  if (!is_started) return 0;
  
  double avg_gaps = 0;
  for (int i = 0; i < n_threads; i++)
#ifndef CPU_ONLY
    if (use_gpu)
      avg_gaps += args[i]->hsieve->avg_gaps_per_second();
    else {
#endif
      if (use_chinese) 
        avg_gaps += args[i]->csieve->avg_gaps_per_second();
      else 
        avg_gaps += args[i]->sieve->avg_gaps_per_second();
#ifndef CPU_ONLY
    }
#endif

  return avg_gaps;
}

/**
 * returns the primes per seconds
 */
double Miner::gaps_per_second() {
  
  if (!is_started) return 0;
  
  double gaps = 0;
  for (int i = 0; i < n_threads; i++)
#ifndef CPU_ONLY
    if (use_gpu)
      gaps += args[i]->hsieve->gaps_per_second();
    else {
#endif    
      if (use_chinese) 
        gaps += args[i]->csieve->gaps_per_second();
      else 
        gaps  += args[i]->sieve->gaps_per_second();
#ifndef CPU_ONLY
    }
#endif

  return gaps;
}

/**
 * returns the average primes per seconds
 */
double Miner::avg_tests_per_second() {
  
  if (!is_started) return 0;
  
  double avg_tests = 0;
  for (int i = 0; i < n_threads; i++)
#ifndef CPU_ONLY
    if (use_gpu)
      avg_tests += args[i]->hsieve->avg_tests_per_second();
    else {
#endif
      if (use_chinese) 
        avg_tests += args[i]->csieve->avg_tests_per_second();
      else 
        avg_tests += args[i]->sieve->avg_tests_per_second();
#ifndef CPU_ONLY
    }
#endif

  return avg_tests;
}

/**
 * returns the primes per seconds
 */
double Miner::tests_per_second() {
  
  if (!is_started) return 0;
  
  double tests = 0;
  for (int i = 0; i < n_threads; i++)
#ifndef CPU_ONLY
    if (use_gpu)
      tests += args[i]->hsieve->tests_per_second();
    else {
#endif    
      if (use_chinese) 
        tests += args[i]->csieve->tests_per_second();
       else 
        tests += args[i]->sieve->tests_per_second();
#ifndef CPU_ONLY
    }
#endif

  return tests;
}

uint64_t Miner::get_target_difficulty() {
  if (!is_started)
    return 0;

  pthread_mutex_lock(&mutex);
  uint64_t target = 0;
  if (n_threads > 0 && args && args[0] && args[0]->header)
    target = args[0]->header->target;
  pthread_mutex_unlock(&mutex);

  return target;
}

/**
 * returns the crt_status
 */
double Miner::get_crt_status() {
  
  if (!is_started) return 0;
  
  double status = 0;
  for (int i = 0; i < fermat_threads; i++)
    if (use_chinese) 
      status += args[i]->csieve->get_crt_status();

  return status / fermat_threads;
}


/**
 * returns the percent we have already calculated form the current share
 */
double Miner::next_share_percent() {
  
  if (!is_started) return 0;

  if (use_chinese) 
    return ChineseSieve::next_share_percent();

  return 0;
}

/**
 * returns whether this is running
 */
bool Miner::started() { return is_started && running; }
