/**
 * Header file of a Gapcoin miner
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
#ifndef __MINER_H__
#define __MINER_H__
#include <pthread.h>
#include "BlockHeader.h"
#include "PoWCore/src/Sieve.h"
#include "HybridSieve.h"
#include "ChineseSieve.h"


class Miner {

  public:

    /* create a new miner */
    Miner(uint64_t sieve_size, 
          uint64_t sieve_primes, 
          int n_threads);

    /* delete a miner */
    ~Miner();

    /* start processing */
    void start(BlockHeader *header);

    /* stops all threads and waits until they are finished */
    void stop();

    /* updates the BlockHeader for all threads */
    bool update_header(BlockHeader *header);

    /**
     * returns the average primes per seconds
     */
    double avg_primes_per_sec();

    /**
     * returns the primes per seconds
     */
    double primes_per_sec();

    /**
     * returns the prime gaps per second
     */
    double gaps_per_second();

    /**
     * returns average the prime gaps per second
     */
    double avg_gaps_per_second();

    /**
     * returns the prime tests per second
     */
    double tests_per_second();

    /**
     * returns average the prime tests per second
     */
    double avg_tests_per_second();

    /* returns the current target difficulty (0 if unavailable) */
    uint64_t get_target_difficulty();

    /* returns a signature for current work template (0 if unavailable) */
    uint64_t get_work_signature();

    /**
     * returns the percent we have already calculated form the current share
     */
    double next_share_percent();

    /**
     * returns whether this is running
     */
    bool started();

    /* return the crt status */
    double get_crt_status();

  private:
    
    /* number of fermat threads */
    int fermat_threads;

    /* sieve size */
    uint64_t sieve_size;

    /* sieve primes */
    uint64_t sieve_primes;

    /* number of threads */
    int n_threads;

    /* indicates if this should run */
    bool running;

    /* indicates if this is started */
    bool is_started;

    /* indicates if we should use Chinese Remainder theorem or not */
    bool use_chinese;

#ifndef CPU_ONLY
    /* indicates if we should use gpu or not */
    bool use_gpu;
#endif    

    /* synchronization mutex */
    static pthread_mutex_t mutex;       

    /* the threads of this */
    pthread_t *threads;
    
    /* thread arguments */
    class ThreadArgs {
      
      public:

        /* id of this thread */
        int id;

        /* number of threads */
        int n_threads;

        /* indicates if this should run */
        bool *running;

        /* the Block header to mine for */
        BlockHeader *header;

        /* the ChineseSieve for this */
        ChineseSieve *csieve;

#ifndef CPU_ONLY
        /* the HybridSieve for this */
        HybridSieve *hsieve;
#endif        

        /* the Sieve for this */
        Sieve *sieve;

        /* create a new ThreadArgs */
        ThreadArgs(int id, 
                   int n_threads,
                   bool *running, 
                   BlockHeader *header);
    };

    /* the thread args of this */
    ThreadArgs **args;

    /* the actual miner thread */
    static void *miner(void *args);
};

#endif /* __MINER_H__ */
