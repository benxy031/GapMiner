/**
 * Header file of Gapcoins Proof of Work calculation unit.
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
#ifndef __HYBRID_SIEVE_H__
#define __HYBRID_SIEVE_H__
#include <inttypes.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <gmp.h>
#include <mpfr.h>
#include <queue>

#include "PoWCore/src/PoW.h"
#include "PoWCore/src/PoWUtils.h"
#include "PoWCore/src/PoWProcessor.h"
#include "PoWCore/src/Sieve.h"
#include "GPUFermat.h"
#include "Opts.h"

using namespace std;

class HybridSieve : public Sieve {

  public :

    /* sha256 hash of the previous block */
    uint8_t hash_prev_block[SHA256_DIGEST_LENGTH];

    /* stop the current running sieve */
    void stop();

    /**
     * create a new HybridSieve
     */
    HybridSieve(PoWProcessor *pprocessor, 
                uint64_t n_primes, 
                uint64_t sievesize,
                uint64_t work_items = 512,
                uint64_t n_tests    = 8,
                uint64_t queue_size = 10);

    ~HybridSieve();

    /** 
     * sieve for the given header hash 
     *
     * returns an adder (within pow)  starting a gap greater than difficulty
     *         or NULL if no such prime was found
     */
   void run_sieve(PoW *pow, 
                  vector<uint8_t> *offset,
                  uint8_t hash[SHA256_DIGEST_LENGTH]);
 
  protected :

    /**
     * calculate for every prime the first
     * index in the sieve which is divisible by that prime
     * (and not divisible by two)
     */
    void calc_muls();

    /* check if we should stop sieving */
    bool should_stop(uint8_t hash[SHA256_DIGEST_LENGTH]);

    /* indicates that the sieve should stop calculating */
    bool running;

    /* the number of work items pushed to the gpu at once */
    uint64_t work_items;

    /* template array for the Fermat candidates */
    uint64_t *candidates_template;
    
    /* one GPU work item (set of prime candidates for a prime gap */
    class GPUWorkItem {
      
      private:

        /* the gap start (this is 0 till it gets set from the prevoius work in the work list*/
        uint32_t start, end;

        /* the first found end */
        uint32_t first_end;

        /* the min gap length */
        uint16_t min_len;

#ifndef DEBUG_BASIC
        /* the prime candidate offsets */
        uint32_t *offsets;
#endif        
       
        /* the length of the offsets arrays */
        int16_t len;
        
        /* the current index */
        int16_t index;

      public:

#ifdef DEBUG_BASIC
        /* public offsets for better debugging */
        uint32_t *offsets;
#endif

        /* the next GPUWorkItem in the list */
        GPUWorkItem *next;

        /* creat new work item */
        GPUWorkItem(uint32_t *offsets, uint16_t len, uint16_t min_len, uint32_t start);

        ~GPUWorkItem();

        /* get the next candidate offset */
        uint32_t pop();

        /* set a number to be prime (i relative to index) 
         * returns true if this can be skipped */
#ifndef DEBUG_BASIC
        void set_prime(int16_t i);
#else
        void set_prime(int16_t i, uint32_t prime_base[10]);

        /* returns the prime at a given index offset i */
        uint32_t get_prime(int32_t i);
#endif

        /* sets the gapstart of this */
        void set_start(uint32_t start);

        /* returns wheter this gap can be skipped */
        bool skip();

        /* returns whether this is a valid gap */
        bool valid();

        /* tells this that it souzld be skiped anyway */
        void mark_skipable();

        /* returns the start offset */
        uint32_t get_start();

        /* returns the end offset */
        uint32_t get_end();

        /* sets the end of this so that 
         * it don't sets the start of the next item */
        void set_end();

        /* returns the number of offsets of this */
        uint16_t get_len();

        /* returns the number of current offsets of this */
        uint16_t get_cur_len();

#ifdef DEBUG_BASIC
        /* simple xor check to validate the items */
        uint32_t get_xor();

        /* prints this */
        void print(uint32_t prime_base[10]);
#endif
    };

    /* a list of GPUWorkItem  */
    class GPUWorkList {
      
      private :

#ifdef DEBUG_BASIC
        /* simple xor check to validate the items */
        uint32_t get_xor();

        /* storage value for the xor check */
        uint32_t check; 
#endif        
        
        /* number of work items */
        uint32_t len, cur_len;
 
        /* number of candidates to test at once */
        uint32_t n_tests;
        uint32_t max_n_tests;
        uint32_t min_n_tests;
        uint32_t active_n_tests;
        uint32_t last_cycle_tests;
        uint32_t last_candidate_count;
 
        /* List start and end */
        GPUWorkItem *start, *end;

        /* the candidates array */
        uint32_t *candidates;

        /* the prime base of this */
        uint32_t *prime_base;

        /* the PoWProcessor */
        PoWProcessor *pprocessor;

        /* the sieve */
        HybridSieve *sieve;
 
        /* synchronization */
        pthread_mutex_t access_mutex;
        pthread_cond_t  notfull_cond;
        pthread_cond_t  full_cond;

        /* reorders awaiting work items by urgency */
        void rebalance_locked();

        /* mpz values */
        mpz_t mpz_hash, mpz_adder;
    
        /* header target */
        uint64_t target;

        /* header nonce */
        uint32_t nonce;

        /* use extra verbose ? */
        bool extra_verbose;

      public : 

        /* the number of test made by the gpu */
        uint64_t *tests, *cur_tests;

#ifdef DEBUG_BASIC
        /* returns the current prime_base of this */
        uint32_t *get_prime_base();
#endif

        /* indecates if this sould continue running */
        bool running;
        
        /* creat a new gpu work list */
        GPUWorkList(uint32_t len, 
                    uint32_t n_tests,
                    PoWProcessor *pprocessor,
                    HybridSieve *sieve,
                    uint32_t *prime_base,
                    uint32_t *candidates,
                    uint64_t *tests,
                    uint64_t *cur_tests);

        ~GPUWorkList();

        /* returns the size of this */
        size_t size();

        /* returns the average length*/
        uint16_t avg_len();

        /* returns the average length*/
        uint16_t avg_cur_len();

        /* returns the min length*/
        uint16_t min_cur_len();

        /* reinits this */
        void reinit(uint32_t prime_base[10], uint64_t target, uint32_t nonce);

        /* returns the nuber of candidates */
        uint32_t n_candidates();

        /* add a item to the list */
        void add(GPUWorkItem *item);

        /* creates the candidate array to process */
        uint32_t create_candidates();

        /* parse the gpu results */
        void parse_results(uint32_t *results);

        /* submits a given offset */
        bool submit(uint32_t offset);

        /* clears the list */
        void clear();

                        private:
                                void adjust_active_tests(uint16_t observed_min, uint16_t observed_avg);
    };

    /* the GPUWorkList of this */
    GPUWorkList *gpu_list;

    /* one set of work items for the GPU */
    class SieveItem {

      public :

        /* the sieve */
        sieve_t *sieve;

        /* candidate size */
        sieve_t sievesize;

        /* min gap length */
        sieve_t min_len;

        /* first prime */
        sieve_t start;

        /* sieve index */
        sieve_t i;

        /* the pow nonce */
        uint32_t nonce;

        /* the pow target difficulty */
        uint64_t target;

        /* hash of the previous block */
        uint8_t hash[SHA256_DIGEST_LENGTH];

        /* the current sieve round */
        sieve_t sieve_round;

        /* the current pow */
        PoW *pow;

        /* the current mpz_start */
        mpz_t mpz_start;
       
        /* create a new SieveItem */
        SieveItem(sieve_t *sieve, 
                  sieve_t sievesize, 
                  sieve_t sieve_round,
                  uint8_t hash[SHA256_DIGEST_LENGTH],
                  mpz_t mpz_start,
                  PoW *pow);
       
        /* destroys a SieveItem */
        ~SieveItem();
    };


    /**
     * a class to store prime chain candidates
     */
    class SieveQueue {

      public :

        /* indecates if this sould continue running */
        bool running;

        /* pointer to the HybridSieve */
        HybridSieve *hsieve;

        /* pps measurment */
        uint64_t *cur_found_primes;
        uint64_t *found_primes;

        /* pointer to the sieve's gpu work list */
        HybridSieve::GPUWorkList *gpu_list;


        SieveQueue(unsigned capacity,
                   HybridSieve *hsieve, 
                   GPUWorkList *gpu_list,
                   uint64_t *cur_found_primes,
                   uint64_t *found_primes);
        ~SieveQueue();

        /* get the size of this */
        size_t size();

        /* indicates that this queue is full */
        bool full();

        /* remove the oldest gpu work */
        SieveItem *pull();

        /* add an new SieveItem */
        void push(SieveItem *work);

        /* clear this */
        void clear();

        unsigned get_capacity() { return capacity; }

      private :

        /* the capacity of this */
        unsigned capacity;

        /* the SieveItem queue */
        queue<SieveItem *> q;

        /* synchronization */
        pthread_mutex_t access_mutex;
        pthread_cond_t  notfull_cond;
        pthread_cond_t  full_cond;
    };

    /* work input for the gpu */
    SieveQueue *sieve_queue;

    /* the gpu thread */
    static void *gpu_work_thread(void *args);

    /* the gpu results processing thread */
    static void *gpu_results_thread(void *args);

    /* thread objects */
    pthread_t gpu_thread;
    pthread_t results_thread;

};
#endif /* __HYBRID_SIEVE_H__ */
#endif /* CPU_ONLY */
