/**
 * Header file for a prime gap sieve based on the chinese remainder theorem
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
#include <gmp.h>
#include <pthread.h>
#include "ChineseSet.h"
#include "GapCandidate.h"
#include "PoWCore/src/PoW.h"
#include "PoWCore/src/Sieve.h"
#include "utils.h"
#include <vector>
#include <atomic>
#include <openssl/sha.h>


class ChineseSieve : public Sieve {
  
  private :

    /* the ChineseSet used in these */
    ChineseSet *cset;

    /* the prime reminder based on the primorial */
    sieve_t *primorial_reminder;

    /* the reminders based on the start */
    sieve_t *start_reminder;

    /* the init status of the CRT in percent */
    double crt_status;

    /* indicates that the primes and starts where loaded */
    bool primes_loaded;

    /* calculates the primorial reminders */
    void calc_primorial_reminder();

    /* calculates the start reminders */
    void calc_start_reminder();

    /* recalc sarts */
    void recalc_starts();

    /* calculate the avg sieve candidates */
    void calc_avg_prime_candidates();

    /* the number of average candidates in the sieve */
    double avg_prime_candidates;

    /* returns the theoreticaly speed increas factor for a given merit */
    double get_speed_factor(double merit, sieve_t n_candidates);

    /* stores the found gaps in an shared heap */
    static vector<GapCandidate *> gaps;

    /* pooled GapCandidates to avoid repeated alloc/free */
    static vector<GapCandidate *> gap_pool;
    static uint32_t gap_pool_limit;
    
    /* calculated gaps since the last share */
    static sieve_t gaps_since_share;

    /* candidate submission ratio tracking */
    static uint64_t total_candidates_tested;
    static uint64_t total_candidates_submitted;

    /* sync mutex */
    static pthread_mutex_t mutex;

    /* condition variable to signal gap availability */
    static pthread_cond_t gap_cv;

    /* maximum queued gaps before sieving pauses */
    static uint32_t gap_queue_limit;

    /* the maximum possible merit with the crt */
    double max_merit;

    /* the current merit */
    static double cur_merit;

    /* gap pool helpers to reduce contention */
    static GapCandidate *acquire_gap_from_pool();
    static bool release_gap_to_pool(GapCandidate *gap);
    static void flush_local_gap_pool();
    
    /* check if we should stop sieving */
    bool should_stop(uint8_t hash[SHA256_DIGEST_LENGTH]);

    /* indicates that the sieve should stop calculating */
    bool running;

    /* finds the prevoius prime for a given mpz value (if src is not a prime) */
    void mpz_previous_prime(mpz_t mpz_dst, mpz_t mpz_src);

    /* primality testing */
    mpz_t mpz_e, mpz_r, mpz_two;

    /* random */
    rand128_t *rand; 
    
    /* optimization: batch GMP updates and log caching */
    sieve_t gmp_batch_counter;
    double cached_log_start;
    sieve_t gmp_batch_size;  /* OPT #10: Configurable batch size (default 256) */
    static const uint32_t BLOCK_CHECK_FREQUENCY = 1024u;
    
    /* optimization: queue-depth aware sieve tuning */
    static const uint32_t QUEUE_PRESSURE_THRESHOLD = 40000u;
    static const uint32_t QUEUE_CRITICAL_THRESHOLD = 50000u;
    double queue_pressure_factor;

    /**
     * OPT #10: Batched GMP Updates with Hybrid Arithmetic
     * Fast modular state for batch processing without GMP overhead
     */
    struct FastModState {
        uint64_t *primorial_mod;  // primorial % prime[i] for each prime
        uint64_t *start_mod;       // current (start % prime[i]) for each prime
        uint64_t allocated_size;   // number of allocated entries
        
        FastModState() : primorial_mod(nullptr), start_mod(nullptr), allocated_size(0) {}
        
        void allocate(uint64_t n_primes) {
            if (allocated_size >= n_primes) return;
            if (primorial_mod) delete[] primorial_mod;
            if (start_mod) delete[] start_mod;
            primorial_mod = new uint64_t[n_primes];
            start_mod = new uint64_t[n_primes];
            allocated_size = n_primes;
        }
        
        ~FastModState() {
            if (primorial_mod) delete[] primorial_mod;
            if (start_mod) delete[] start_mod;
        }
    };
    
    FastModState fast_mod_state;
    bool use_batched_gmp;  // Enable/disable batched processing
    
    /**
     * Initialize batch state with current GMP values
     * Converts GMP remainders to native uint64_t for fast processing
     */
    void init_batch_state();
    
    /**
     * Update remainders in native arithmetic (no GMP calls)
     * Equivalent to recalc_starts() but using uint64_t
     */
    inline void update_batch_remainders();
    
    /**
     * Synchronize GMP state after batch completes
     * Updates mpz_start and recalculates start_reminder[] from GMP
     */
    void sync_gmp_after_batch(uint64_t batch_count);

    /**
     * Fermat pseudo prime test
     */
    inline bool fermat_test(mpz_t mpz_p);
    
    /**
     * Enhanced multi-base Fermat primality test for accuracy
     */
    inline bool miller_rabin_test(mpz_t mpz_p, int rounds = 2);

    /**
     * OPT #11: Multi-Stage Primality Testing
     * Fast single-base Fermat filter (eliminates ~99% of composites in ~0.2ms)
     */
    inline bool quick_fermat_test(mpz_t mpz_p);
    
    /**
     * Medium confidence primality test (3 rounds Miller-Rabin)
     * Filters ~99% of remaining composites after quick_fermat in ~0.5ms
     */
    inline bool medium_miller_rabin_test(mpz_t mpz_p);
    
    /**
     * Full strength primality test (25 rounds Miller-Rabin)
     * Final verification stage (~2ms, only for likely primes)
     */
    inline bool full_miller_rabin_test(mpz_t mpz_p);

    /* Multi-stage testing control and statistics */
    bool use_multistage_tests;  /* Enable/disable multi-stage optimization */
    
    /* Performance tracking for multi-stage testing */
    static std::atomic<uint64_t> stage1_tests;      /* Total candidates tested */
    static std::atomic<uint64_t> stage1_passed;     /* Passed quick Fermat */
    static std::atomic<uint64_t> stage2_tests;      /* Tested with medium MR */
    static std::atomic<uint64_t> stage2_passed;     /* Passed medium MR */
    static std::atomic<uint64_t> stage3_tests;      /* Tested with full MR */
    static std::atomic<uint64_t> stage3_passed;     /* Confirmed primes */

  public:

    /* reste the sieve */
    static void reset();

    /* get gap list count */
    static uint64_t gaplist_size();

    /* stop the current running sieve */
    void stop();

    /* return the crt status */
    double get_crt_status();

    /* sha256 hash of the previous block */
    static uint8_t hash_prev_block[SHA256_DIGEST_LENGTH];
    
    ChineseSieve(PoWProcessor *processor,
                 uint64_t n_primes, 
                 ChineseSet *set);

    ~ChineseSieve();

    /**
     * scan all gaps form start * primorial to end * primorial 
     * where start = (hash << (log2(primorial) + x) / primorial + 1
     * and   end   ~= 2^x 
     */
    void run_sieve(PoW *pow, uint8_t hash[SHA256_DIGEST_LENGTH]);

    /**
     * process the GapCandidates (allways most promising first)
     */
    void run_fermat();

    /** returns the calulation percent of the next share */
    static double next_share_percent();

};
