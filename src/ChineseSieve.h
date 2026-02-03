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

    /**
     * Fermat pseudo prime test
     */
    inline bool fermat_test(mpz_t mpz_p);

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
