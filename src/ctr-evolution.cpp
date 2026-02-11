/**
 * Implementation of a Chinese Remainder Theorem optimizer
 * using an evolutionary algorithm
 *
 * Copyright (C)  2015  The Gapcoin developers  <info@gapcoin.org>
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
#include <math.h>
#include <unistd.h>
#include <unistd.h>
#include <pthread.h>
#include <iostream>

using namespace std;

#include "ctr-evolution.h"
#include "Opts.h"
#include "BestChinese.h"
#include "ChineseSet.h"
#include "ChineseRemainder.h"
#include "utils.h"
#include "PoWCore/src/PoWUtils.h"
extern "C" {
#include "Evolution/src/evolution.h"
}

/**
 * sieve the first n primes optimal
 */
void set_sieve_optimal(Chinese *c);

/**
 * returns the number of candidates a given Chinese has
 */
inline sieve_t get_candidate_count(Chinese *c);

/**
 * calculates the fitness for one Chinese 
 */
int64_t fitness_chinese(Individual *iv, void *opts);

/**
 * synchronization mutex
 */
static pthread_mutex_t creation_mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * create a new Chinese
 */
void *new_chinese(void *opts) {
  
  CreatOpts *copt  = (CreatOpts *) opts;
  Chinese *chinese = (Chinese *) malloc(sizeof(Chinese));

  mpz_init_set_ui(chinese->mpz_primorial, 1);

  chinese->rand       = new_rand128((uint32_t) PoWUtils::gettime_usec() ^ 
                                    (uint32_t) (PoWUtils::gettime_usec() >> 32) ^
                                    rand());

  /* interpret range as percentage deviation from n_primes */
  const double range_pct = copt->range;
  const sieve_t max_rand = (range_pct > 0.0) ? (sieve_t) (copt->n_primes * (range_pct / 100.0)) : 0;
  const sieve_t min_primes = (copt->fixed_len > 1 ? copt->fixed_len : 1);
  const sieve_t max_primes = copt->n_primes + max_rand;

  if (max_rand) {
    int64_t delta = (int64_t)(rand128(chinese->rand) % (max_rand * 2 + 1)) - (int64_t) max_rand;
    int64_t candidate = (int64_t) copt->n_primes + delta;
    if (candidate < (int64_t) min_primes)
      candidate = min_primes;
    if (candidate > (int64_t) max_primes)
      candidate = max_primes;
    chinese->n_primes = (sieve_t) candidate;
  } else {
    chinese->n_primes = copt->n_primes;
  }

  for (sieve_t i = 0; i < max_primes; i++)
    mpz_mul_ui(chinese->mpz_primorial, chinese->mpz_primorial, first_primes[i]);

  sieve_t max_sievesize = copt->merit * (mpz_log(chinese->mpz_primorial) + log(2) * copt->bits);

  mpz_set_ui(chinese->mpz_primorial, 1);
  for (sieve_t i = 0; i < chinese->n_primes; i++)
    mpz_mul_ui(chinese->mpz_primorial, chinese->mpz_primorial, first_primes[i]);

  chinese->bits      = copt->bits;
  chinese->merit     = copt->merit;
  chinese->sievesize = copt->merit * (mpz_log(chinese->mpz_primorial) + log(2) * copt->bits);
  chinese->byte_size = bound(chinese->sievesize, sizeof(sieve_t) * 8) / 8;
  chinese->word_size = bound(chinese->sievesize, sizeof(sieve_t) * 8) / (8 * sizeof(sieve_t));
  chinese->base      = (sieve_t *) calloc(max_sievesize / 8 + 64, 1);
  chinese->sieve     = (sieve_t *) malloc(max_sievesize / 8 + 64);
  chinese->fixed_len = copt->fixed_len;

  const sieve_t alloc_primes = max_primes + 10;
  chinese->primes    = (sieve_t *) malloc(sizeof(sieve_t) * alloc_primes);
  chinese->offsets   = (sieve_t *) malloc(sizeof(sieve_t) * alloc_primes);

  memcpy(chinese->primes,  first_primes, sizeof(sieve_t) * alloc_primes);
  memset(chinese->offsets, 0, sizeof(sieve_t) * alloc_primes);
  set_sieve_optimal(chinese);

  return (void *) chinese;
}

/**
 * returns the number of candidates a given Chinese have 
 */
inline sieve_t get_candidate_count(Chinese *c) {
  sieve_t ret = 0;

  for (sieve_t i = 0; i < c->word_size; i++)
    ret += popcount(c->sieve[i]);

  return ret;
}

/**
 * print debugging informations about a sieve
 */
void print_sieve(sieve_t *sieve, sieve_t word_size) {
  for (sieve_t w = 0; w < word_size; w++) {
    
    for (sieve_t b = 0; b < sizeof(sieve_t) * 8; b++) {
      cout << ((sieve[w] & (((sieve_t) 1) << b)) >> b);
      if (b % 8 == 7)
        cout << " ";
    }
    cout << endl;
  }
}

/**
 * sieve the first n primes optimal
 */
void set_sieve_optimal(Chinese *c) {

  pthread_mutex_lock(&creation_mutex);

  sieve_t max_gready = MAX_GREADY * 10;
  if (Opts::get_instance()->has_ctr_strength()) {
    sieve_t ctr_strength = atoi(Opts::get_instance()->get_ctr_strength().c_str());

    if (max_gready < ctr_strength)
      max_gready = ctr_strength;
  }

  static sieve_t index = 0;
  BestChinese *start_chinese = new BestChinese(c->n_primes, 
                                               c->merit, 
                                               c->bits, 
                                               100,
                                               rand128_range(c->rand, 
                                                             MAX_GREADY, 
                                                             max_gready),
                                               index,
                                               false);


  pthread_mutex_unlock(&creation_mutex);
  start_chinese->calc_best_residues(false);
  pthread_mutex_lock(&creation_mutex);


  ChineseSet *set = start_chinese->get_best_set();
  mpz_t mpz_start;
  mpz_init(mpz_start);
  mpz_add(mpz_start, set->mpz_primorial, set->mpz_offset);

  for (sieve_t i = 0; i < c->n_primes; i++)
    c->offsets[i] = (first_primes[i] - mpz_tdiv_ui(mpz_start, first_primes[i])) % first_primes[i];


  for (sieve_t i = 0; i < c->fixed_len; i++) {
    
    const sieve_t prime = first_primes[i];
    for (sieve_t p = c->offsets[i]; p < c->sievesize; p += prime)
      set_composite(c->base, p);
  }

  Individual iv;
  iv.iv = c;
  int64_t fittnes = (fitness_chinese(&iv, NULL) * c->sievesize) / 10000000000;
  if (set->n_candidates < (sieve_t) fittnes)
    cout << "Init iv failed: " << set->n_candidates << " < " << fittnes << endl;


  delete start_chinese;
  delete set;
  mpz_clear(mpz_start);
  index++;

  pthread_mutex_unlock(&creation_mutex);
}

/**
 * create a copy of src into dst
 * both need to be initialized
 */
void clone_chinese(void *dst, void *src, void *opt) {
  (void) opt;

  Chinese *c1 = (Chinese *) dst;
  Chinese *c2 = (Chinese *) src;

  if (c1->n_primes != c2->n_primes) {
    c1->n_primes = c2->n_primes;
    c1->sievesize = c2->sievesize;
    c1->byte_size = c2->byte_size;
    c1->word_size = c2->word_size;
    memcpy(c1->base, c2->base, c2->byte_size);
  }

  memcpy(c1->offsets, c2->offsets, c1->n_primes * sizeof(sieve_t));
}

/**
 * frees all rescues of the given Chinese
 */
void free_chinese(void *src, void *opts) {
  (void) opts;
  Chinese *c = (Chinese *) src;

  free(c->primes);
  free(c->offsets);
  free(c->sieve);
  free(c->rand);
}

/**
 * Changes the given Chinese in a semi random way.
 * The higher the evolution level, the greater is the optimization effort
 */
void mutate_chinese(Individual *iv, void *opts) {
  
  CreatOpts *opt = (CreatOpts *) opts;
  Chinese *c = (Chinese *) iv->iv;

  sieve_t max_mut = opt->n_primes / 10;
  log_str("mutate iv with fitness: " + 
          itoa(c->sievesize - (iv->fitness * c->sievesize) / 10000000000), LOG_I);
  
  /* set several random offsets */
  if (max_mut - 1 > opt->level) {
    for (sieve_t i = (rand128(c->rand) % (max_mut - opt->level)) + 1; i > 0; i--) {
      sieve_t index = c->fixed_len + (rand128(c->rand) % (c->n_primes - c->fixed_len));
      c->offsets[index] = rand128(c->rand) % c->primes[index];
    }

  /* try several random offsets (take the best) */
  } else if (opt->level < max_mut * 4) {

    sieve_t index = c->fixed_len + (rand128(c->rand) % (c->n_primes - c->fixed_len));

    int64_t best_fitness = INT64_MAX;
    sieve_t best_offset  = 0;

    for (sieve_t i = 0; i < opt->level - (3 * max_mut / 4); i++) {
      c->offsets[index] = rand128(c->rand) % c->primes[index];
      int64_t cur_fitness = fitness_chinese(iv, opts);

      if (cur_fitness < best_fitness) {
        best_fitness = cur_fitness;
        best_offset  = c->offsets[index];
      }
    }
    c->offsets[index] = best_offset;

  /* try several random offsets (take the best) do it for several primes */
  } else if (opt->level < opt->n_primes) {
    
    for (sieve_t i = 0; i < opt->level / 10; i++) {
      log_str("mutate level: " +  itoa(opt->level) + " " + itoa(i + 1) + " / " + itoa(opt->level / 10), LOG_D);

      sieve_t index = c->fixed_len + (rand128(c->rand) % (c->n_primes - c->fixed_len));
 
      int64_t best_fitness = INT64_MAX;
      sieve_t best_offset  = 0;
 
      for (sieve_t i = 0; i < opt->level - max_mut / 4; i++) {
        c->offsets[index] = rand128(c->rand) % c->primes[index];
        int64_t cur_fitness = fitness_chinese(iv, opts);
 
        if (cur_fitness < best_fitness) {
          best_fitness = cur_fitness;
          best_offset  = c->offsets[index];
        }
      }
      c->offsets[index] = best_offset;
    }

  /* fully optimize several random primes */
  } else if (opt->level < opt->n_primes * 2) {

    sieve_t max = (opt->n_primes * 2 - opt->level) / 5;
    if (max == 0)
      max = 1;

    for (sieve_t i = 0; i < max; i++) {
      log_str("mutate level: " +  itoa(opt->level) + " " + itoa(i + 1) + " / " + itoa(max), LOG_D);

      sieve_t index = c->fixed_len + (rand128(c->rand) % (c->n_primes - c->fixed_len));
 
      int64_t best_fitness = INT64_MAX;
      sieve_t best_offset  = 0;
 
      for (sieve_t i = 0; i < c->primes[index]; i++) {
        c->offsets[index] = i;
        int64_t cur_fitness = fitness_chinese(iv, opts);
 
        if (cur_fitness < best_fitness) {
          best_fitness = cur_fitness;
          best_offset  = c->offsets[index];
        }
      }
      c->offsets[index] = best_offset;
    }

  /* fully optimizes several random primes pairs */
  } else {
  
    sieve_t times = (opt->n_primes * 8 - opt->level) / 10;
    if (times == 0)
      times = 1;

    for (sieve_t i = 0; i < times; i++) {
      log_str("mutate level: " +  itoa(opt->level) + " " + itoa(i + 1) + " / " + itoa(times), LOG_D);

      sieve_t i1 = c->fixed_len + (rand128(c->rand) % (c->n_primes - c->fixed_len));
      sieve_t i2 = c->fixed_len + (rand128(c->rand) % (c->n_primes - c->fixed_len));
 
      while (i1 == i2)
        i2 = c->fixed_len + (rand128(c->rand) % (c->n_primes - c->fixed_len));
  
      int64_t best_fitness = INT64_MAX;
      sieve_t bo1 = 0, bo2 = 0;
      sieve_t max =  c->primes[i1] * c->primes[i2];
  
      for (sieve_t i = 0; i < max; i++) {
        c->offsets[i1] = i % c->primes[i1];
        c->offsets[i2] = i % c->primes[i2];
 
        int64_t cur_fitness = fitness_chinese(iv, opts);
  
        if (cur_fitness < best_fitness) {
          best_fitness = cur_fitness;
          bo1  = c->offsets[i1];
          bo2  = c->offsets[i2];
        }
      }
      c->offsets[i1] = bo1;
      c->offsets[i2] = bo2;
    }
  }
}

/**
 * calculates how good this ctr is
 */
int64_t fitness_chinese(Individual *iv, void *opts) {
  (void)  opts;
  
  Chinese *c = (Chinese *) iv->iv;
  memcpy(c->sieve, c->base, c->byte_size);

  for (sieve_t i = c->fixed_len; i < c->n_primes; i++) {
    
    const sieve_t prime = first_primes[i];
    for (sieve_t p = c->offsets[i]; p < c->sievesize; p += prime)
      set_composite(c->sieve, p);
  }

  int64_t count = c->sievesize;
  for (sieve_t w = 0; w < c->word_size; w++)
    count -= popcount(c->sieve[w]);

  return (10000000000LL * count) / c->sievesize;
}

/**
 * chooses the sieve level, and stops the evolutional algorithm if no
 * more optimations occour
 */
char continue_chinese(Evolution *ev) {
  
  static bool init = false;
  if (!init) {
    init = true;
    return 1;
  }
  CreatOpts *opt = ((CreatOpts *) ev->opts[0]);
  static int zeros = 0;

  printf("\33[2K\rimproovs: %10d -> %15.5f%%  best fitness: %7.5f%% %7" PRISIEVE " / %7" PRISIEVE "\n ",       
         ev->info.improovs,                                                
         ev->info.improovs / ((double) ev->deaths) * 100.0,             
         ((double) ev->population[0]->fitness) / 100000000,
         (ev->population[0]->fitness * ((Chinese *) ev->population[0]->iv)->sievesize) / 10000000000,
         ((Chinese *) ev->population[0]->iv)->n_primes);
  
  /* no individual could be improved in the last generation */
  if (!ev->info.improovs) {
    if (opt->level > opt->n_primes * 4) {
      cout << opt->level << endl;
      opt->level += (opt->n_primes * 8 - opt->level) / 2;
    } else
      opt->level += (opt->n_primes * 2 - opt->level) / 2;
    cout << "No improves, set level: " << opt->level << " and mutation-iterations: ";
    if (opt->level > opt->n_primes * 4)
      cout << (opt->n_primes * 8 - opt->level) / 10 << endl;
    else
      cout << (opt->n_primes * 2 - opt->level) / 5 << endl;
    zeros++;
  } else
    zeros = 0;

  if (zeros >= 4 && opt->level < opt->n_primes * 4) {
    opt->level = opt->n_primes * 4 + 1;
    zeros = 0;
    cout << "4 Times no improves, set level: " << opt->level << endl;
  } else if (zeros >= 4 && opt->level > opt->n_primes * 4)
    return 0;

  return 1;
}

/**
 * starts the Chinese Evolution
 */
void start_chinese_evolution(sieve_t n_primes, 
                             double merit, 
                             sieve_t fixed_len,
                             sieve_t population,
                             sieve_t n_threads,
                             double range,
                             sieve_t bits) {
  srand(time(NULL));

  EvInitArgs args;
  CreatOpts **opts = (CreatOpts **) malloc(sizeof(CreatOpts *) * n_threads);
  CreatOpts opt;

  opt.n_primes  = n_primes;
  opt.merit     = merit;
  opt.fixed_len = fixed_len;
  opt.range     = range;
  opt.bits      = bits;
  opt.level     = n_primes;

  for (unsigned i = 0; i < n_threads; i++)
    opts[i] = &opt;

  args.init_iv              = new_chinese;
  args.clone_iv             = clone_chinese;
  args.free_iv              = free_chinese;
  args.mutate               = mutate_chinese;
  args.fitness              = fitness_chinese;
  args.recombinate          = NULL;
  args.continue_ev          = continue_chinese;
  args.population_size      = population;
  args.generation_limit     = 10000000;
  args.mutation_probability = 1;
  args.death_percentage     = 0.5;
  args.opts                 = (void **) opts;
  args.num_threads          = n_threads;
  args.verbosity            = EV_VERBOSITY_ONELINE;
  args.flags                = EV_USE_MUTATION |
                              EV_ALWAYS_MUTATE |
                              EV_KEEP_LAST_GENERATION |
                              EV_SORT_MIN |
                              EV_USE_ABORT_REQUIREMENT;

  

  Individual iv = best_evolution(&args);
  Chinese *c = (Chinese *) iv.iv;

  sieve_t avg_count = 0;
  sieve_t max_count = c->sievesize - (fitness_chinese(&iv, NULL) * c->sievesize) / 10000000000;

  /** calculate the average candidates per sieve */
  for (ssieve_t i = 0; i < 10000; i++) {

    cout << "running: " << 10000 - i << "    \r";
    memset(c->sieve, 0, c->byte_size);

    for (sieve_t x = 0; x < c->n_primes; x++) {
    
      const sieve_t index = rand128(c->rand) % first_primes[x];
      const sieve_t prime = first_primes[x];
      
      for (sieve_t p = prime - index; p < c->sievesize; p += prime)
        set_composite(c->sieve, p);

    }

    /* count the candidates */
    sieve_t cur_count = 0;
    for (sieve_t s = 0; s < c->word_size; s++)
      cur_count += popcount(c->sieve[s]);

    avg_count += cur_count;
  }

  double max_candidates = c->sievesize - max_count;
  double avg_candidates = c->sievesize - (((double) avg_count) / 10000);

  cout << "[II] max: " << max_count;
  cout << " avg: " << avg_count / 10000 << endl;
  cout << " " << 100.0 - (max_candidates / avg_candidates) * 100;
  cout << " % more composite numbers than average" << endl;
  cout << " " << exp((1.0 - (max_candidates / avg_candidates)) * merit) << " factor speed increase" << endl;
  cout << " " << max_count - (avg_count / 10000) << " candidates less" << endl;
  cout << " " << c->sievesize - (avg_count / 10000) << " candidates avg" << endl;
  cout << " " << c->sievesize - max_count << " candidates min" << endl;
  cout << " " << (avg_candidates / c->sievesize) * 100 << " % prime candidates avg" << endl;
  cout << " " << (max_candidates / c->sievesize) * 100 << " % prime candidates min" << endl;

  /* convert residues */
  for (sieve_t i = 0; i < c->n_primes; i++)
    c->offsets[i] = (first_primes[i] - c->offsets[i]) % first_primes[i];
 
  /* create reminder */
  ChineseRemainder cr(c->primes, c->offsets, c->n_primes);

  mpz_t mpz_tmp;
  mpz_init_set(mpz_tmp, cr.mpz_target);

  ChineseSet *best_set = new ChineseSet(c->n_primes, c->sievesize, max_candidates + 1, mpz_tmp);
  best_set->save(Opts::get_instance()->get_ctr_file().c_str());

}
