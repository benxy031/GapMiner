/**
 * Implementation of an storage class for the first
 * n primes with their OnePrimeSieve
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
#include "BestChinese.h"
#include "ChineseRemainder.h"
#include "PoWCore/src/PoWUtils.h"
#include "ChineseSet.h"
#include "Opts.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <pthread.h>

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

using namespace std;


/* create a new BestChinese */
BestChinese::BestChinese(sieve_t n, 
                         double merit, 
                         sieve_t min_bits,
                         ssieve_t avg_tests,
                         sieve_t max_gready, 
                         sieve_t index,
                         bool verbose) {

  /* generate the primorial */
  mpz_t mpz_primorial;
  mpz_init_set_ui(mpz_primorial, 1);
  for (sieve_t i = 0; i < n; i++)
    mpz_mul_ui(mpz_primorial, mpz_primorial, first_primes[i]);

  this->merit       = merit;
  this->sievesize   = merit * (mpz_log(mpz_primorial) + min_bits * log(2)) ;
  this->byte_size   = bound(sievesize, sizeof(sieve_t) * 8) / 8;
  this->sieve       = (sieve_t *) malloc(byte_size);
  this->prev_layers = (sieve_t *) malloc(byte_size);
  this->n_primes    = n;
  this->avg_tests   = avg_tests;
  this->max_gready  = max_gready;
  this->i           = index;
  this->verbose     = verbose;
  this->primorial   = 1;
  this->primes      = (OnePrimeSieve **) malloc(sizeof(OnePrimeSieve *) * n);

  cout << "[" << this->i << "] Initialized(n_primes = " << n_primes << ", sievesize = " << sievesize << ")\n";
  mpz_clear(mpz_primorial);
}

/* deletes this */
BestChinese::~BestChinese() {

  free(primes);
  free(sieve);
  free(prev_layers);
}


/* returns the best_set */
ChineseSet *BestChinese::get_best_set() {
  return best_set;
}

/* calculate the perfect residue classes */
sieve_t BestChinese::calc_best_residues(bool save) {

  if (verbose) {
    cout << "[" << this->i << "] Running with " << n_primes << " primes" << endl;
    cout << "[" << this->i << "] sievesize is " << sievesize << endl << endl;
  }
  
  sieve_t max = 0;
  sieve_t max_count = 1;
  sieve_t sieve_end = byte_size / sizeof(sieve_t);
  sieve_t *residues = (sieve_t *) malloc(sizeof(sieve_t) * n_primes);
  memset(residues, 0, sizeof(sieve_t) * n_primes);
  sieve_t gready_n = 0;
  primorial = 1;

  srand(time(NULL));
  sieve_t avg_count = 0;

  /** calculate the average candidates per sieve */
  for (ssieve_t i = 0; i < this->avg_tests; i++) {

    if (verbose) {
      cout << "running: " << this->avg_tests - i << "    \r";
    }
    memset(sieve, 0, byte_size);

    /* create a random sieve */
    for (sieve_t x = 0; x < n_primes; x++) {
    
      const sieve_t index = rand() % first_primes[x];
      const sieve_t prime = first_primes[x];
      
      for (sieve_t p = prime - index; p < sievesize; p += prime)
        set_composite(sieve, p);

    }

    /* count the candidates */
    sieve_t cur_count = 0;
    for (sieve_t s = 0; s < sieve_end; s++)
      cur_count += popcount(sieve[s]);

    avg_count += cur_count;
  }


  memset(sieve, 0, byte_size);
  memset(prev_layers, 0, byte_size);
  gready_n = 0;

  

  /**
   * start greedy algorithm 
   *
   * for several primes (prime-layers) calculate the optimal offset
   * reduce the number of used layers dynamically while work increases 
   */
  sieve_t g;
  for (g = gready_n; g < n_primes - gready_n && g < LEVEL1_LAYERS - gready_n; g++) {

    primorial = 1;
    for (gready_n = 0; gready_n < n_primes - g; gready_n++) {

      const sieve_t prime = first_primes[gready_n + g];
      primes[gready_n + g] = new OnePrimeSieve(prime, sievesize, false);

      /* for the current prime calculate all needed indices */
      for (sieve_t s = 0; s < sievesize; s++) {
        if (is_prime(prev_layers, s)) {
          
          sieve_t index = prime - s % prime;
          if (index == prime)
            index = 0;

          primes[gready_n + g]->set(index);
        }
      }
  
      primorial *= primes[gready_n + g]->n_setted();
      
      if (primorial > this->max_gready) {
        primorial /= primes[gready_n + g]->n_setted();
        delete primes[gready_n + g];
        break;
      }

      if (verbose) {
        cout << "[" << this->i << "] (greedy: " << gready_n + 1 << ") indices: ";
        cout << primes[gready_n + g]->n_setted() << " / " << prime << endl;
      }

      if (primorial * (first_primes[gready_n + g + 1] / 2) > this->max_gready) {
        gready_n++;
        break;
      }
    }
 
    if (verbose) {
      cout << "[" << this->i << "] (greedy: " << gready_n << ") Rounds: " << g << " / ";
      cout << n_primes - gready_n << " Primorial: " << primorial << endl;
    } else {
      pthread_mutex_lock(&mutex);
      cout << "[" << this->i << "][" << ((double) g) / n_primes << "]    \r";
      pthread_mutex_unlock(&mutex);
    }

 
    max_count = 0; /* apply previous layers */
    for (sieve_t i = 0; i < primorial; i++) {
 
      /* apply previous layers */
      memcpy(sieve, prev_layers, byte_size);
 
      /* apply current layers */
      for (sieve_t n = g; n < g + gready_n; n++) {
  
        sieve_t index  = i % primes[n]->setted.size();
        index          = primes[n]->setted[index];
        sieve_t *layer = primes[n]->sieves[index];
      
        /* apply the layer */
        for (sieve_t s = 0; s < sieve_end; s++)
          sieve[s] |= layer[s];
  
      }
      
      /* count the candidates */
      sieve_t cur_count = 0;
      for (sieve_t s = 0; s < sieve_end; s++)
        cur_count += popcount(sieve[s]);
      
      if (cur_count > max_count) {
        max = i;
        max_count = cur_count;
        if (verbose) {
          cout << "[" << this->i << "] (greedy) avg: " << avg_count / this->avg_tests;
          cout << " max: " << max_count << " / " << sievesize;
          double max_candidates = sievesize - max_count;
          double avg_candidates = sievesize - (((double) avg_count) / this->avg_tests);
          cout << " => " << ((long long) max_count) - ((long long) avg_count) / this->avg_tests << ", ";
          cout << 100.0 - (max_candidates / avg_candidates) * 100;
          cout << " % less";
          cout << "                          " << endl;
        }
      }
    }
  
    /* safe the new residues */
    for (sieve_t n = g; n < g + gready_n; n++) {
      sieve_t index = max % primes[n]->setted.size();
      residues[n]   = primes[n]->setted[index];
    }

    /* add next layer */
    sieve_t *layer = primes[g]->sieves[residues[g]];
    
    /* apply the layer */
    for (sieve_t s = 0; s < sieve_end; s++)
      prev_layers[s] |= layer[s];
  
    /* free layers */
    for (sieve_t n = g; n < g + gready_n; n++)
      delete primes[n];

  }
  

  /* above prime ~ 35 sieving is faster than OR-ing the layers */
  for (;g < n_primes - gready_n; g++) {

    primorial = 1;
    for (gready_n = 0; gready_n < n_primes - g; gready_n++) {

      const sieve_t prime = first_primes[gready_n + g];
      primes[gready_n + g] = new OnePrimeSieve(prime, sievesize, false);

      /* for the current prime calculate all needed indices */
      for (sieve_t s = 0; s < sievesize; s++) {
        if (is_prime(prev_layers, s)) {
          
          sieve_t index = prime - s % prime;
          if (index == prime)
            index = 0;

          primes[gready_n + g]->set2(index);
        }
      }
      primes[gready_n + g]->set2_finish();
  
      primorial *= primes[gready_n + g]->n_setted();
      
      if (primorial > this->max_gready && gready_n == 0) {
        if (verbose) {
          cout << "[" << this->i << "] (greedy: " << gready_n + 1 << ") indices: ";
          cout << primes[gready_n + g]->n_setted() << " / " << prime << endl;
        }
        gready_n = 1;
        break;
      } else if (primorial > this->max_gready) {
        primorial /= primes[gready_n + g]->n_setted();
        delete primes[gready_n + g];
        break;
      }

      if (verbose) {
        cout << "[" << this->i << "] (greedy: " << gready_n + 1 << ") indices: ";
        cout << primes[gready_n + g]->n_setted() << " / " << prime << endl;
      }

      if (primorial * (first_primes[gready_n + g + 1] / 2) > this->max_gready) {
        gready_n++;
        break;
      }
    }
    sieve_t layer_count = 0;
    for (sieve_t s = 0; s < sieve_end; s++)
      layer_count += popcount(prev_layers[s]);
 
    if (verbose) {
      cout << "[" << this->i << "] (greedy: " << gready_n << ") Rounds: " << g << " / ";
      cout << n_primes - gready_n << " Primorial: " << primorial << endl;
    } else {
      pthread_mutex_lock(&mutex);
      cout << "[" << this->i << "][" << ((double) g) / n_primes << "]    \r";
      pthread_mutex_unlock(&mutex);
    }

 
    max_count = 0; /* apply previous layers */
    for (sieve_t i = 0; i < primorial; i++) {
 
      /* apply previous layers */
      memcpy(sieve, prev_layers, byte_size);
      sieve_t cur_count = layer_count;
 
      /* apply current layers */
      for (sieve_t n = g; n < g + gready_n; n++) {
  
        sieve_t index  = i % primes[n]->setted.size();
        index          = primes[n]->setted[index];
        const sieve_t prime = first_primes[n];

        for (sieve_t p = prime - index; p < sievesize; p += prime) {
          if (is_prime(sieve, p)) {
            set_composite(sieve, p);
            cur_count++;
          }
        }
      }
      
      if (cur_count > max_count) {
        max = i;
        max_count = cur_count;
        if (verbose) {
          cout << "[" << this->i << "] (greedy) avg: " << avg_count / this->avg_tests;
          cout << " max: " << max_count << " / " << sievesize;
          double max_candidates = sievesize - max_count;
          double avg_candidates = sievesize - (((double) avg_count) / this->avg_tests);
          cout << " => " << ((long long) max_count) - ((long long) avg_count) / this->avg_tests<< ", ";
          cout << 100.0 - (max_candidates / avg_candidates) * 100;
          cout << " % less";
          cout << "                          " << endl;
        }
      }
    }
  
    /* safe the new residues */
    for (sieve_t n = g; n < g + gready_n; n++) {
      sieve_t index = max % primes[n]->setted.size();
      residues[n]   = primes[n]->setted[index];
    }

    const sieve_t index = residues[g];
    const sieve_t prime = first_primes[g];
    
    for (sieve_t p = prime - index; p < sievesize; p += prime)
      set_composite(prev_layers, p);
  
  
    /* free layers */
    for (sieve_t n = g; n < g + gready_n; n++)
      delete primes[n];

  }
  

  double max_candidates = sievesize - max_count;
  double avg_candidates = sievesize - (((double) avg_count) / this->avg_tests);

  if (verbose) {
    cout << "[" << this->i << "] max: " << max_count;
    cout << " avg: " << avg_count / this->avg_tests << endl;
    cout << " " << 100.0 - (max_candidates / avg_candidates) * 100;
    cout << " % more composite numbers than average" << endl;
    cout << " " << exp((1.0 - (max_candidates / avg_candidates)) * merit) << " factor speed increase" << endl;
    cout << " " << max_count - (avg_count / this->avg_tests) << " candidates less" << endl;
    cout << " " << sievesize - (avg_count / this->avg_tests) << " candidates avg" << endl;
    cout << " " << sievesize - max_count << " candidates max" << endl;
    cout << " " << (avg_candidates / sievesize) * 100 << " % prime candidates avg" << endl;
    cout << " " << (max_candidates / sievesize) * 100 << " % prime candidates min" << endl;
  }
 
  /* create reminder */
  ChineseRemainder cr(first_primes, residues, n_primes);

  mpz_t mpz_tmp;
  mpz_init_set(mpz_tmp, cr.mpz_target);

  best_set = new ChineseSet(n_primes, sievesize, max_candidates, mpz_tmp);
  if (save)
    best_set->save(Opts::get_instance()->get_ctr_file().c_str());

  free(residues);

  return max;
}
