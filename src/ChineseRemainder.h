/**
 * Header for a Chinese Remainder Theorem computation class
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
#include "PoWCore/src/Sieve.h"
#include <vector>
#include <gmp.h>

using namespace std;

class ChineseRemainder {
  
  public:
    
    /* numbers */
    sieve_t *numbers;

    /* reminders */
    sieve_t *reminders;

    /* number of numbers */
    sieve_t len;
    
    /* primorial */
    mpz_t mpz_primorial;

    /* target */
    mpz_t mpz_target;

    /* run the test */
    ChineseRemainder(const sieve_t *numbers, 
                     const sieve_t *reminders, 
                     const sieve_t len);

    /* cleanup */
    ~ChineseRemainder();
};
