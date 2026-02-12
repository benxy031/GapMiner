/**
 * Implementation of a prime gap candidate used in the ChineseSieve
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
#include "GapCandidate.h"
#include <cstring>

/* creat a new GapCandidate */
GapCandidate::GapCandidate(uint32_t nonce,
                           uint64_t target,
                           mpz_t mpz_gap_start,
                           const uint32_t* candidates,
                           uint32_t n_candidates,
                           double score) {

  this->nonce        = nonce;
  this->target       = target;
  this->n_candidates = n_candidates;
  this->score        = score;
  if (candidates && n_candidates > 0) {
    this->candidates.assign(candidates, candidates + n_candidates);
  } else {
    this->candidates.clear();
  }
  mpz_init_set(this->mpz_gap_start, mpz_gap_start);
}

void GapCandidate::reset(uint32_t nonce,
                         uint64_t target,
                         mpz_t mpz_gap_start,
                         const uint32_t* candidates,
                         uint32_t n_candidates,
                         double score) {
  this->nonce        = nonce;
  this->target       = target;
  this->n_candidates = n_candidates;
  this->score        = score;
  if (candidates && n_candidates > 0) {
    this->candidates.assign(candidates, candidates + n_candidates);
  } else {
    this->candidates.clear();
  }
  mpz_set(this->mpz_gap_start, mpz_gap_start);
}

GapCandidate::~GapCandidate() {
  mpz_clear(mpz_gap_start);
}
