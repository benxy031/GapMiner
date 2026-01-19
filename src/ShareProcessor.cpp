/**
 * Implementation of GapMiners pow processing functionality
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
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <vector>
#include "PoWCore/src/PoWProcessor.h"
#include "PoWCore/src/PoW.h"
#include "BlockHeader.h"
#include "ShareProcessor.h"
#include "Rpc.h"
#include "utils.h"
#include "Opts.h"
#include "Stratum.h"
//#include "UTILS.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;


/* synchronization mutexes */
pthread_mutex_t ShareProcessor::queue_mutex    = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t ShareProcessor::wait_mutex     = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t ShareProcessor::creation_mutex = PTHREAD_MUTEX_INITIALIZER;

/* indicates if this was initialized */
bool ShareProcessor::initialized = false;

/* indicates that the share thread is running */
bool ShareProcessor::running = true;

/* this will be a singleton */
ShareProcessor *ShareProcessor::only_instance = NULL;


/* return the only instance of this */
ShareProcessor *ShareProcessor::get_processor() {

  log_str("returning only instance", LOG_D);
  pthread_mutex_lock(&creation_mutex);
  if (!initialized) {
    initialized   = true;
    only_instance = new ShareProcessor();
  }
  pthread_mutex_unlock(&creation_mutex);

  return only_instance;
}

/* run clean up */
void ShareProcessor::cleanup() {

  log_str("cleanup", LOG_D);
  pthread_mutex_lock(&creation_mutex);
  if (initialized) {
    initialized = false;
    delete only_instance;
    only_instance = NULL;
  }
  pthread_mutex_unlock(&creation_mutex);
}

/**
 * Processes a given PoW
 */
bool ShareProcessor::process(PoW *pow) {
  
  log_str("processing pow with target: " + itoa(pow->get_target()) +
      " and nonce: " + itoa(pow->get_nonce()) + 
      " and difficulty: " + itoa(pow->difficulty()), LOG_D);

  /* stop sieve if we mine stales */
  bool stop_sieveing = true;

  if (header != NULL) {
    BlockHeader *cur = get_header_from(pow);
 
    /* is PoW for the current hash */
    if (cur != NULL) {
 
      pthread_mutex_lock(&queue_mutex);
      shares.push(cur);
 
      if (!running) {
        running = true;
 
        /* wake up share thread */
        pthread_mutex_unlock(&wait_mutex);
      }
      pthread_mutex_unlock(&queue_mutex);
      
      stop_sieveing = false;
    } else
      log_str("got stale pow", LOG_W);
  }

  return stop_sieveing;
}

/* updates the current block header */
void ShareProcessor::update_header(BlockHeader *header) {
  log_str("updating BlockHeader", LOG_D);
  pthread_mutex_lock(&queue_mutex);
  
  /* discard now obsolete shares */
  while (!shares.empty()) {
    BlockHeader *cur = shares.front();
    shares.pop();
    delete cur;
  }

  if (header != NULL)
    delete this->header;

  this->header = header->clone();

  pthread_mutex_unlock(&queue_mutex);
}

/* private constructor to allow only one instance */
ShareProcessor::ShareProcessor() {
  log_str("creating", LOG_D);
  this->header = NULL;

  /* lock the shares thread */
  pthread_mutex_lock(&wait_mutex);
  
  pthread_create(&thread, NULL, share_processor, (void *) &shares);
}

/* private destructor */
ShareProcessor::~ShareProcessor() {
  log_str("deleting", LOG_D);
  pthread_mutex_lock(&queue_mutex);

  /* clear shares */
  while (!shares.empty()) {
    BlockHeader *cur = shares.front();
    shares.pop();
    delete cur;
  }

  pthread_mutex_unlock(&queue_mutex);

  pthread_join(thread, NULL);

  if (header != NULL)
    delete header;
}

/* thread which processes the share queue */
void *ShareProcessor::share_processor(void *args) {

  log_str("share_processor thread started", LOG_D);
  queue<BlockHeader *> *shares = (queue<BlockHeader *> *) args;
  BlockHeader *cur;
  Rpc *rpc = NULL;
  Stratum *stratum = NULL;
  Opts *opts = Opts::get_instance();
  bool has_stratum = opts->has_stratum();

  if (has_stratum)
    stratum = Stratum::get_instance();
  else
    rpc = Rpc::get_instance();

  while (initialized) {
    
    /* get the next share to process */
    pthread_mutex_lock(&queue_mutex);

    /* break if nothing to do */
    if (shares->empty()) {
      running = false;
      pthread_mutex_unlock(&queue_mutex);

      /* wait till we get aroused */
      pthread_mutex_lock(&wait_mutex);
      continue;
    } else {

      cur = shares->front();
      shares->pop();
      pthread_mutex_unlock(&queue_mutex);
    }


    /* send the share to server */
    if (has_stratum) {
      stratum->sendwork(cur);
    } else {

      PoW pow = cur->get_pow();
      const uint64_t share_difficulty = pow.difficulty();
      const bool is_block_candidate = share_difficulty >= pow.get_target();

      /* submit share */
      log_str("Submitting Share: " + itoa(share_difficulty), LOG_D);
      bool accepted = rpc->sendwork(cur);

      const string base_status = (accepted ? "accepted" : "stale!");
      const string block_status = (is_block_candidate ? " (block)" : "");
      log_str("Found Share: " + itoa(share_difficulty) +
              "  =>  " + base_status + block_status, LOG_I);

      const double normalized_merit = ((double) share_difficulty) / TWO_POW48;
      const string status = base_status + block_status;
      ostringstream share_line;
      share_line.setf(std::ios::fixed, std::ios::floatfield);
      share_line.precision(7);
      share_line << get_time();
      share_line << "Found Share: ";
      share_line << normalized_merit;
      share_line << "  =>  " << status;
      const string formatted_share = share_line.str();

      pthread_mutex_lock(&io_mutex);
      cout.setf(std::ios::fixed, std::ios::floatfield);
      cout.precision(7);
      cout << formatted_share << endl;

      if (opts->log_shares()) {
        static ofstream share_log_file;
        static bool share_log_initialized = false;
        static bool share_log_failed = false;

        if (!share_log_initialized) {
          share_log_file.open("shares.txt", ios::app);
          if (share_log_file) {
            share_log_file.setf(std::ios::fixed, std::ios::floatfield);
            share_log_file.precision(7);
          } else if (!share_log_failed) {
            log_str("failed to open shares.txt for share logging", LOG_W);
            share_log_failed = true;
          }
          share_log_initialized = true;
        }

        if (share_log_file) {
          share_log_file << formatted_share << endl;
          share_log_file.flush();
        }
      }

      pthread_mutex_unlock(&io_mutex);
    }

    delete cur;
  }

  log_str("share_processor thread stopped", LOG_D);
  return NULL;
}

/**
 * test if share is for the current block
 * if true returns the adjusted Block,
 * else NULL
 */
BlockHeader *ShareProcessor::get_header_from(PoW *share) {

  log_str("getting header for pow with nonce: " + itoa(share->get_nonce()) +
          " and difficulty: " + itoa(share->difficulty()), LOG_D);
  pthread_mutex_lock(&queue_mutex);
  BlockHeader *cur = header->clone();
  pthread_mutex_unlock(&queue_mutex);

  cur->nonce = share->get_nonce(); 
  cur->shift = share->get_shift(); 
  share->get_adder(&cur->adder);

  mpz_t mpz_hhash, mpz_shash;
  mpz_init(mpz_hhash);
  mpz_init(mpz_shash);

  share->get_hash(mpz_shash);
  cur->get_hash(mpz_hhash);

  BlockHeader *ret = NULL;

  /* is share for the current hash */
  if (!mpz_cmp(mpz_shash, mpz_hhash))
    ret = cur;
  else {
    log_str("processed stale share", LOG_W);
    delete cur;
  }

  mpz_clear(mpz_shash);
  mpz_clear(mpz_hhash);

  return ret;
}
