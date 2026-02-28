/**
 * GapMiners main 
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
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <iomanip>
#include <algorithm>
#include <deque>
#include <cmath>
#include "BlockHeader.h"
#include "Miner.h"
#include "PoWCore/src/PoWUtils.h"
#include "Opts.h"
#include "Rpc.h"
#include "utils.h"
#include "Stratum.h"
#include "GPUFermat.h"
#include "BestChinese.h"
#include "ctr-evolution.h"

using namespace std;

/* indicates if program should continue running */
static bool running = true;

/* indicates that we are waiting for server */
static bool waiting = false;

/* async-signal-safe shutdown flags */
static volatile sig_atomic_t shutdown_requested = 0;
static volatile sig_atomic_t shutdown_count = 0;

/* the miner */
static Miner *miner;

#ifndef WINDOWS
/**
 * signal handler to exit program softly
 */
void soft_shutdown(int signum) {

  log_str("soft_shutdown", LOG_D);
  (void) signum;

  const sig_atomic_t count = ++shutdown_count;
  shutdown_requested = 1;
  running = false;
  waiting = true;

  if (count >= 3) {
    const char msg[] = "\nforced shutdown\n";
    write(STDERR_FILENO, msg, sizeof(msg) - 1);
    _exit(128 + signum);
  }
}

/* init signal handler */
void init_signal() {

  log_str("init_signal", LOG_D);

  /* set signal handler for soft shutdown */
  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));

  action.sa_handler = soft_shutdown;
  sigaction(SIGINT,  &action, NULL);
  sigaction(SIGTERM, &action, NULL);
  sigaction(SIGALRM, &action, NULL);

  /* continue mining if terminal lost connection */
  action.sa_handler = SIG_IGN;
  sigaction(SIGHUP,  &action, NULL);
  sigaction(SIGPIPE, &action, NULL); 
  sigaction(SIGQUIT, &action, NULL);
}
#endif

/* periodically look if new work is available */
void *getwork_thread(void *arg) {

  log_str("getwork_thread started", LOG_D);
  Miner *miner = (Miner *) arg;
  Opts  *opts  = Opts::get_instance();

  if (opts->has_extra_vb()) {
    extra_verbose_log(get_time() + "Getwork thread started");
  }
  

  Rpc *rpc            = Rpc::get_instance();
  BlockHeader *header = rpc->getwork();

  while (header == NULL) {
    waiting = true;
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "waiting for server ..." << endl;
    pthread_mutex_unlock(&io_mutex);
    sleep(5);

    if (!running) return NULL;
    header = rpc->getwork();
    waiting = false;
  }

  pthread_mutex_lock(&io_mutex);
  cout.precision(7);
  cout << get_time() << "Got new target: ";
  cout << fixed << (((double) header->target) / TWO_POW48) << " @ ";
  cout << fixed << (((double) header->difficulty) / TWO_POW48) << endl;
  pthread_mutex_unlock(&io_mutex);

  bool longpoll = rpc->has_long_poll();

  uint16_t shift = (opts->has_shift() ?  atoi(opts->get_shift().c_str()) : 20);
  header->shift  = shift;

#ifndef CPU_ONLY
  if (opts->has_use_gpu() && !opts->has_shift())
    shift = 45;
#endif    
  

  /* start mining */
  miner->start(header);

  /* 5 seconds default */
  unsigned int sec = (opts->has_pull() ? atoll(opts->get_pull().c_str()) : 5);
  time_t work_time = time(NULL);

  PoWUtils pow_utils;

  while (running) {
    if (!longpoll)
      sleep(sec);

    BlockHeader *new_header = rpc->getwork(longpoll);

    while (new_header == NULL) {
      waiting = true;

      pthread_mutex_lock(&io_mutex);
      cout << get_time() << "waiting for server ..." << endl;
      pthread_mutex_unlock(&io_mutex);

      sleep(sec);

      if (!running) return NULL;
      new_header = rpc->getwork();
      waiting = false;
    }
  
    new_header->shift = shift;

    const bool same_height = header->equal_block_height(new_header);
    const bool same_template = header->equal(new_header);

    if (!same_height ||
        time(NULL) >= work_time + 180) {

      miner->update_header(new_header);

      delete header;
      header = new_header;
      time(&work_time);

      if (!opts->has_quiet()) {
        pthread_mutex_lock(&io_mutex);
        cout.precision(7);
        cout << get_time() << "Got new target: ";
        cout << fixed << (((double) header->target) / TWO_POW48) << " @ ";
        cout << fixed << (((double) header->difficulty) / TWO_POW48) << endl;
        pthread_mutex_unlock(&io_mutex);
      }
    } else {
      delete new_header;
    }
  }

  log_str("getwork_thread stopped", LOG_D);
  return NULL;
}

int main(int argc, char *argv[]) {

  log_str("gapminer started", LOG_D);
  Opts *opts = Opts::get_instance(argc, argv);
  log_str("args parsed", LOG_D);

  if (opts->has_license()) {
    cout << "    GapMiner is a standalone Gapcoin (GAP) CPU rpc miner                 " << endl;
    cout << "                                                                         " << endl;
    cout << "    Copyright (C)  2014  The Gapcoin developers  <info@gapcoin.org>      " << endl;
    cout << "                                                                         " << endl;
    cout << "    This program is free software: you can redistribute it and/or modify " << endl;
    cout << "    it under the terms of the GNU General Public License as published by " << endl;
    cout << "    the Free Software Foundation, either version 3 of the License, or    " << endl;
    cout << "    (at your option) any later version.                                  " << endl;
    cout << "                                                                         " << endl;
    cout << "    This program is distributed in the hope that it will be useful,      " << endl;
    cout << "    but WITHOUT ANY WARRANTY; without even the implied warranty of       " << endl;
    cout << "    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        " << endl;
    cout << "    GNU General Public License for more details.                         " << endl;
    cout << "                                                                         " << endl;
    cout << "    You should have received a copy of the GNU General Public License    " << endl;
    cout << "    along with this program.  If not, see <http://www.gnu.org/licenses/>." << endl;
    exit(EXIT_SUCCESS);
  }

#ifndef CPU_ONLY
  if (opts->has_benchmark()) {
    unsigned dev_id = (opts->has_gpu_dev() ? atoi(opts->get_gpu_dev().c_str()) : 0);
    string platfrom  = (opts->has_platform() ? string(opts->get_platform()) : "amd", "nvidia");
    unsigned workItems = (opts->has_work_items() ? atoi(opts->get_work_items().c_str()) : 2048);

    // Debug: print parsed GPU/device/group options
    pthread_mutex_lock(&io_mutex);
    cout << get_time() << "DEBUG: has_gpu_dev=" << opts->has_gpu_dev();
    if (opts->has_gpu_dev()) cout << " gpu_dev=" << opts->get_gpu_dev();
    cout << " has_group_size=" << opts->has_group_size();
    if (opts->has_group_size()) cout << " group_size=" << opts->get_group_size();
    cout << endl;
    pthread_mutex_unlock(&io_mutex);

    if (opts->has_group_size()) {
      int gs = atoi(opts->get_group_size().c_str());
      if (gs > 0)
        GPUFermat::set_group_size(static_cast<unsigned>(gs));
    }

    GPUFermat *fermat = GPUFermat::get_instance(dev_id, platfrom.c_str(), workItems);
    fermat->benchmark();
    exit(EXIT_SUCCESS);
  }
#endif

#ifndef CPU_ONLY
  if (opts->use_cuda_comba())
    std::cout << "CUDA Comba: enabled" << std::endl;
#endif

  if (opts->has_calc_ctr() && 
      opts->has_ctr_primes() &&
      opts->has_ctr_merit() &&
      opts->has_ctr_file() &&
      !opts->has_ctr_evolution()) {

    sieve_t bits = 256;
    if (opts->has_ctr_bits())
      bits += atoi(opts->get_ctr_bits().c_str());

    sieve_t max_greedy = MAX_GREADY;

    if (opts->has_ctr_strength())
      max_greedy = atoi(opts->get_ctr_strength().c_str());

    BestChinese bc(atoi(opts->get_ctr_primes().c_str()),
                   atof(opts->get_ctr_merit().c_str()),
                   bits,
                   N_TEST,
                   max_greedy);

    bc.calc_best_residues(true);
    exit(EXIT_SUCCESS);
  }

  if (opts->has_calc_ctr() && 
      opts->has_ctr_primes() &&
      opts->has_ctr_merit() &&
      opts->has_ctr_file() &&
      opts->has_ctr_evolution() &&
      opts->has_ctr_n_ivs()) {
      
    sieve_t bits     = (opts->has_ctr_bits() ? 256 + atoi(opts->get_ctr_bits().c_str()) : 256);
    sieve_t n_primes = atoi(opts->get_ctr_primes().c_str());
    double  merit    = atof(opts->get_ctr_merit().c_str());
    sieve_t fixed_len = (opts->has_ctr_fixed() ? atoi(opts->get_ctr_fixed().c_str()) : 10);
    sieve_t n_ivs    = atoi(opts->get_ctr_n_ivs().c_str());
    double  range    = (opts->has_ctr_range() ? atof(opts->get_ctr_range().c_str()) : 0);
    sieve_t n_threads = (opts->has_threads() ? atoi(opts->get_threads().c_str()) : 1);

    start_chinese_evolution(n_primes, merit, fixed_len, n_ivs, n_threads, range, bits);
    exit(EXIT_SUCCESS);
  }


  if (opts->has_help()  ||
      !opts->has_host() ||
      !opts->has_port() ||
      !opts->has_user() ||
      !opts->has_pass()) {
    
    cout << opts->get_help();
    exit(EXIT_FAILURE);
  }

#ifndef WINDOWS
  init_signal();
#endif



  /* 1 thread default */
  int n_threads = (opts->has_threads() ? atoi(opts->get_threads().c_str()) : 1);

  /* default 25 sec timeout */
  int timeout = (opts->has_timeout() ? atoi(opts->get_timeout().c_str()) : 25);

  /* default shift 25 */
  uint16_t shift = (opts->has_shift() ?  atoi(opts->get_shift().c_str()) : 45);

  /* 10 seconds default */
  unsigned int sec = (opts->has_stats() ? atoll(opts->get_stats().c_str()) : 5);

  string host = opts->get_host();
  string port = opts->get_port();
  string user = opts->get_user();
  string pass = opts->get_pass();

  string http = (opts->get_host().find("http://") == 0) ? string("") : 
                                                          string("http://");

  if (!opts->has_stratum()) {
    Rpc::init_curl(user + string(":") + pass, 
                  http + host + string(":") + port, 
                  timeout);
  }

  uint64_t sieve_size = (opts->has_sievesize() ? 
                         atoll(opts->get_sievesize().c_str()) :
                         33554432); 

  uint64_t primes     = (opts->has_primes() ? 
                         atoll(opts->get_primes().c_str()) :
                         900000);

#ifndef CPU_ONLY
  if (opts->has_use_gpu()) {

    sieve_size = (opts->has_sievesize() ? 
                  atoll(opts->get_sievesize().c_str()) :
                  12000000); 
 
    primes     = (opts->has_primes() ? 
                 atoll(opts->get_primes().c_str()) :
                 3000000);

    shift = 45;

    unsigned dev_id = (opts->has_gpu_dev() ? atoi(opts->get_gpu_dev().c_str()) : 0);
    string platfrom  = (opts->has_platform() ? string(opts->get_platform()) : "amd", "nvidia");
    unsigned workItems = (opts->has_work_items() ? atoi(opts->get_work_items().c_str()) : 2048);

    GPUFermat *fermat = GPUFermat::get_instance(dev_id, platfrom.c_str(), workItems);

    if (opts->has_extra_vb() || opts->has_dump_config()) {
      std::ostringstream cfg;
      cfg << get_time() << "Config: threads=" << n_threads
          << " stats=" << sec
          << " shift=" << shift
          << " sieve_size=" << sieve_size
          << " sieve_primes=" << primes
          << " use_gpu=1"
          << " platform=" << (opts->has_platform() ? opts->get_platform() : "amd")
          << " gpu_dev=" << dev_id
          << " work_items=" << workItems
          << " num_gpu_tests=" << (opts->has_n_tests() ? opts->get_n_tests() : "")
          << " queue_size=" << (opts->has_queue_size() ? opts->get_queue_size() : "")
          << " cuda_comba=" << (opts->use_cuda_comba() ? "on" : "off")
          << " cuda_comba_soa=" << (opts->use_comba_soa() ? "on" : "off")
          << " cuda_sieve_proto=" << (opts->use_cuda_sieve_proto() ? "on" : "off")
          << " bitmap_pool_buffers=" << (opts->has_bitmap_pool_buffers() ? opts->get_bitmap_pool_buffers() : "")
          << " snapshot_pool_buffers=" << (opts->has_snapshot_pool_buffers() ? opts->get_snapshot_pool_buffers() : "")
          << " gpu_launch_divisor=" << (opts->has_gpu_launch_divisor() ? opts->get_gpu_launch_divisor() : "")
          << " gpu_launch_wait_ms=" << (opts->has_gpu_launch_wait_ms() ? opts->get_gpu_launch_wait_ms() : "")
          << " gpu_low_latency_launch=" << (opts->use_gpu_low_latency_launch() ? "on" : "off");
#ifdef USE_CUDA_BACKEND
      if (fermat) {
        cfg << " cuda_block=" << fermat->get_block_size()
            << " cuda_streams=" << fermat->get_stream_count()
            << " cuda_elements=" << fermat->get_elements_num();
    std::string at = fermat->get_autotune_choice();
    if (!at.empty())
      cfg << " autotune='" << at << "'";
      }
#endif
      if (opts->has_extra_vb()) {
        extra_verbose_log(cfg.str());
      } else {
        pthread_mutex_lock(&io_mutex);
        cout << cfg.str() << endl;
        pthread_mutex_unlock(&io_mutex);
      }
    }
  }
#endif

  if ((opts->has_extra_vb() || opts->has_dump_config()) && !opts->has_use_gpu()) {
    std::ostringstream cfg;
    cfg << get_time() << "Config: threads=" << n_threads
        << " stats=" << sec
        << " shift=" << shift
        << " sieve_size=" << sieve_size
        << " sieve_primes=" << primes
        << " use_gpu=0";
    if (opts->has_extra_vb()) {
      extra_verbose_log(cfg.str());
    } else {
      pthread_mutex_lock(&io_mutex);
      cout << cfg.str() << endl;
      pthread_mutex_unlock(&io_mutex);
    }
  }
 

  miner = new Miner(sieve_size, primes, n_threads);
  pthread_t thread;

  if (opts->has_stratum()) {
    Stratum::get_instance(&host, &port, &user, &pass, shift, miner);
  } else {
    pthread_create(&thread, NULL, getwork_thread, (void *) miner);
  }

  
  /* print status information while mining */
    struct RateSample {
      double pps;
      double tests;
      double gaps;
    };
    std::deque<RateSample> rate_samples;
    const size_t max_samples = std::max<size_t>(1u, (300u + sec - 1u) / sec);
    const size_t avg60_samples = std::max<size_t>(1u, (60u + sec - 1u) / sec);
    const size_t avg300_samples = max_samples;

  /* open CSV file for stats logging if requested */
  FILE *stats_csv_file = NULL;
  if (opts->has_stats_csv()) {
    stats_csv_file = fopen("stats.csv", "a");
    if (stats_csv_file) {
      /* check if file is empty to write header */
      fseek(stats_csv_file, 0, SEEK_END);
      if (ftell(stats_csv_file) == 0) {
        fprintf(stats_csv_file, "timestamp,pps,pps_avg,tests,tests_avg,gaps,gaps_avg,pps_avg60,tests_avg60,gaps_avg60,pps_avg300,tests_avg300,gaps_avg300\n");
        fflush(stats_csv_file);
      }
    }
  }

  PoWUtils pow_utils;
  uint64_t last_target_difficulty = 0;
  uint64_t last_work_signature = 0;
  time_t target_start_ts = time(NULL);
  uint64_t target_round = 0;
  bool target_is_new = false;

  while (running) {
    sleep(sec);

    static bool shutdown_handled = false;
    if (shutdown_requested && !shutdown_handled) {
      shutdown_handled = true;
      pthread_mutex_lock(&io_mutex);
      if (shutdown_count <= 1)
        cout << get_time() << "shutdown.." << endl;
      else
        cout << get_time() << "I'm on it, just wait! (press again to kill)" << endl;
      pthread_mutex_unlock(&io_mutex);
      if (miner)
        miner->stop();
    }

    if (!opts->has_quiet() && !waiting) {
      pthread_mutex_lock(&io_mutex);
      const uint64_t target_difficulty = miner->get_target_difficulty();
      const uint64_t work_signature = miner->get_work_signature();
      const time_t now_ts = time(NULL);
      target_is_new = false;
        if (target_difficulty > 0 &&
          (target_difficulty != last_target_difficulty ||
           (work_signature != 0 && work_signature != last_work_signature))) {
        last_target_difficulty = target_difficulty;
        last_work_signature = work_signature;
        target_start_ts = now_ts;
        target_round++;
        target_is_new = true;
      }
      const uint64_t target_age_sec = (target_difficulty > 0)
                                      ? static_cast<uint64_t>(now_ts - target_start_ts)
                                      : 0u;

      if (opts->has_cset() && miner->get_crt_status() < 100.0) {
        cout << get_time() << "init CRT [" << miner->get_crt_status() << " %]" << endl;
      } else {
#ifdef WINDOWS      
        string time = get_time();
        cout << time << "pps: " << (uint64_t) miner->primes_per_sec();
        cout << " / "           << (uint64_t) miner->avg_primes_per_sec();
        cout << " tests/s " << (uint64_t) miner->tests_per_second();
        cout << " / "        << (uint64_t) miner->avg_tests_per_second() << endl;

        for (unsigned i = 1; i < time.length(); i++) cout << " ";
        cout << "gaps/s " << (uint64_t) miner->gaps_per_second();
        cout << " / "        << (uint64_t) miner->avg_gaps_per_second();

        double blocks_per_day = 0.0;
        if (target_difficulty > 0) {
          blocks_per_day =
              pow_utils.gaps_per_day(miner->primes_per_sec(), target_difficulty);
          cout << "  blocks/day " << fixed << setprecision(2) << blocks_per_day;
        }

        if (opts->has_cset()) {
          const double block_progress = miner->next_share_percent();
          const double block_prob = (1.0 - std::exp(-(block_progress / 100.0))) * 100.0;
          cout << "  gaplist " << ChineseSieve::gaplist_size();
          cout << "  block [" << setprecision(3);
          cout << block_progress << " %]";
          cout << "  block_prob [" << setprecision(3) << block_prob << " %]";
        } else if (target_difficulty > 0 && blocks_per_day > 0.0) {
          const double expected_blocks = (blocks_per_day * (double) target_age_sec) / 86400.0;
          const double block_progress = expected_blocks * 100.0;
          const double block_prob = (1.0 - std::exp(-expected_blocks)) * 100.0;
          cout << "  block [" << setprecision(3);
          cout << block_progress << " %]";
          cout << "  block_prob [" << setprecision(3) << block_prob << " %]";
        }

        if (target_difficulty > 0) {
          cout << "  target_round " << target_round;
          cout << "  target_age " << target_age_sec << "s";
          if (target_is_new)
            cout << "  target NEW";
        }
#else
        const double pps = miner->primes_per_sec();
        const double tests = miner->tests_per_second();
        const double gaps = miner->gaps_per_second();

        rate_samples.push_back({pps, tests, gaps});
        while (rate_samples.size() > max_samples)
          rate_samples.pop_front();

        auto compute_avg = [&](size_t window, double &pps_avg, double &tests_avg, double &gaps_avg) {
          const size_t count = std::min(window, rate_samples.size());
          if (count == 0) {
            pps_avg = tests_avg = gaps_avg = 0.0;
            return;
          }
          double pps_sum = 0.0;
          double tests_sum = 0.0;
          double gaps_sum = 0.0;
          for (size_t i = rate_samples.size() - count; i < rate_samples.size(); ++i) {
            pps_sum += rate_samples[i].pps;
            tests_sum += rate_samples[i].tests;
            gaps_sum += rate_samples[i].gaps;
          }
          pps_avg = pps_sum / static_cast<double>(count);
          tests_avg = tests_sum / static_cast<double>(count);
          gaps_avg = gaps_sum / static_cast<double>(count);
        };

        double pps_avg_60 = 0.0, tests_avg_60 = 0.0, gaps_avg_60 = 0.0;
        double pps_avg_300 = 0.0, tests_avg_300 = 0.0, gaps_avg_300 = 0.0;
        compute_avg(avg60_samples, pps_avg_60, tests_avg_60, gaps_avg_60);
        compute_avg(avg300_samples, pps_avg_300, tests_avg_300, gaps_avg_300);

        cout << get_time();
        cout << fixed << setprecision(2);
        cout << "pps: "      << pps;
        cout << " / "        << miner->avg_primes_per_sec();
        cout << "  tests/s " << tests;
        cout << " / "        << miner->avg_tests_per_second();
        cout << "  gaps/s " << gaps;
        cout << " / "        << miner->avg_gaps_per_second();
        cout << "  avg60s " << pps_avg_60 << "/" << tests_avg_60 << "/" << gaps_avg_60;
        cout << "  avg300s " << pps_avg_300 << "/" << tests_avg_300 << "/" << gaps_avg_300;
        double blocks_per_day = 0.0;
        if (target_difficulty > 0) {
          blocks_per_day =
              pow_utils.gaps_per_day(pps, target_difficulty);
          cout << "  blocks/day " << blocks_per_day;
        }
        cout.unsetf(ios::floatfield);
        cout << setprecision(6);
        if (opts->has_cset()) {
          const double block_progress = miner->next_share_percent();
          const double block_prob = (1.0 - std::exp(-(block_progress / 100.0))) * 100.0;
          cout << "  gaplist " << ChineseSieve::gaplist_size();
          cout << "  block [" << setprecision(3);
          cout << block_progress << " %]";
          cout << "  block_prob [" << setprecision(3) << block_prob << " %]";
        } else if (target_difficulty > 0 && blocks_per_day > 0.0) {
          const double expected_blocks = (blocks_per_day * (double) target_age_sec) / 86400.0;
          const double block_progress = expected_blocks * 100.0;
          const double block_prob = (1.0 - std::exp(-expected_blocks)) * 100.0;
          cout << "  block [" << setprecision(3);
          cout << block_progress << " %]";
          cout << "  block_prob [" << setprecision(3) << block_prob << " %]";
        }

        if (target_difficulty > 0) {
          cout << "  target_round " << target_round;
          cout << "  target_age " << target_age_sec << "s";
          if (target_is_new)
            cout << "  target NEW";
        }

        /* log to CSV if enabled */
        if (stats_csv_file) {
          time_t now = time(NULL);
          fprintf(stats_csv_file, "%ld,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                  now, pps, miner->avg_primes_per_sec(), tests, miner->avg_tests_per_second(),
                  gaps, miner->avg_gaps_per_second(), pps_avg_60, tests_avg_60, gaps_avg_60,
                  pps_avg_300, tests_avg_300, gaps_avg_300);
          fflush(stats_csv_file);
        }
#endif      
        cout << endl;
      }
      pthread_mutex_unlock(&io_mutex);
    }
  }

  Stratum::stop();

  if (!opts->has_stratum())
    pthread_join(thread, NULL);

  /* close CSV file */
  if (stats_csv_file) {
    fclose(stats_csv_file);
  }

  delete miner;

  return 0;
}
