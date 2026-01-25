/**
 * Implementation of GapMiners (simple) option parsing 
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
#include "Opts.h"

/**
 * Returns if argv contains the given arg
 */
static char has_arg(int argc, 
                    char *argv[], 
                    const char *short_arg, 
                    const char *long_arg) {

  int i;
  for (i = 1; i < argc; i++) {
    if ((short_arg != NULL && !strcmp(argv[i], short_arg)) ||
        (long_arg != NULL  && !strcmp(argv[i], long_arg))) {
      
      return 1;
    }
  }

  return 0;
}

/**
 * Returns the given argument of arg
 */
static char *get_arg(int argc, 
                     char *argv[], 
                     const char *short_arg, 
                     const char *long_arg) {

  int i;
  for (i = 1; i < argc - 1; i++) {
    if ((short_arg != NULL && !strcmp(argv[i], short_arg)) ||
        (long_arg != NULL  && !strcmp(argv[i], long_arg))) {
      
      return argv[i + 1];
    }
  }

  return NULL;
}

/**
 * shorter macros 
 */
#define has_arg1(long_arg) has_arg(argc, argv, NULL, long_arg)
#define has_arg2(short_arg, long_arg) has_arg(argc, argv, short_arg, long_arg)
#define has_arg4(argc, argv, short_arg, long_arg) \
  has_arg(argc, argv, short_arg, long_arg)
#define has_argx(X, T1, T2, T3, T4, FUNC, ...) FUNC
#define has_arg(...) has_argx(, ##__VA_ARGS__,            \
                                has_arg4(__VA_ARGS__),    \
                                "3 args not allowed",     \
                                has_arg2(__VA_ARGS__),    \
                                has_arg1(__VA_ARGS__))

#define get_arg1(long_arg) get_arg(argc, argv, NULL, long_arg)
#define get_arg2(short_arg, long_arg) get_arg(argc, argv, short_arg, long_arg)
#define get_arg4(argc, argv, short_arg, long_arg) \
  get_arg(argc, argv, short_arg, long_arg)
#define get_argx(X, T1, T2, T3, T4, FUNC, ...) FUNC
#define get_arg(...) get_argx(, ##__VA_ARGS__,            \
                                get_arg4(__VA_ARGS__),    \
                                "3 args not allowed",     \
                                get_arg2(__VA_ARGS__),    \
                                get_arg1(__VA_ARGS__))

/**
 * to get integer args
 */
#define get_i_arg(...)                                          \
  (has_arg(__VA_ARGS__) ? atoi(get_arg(__VA_ARGS__)) : -1)

#define get_l_arg(...)                                          \
  (has_arg(__VA_ARGS__) ? atol(get_arg(__VA_ARGS__)) : -1)

#define get_ll_arg(...)                                         \
  (has_arg(__VA_ARGS__) ? atoll(get_arg(__VA_ARGS__)) : -1)

/**
 * to get float args
 */
#define get_f_arg(...)                                         \
  (has_arg(__VA_ARGS__) ? atof(get_arg(__VA_ARGS__)) : -1.0)


/* only instance of this */
Opts *Opts::only_instance = NULL;

/* synchronization mutexes */
pthread_mutex_t Opts::creation_mutex = PTHREAD_MUTEX_INITIALIZER;

/* initializes all possible args */
Opts::Opts(int argc, char *argv[]) :
host(      "-o", "--host",           "host ip address",                               true),
port(      "-p", "--port",           "port to connect to",                            true),
user(      "-u", "--user",           "user for gapcoin rpc authentification",         true),
pass(      "-x", "--pwd",            "password for gapcoin rpc authentification",     true),
quiet(     "-q", "--quiet",          "be quiet (only prints shares)",                 false),
extra_vb(  "-e", "--extra-verbose",  "additional verbose output",                     false),
share_log( "-S", "--share-log",      "append share results to shares.txt",            false),
stats(     "-j", "--stats-interval", "interval (sec) to print mining informations",   true),
threads(   "-t", "--threads",        "number of mining threads",                      true),
pull(      "-l", "--pull-interval",  "seconds to wait between getwork request",       true),
timeout(   "-m", "--timeout",        "seconds to wait for server to respond",         true),
stratum(   "-c", "--stratum",        "use stratum protocol for connection",           false),
sievesize( "-s", "--sieve-size",     "the prime sieve size",                          true),
primes(    "-i", "--sieve-primes",   "number of primes for sieving",                  true),
shift(     "-f", "--shift",          "the adder shift",                               true),
cset(      "-r", "--crt",            "use the given Chinese Remainder Theorem file",  true),
fermat_threads("-d", "--fermat-threads", "number of fermat threads wen using the crt",    true),
#ifndef CPU_ONLY
benchmark( "-b", "--benchmark",      "run a gpu benchmark",                           false),
use_gpu(   "-g", "--use-gpu",        "use the gpu for Fermat testing",                false),
gpu_dev(   "-d", "--gpu-dev",        "the gpu device id",                             true),
work_items("-w", "--work-items",     "gpu work items (default 2048)",                 true),
queue_size("-z", "--queue-size",     "the gpu waiting queue size (memory intensive)", true),
platform(  "-a", "--platform",       "opencl platform (amd or nvidia)",               true),
n_tests(   "-n", "--num-gpu-tests",  "the number of test per gap per gpu run",        true),
cuda_sieve_proto(NULL, "--cuda-sieve-proto", "run the experimental CUDA sieve prototype", false),
bitmap_pool_buffers(NULL, "--bitmap-pool-buffers", "override bitmap buffer pool size (default queue-size+2)", true),
snapshot_pool_buffers(NULL, "--snapshot-pool-buffers", "override CUDA residue snapshot pool size", true),
gpu_launch_divisor(NULL, "--gpu-launch-divisor", "override GPU launch divisor (default 6, lower = faster launches)", true),
gpu_launch_wait_ms(NULL, "--gpu-launch-wait-ms", "maximum wait in ms before forcing a partial GPU batch (default 50)", true),
  min_gap(NULL, "--min-gap", "minimum prime gap length to accept (overrides auto calculation)", true),
#endif
calc_ctr(  NULL, "--calc-ctr",       "calculate a chinese remainder theorem file",    false),
ctr_strength(NULL, "--ctr-strength", "more = longer time and mybe better result",     true),
ctr_primes(NULL, "--ctr-primes",     "the number of to use primes in the ctr file",   true),
ctr_evolution(NULL, "--ctr-evolution",  "whether to use evolutional algorithm",       false),
ctr_fixed( NULL, "--ctr-fixed" ,     "the number of fixed starting prime offsets",    true),
ctr_n_ivs( NULL, "--ctr-ivs",        "the number of individuals used in the evolution", true),
ctr_range( NULL, "--ctr-range" ,     "percent deviation from the number of primes",   true),
ctr_bits(  NULL, "--ctr-bits",       "additional bits added to the primorial",        true),
ctr_merit( NULL, "--ctr-merit",      "the target merit",                              true),
ctr_file(  NULL, "--ctr-file",       "the target ctr file",                           true),
help(      "-h", "--help",           "print this information",                        false),
license(   "-v", "--license",        "show license of this program",                  false) {
       
  /* get command line opts */
  host.active = has_arg(host.short_opt,  host.long_opt);
  if (host.active)
    host.arg = get_arg(host.short_opt,  host.long_opt);
                                          
  port.active = has_arg(port.short_opt,  port.long_opt);
  if (port.active)
    port.arg = get_arg(port.short_opt,  port.long_opt);
                                          
  user.active = has_arg(user.short_opt,  user.long_opt);
  if (user.active)
    user.arg = get_arg(user.short_opt,  user.long_opt);
                                          
  pass.active = has_arg(pass.short_opt,  pass.long_opt);
  if (pass.active)
    pass.arg = get_arg(pass.short_opt,  pass.long_opt);

  quiet.active = has_arg(quiet.short_opt, quiet.long_opt);
  extra_vb.active = has_arg(extra_vb.short_opt,  extra_vb.long_opt);
  share_log.active = has_arg(share_log.short_opt, share_log.long_opt);

  stats.active = has_arg(stats.short_opt, stats.long_opt);
  if (stats.active)
    stats.arg = get_arg(stats.short_opt, stats.long_opt);

  threads.active = has_arg(threads.short_opt, threads.long_opt);
  if (threads.active)
    threads.arg = get_arg(threads.short_opt, threads.long_opt);

  pull.active = has_arg(pull.short_opt,  pull.long_opt);
  if (pull.active)
    pull.arg = get_arg(pull.short_opt,  pull.long_opt);
                                          
  timeout.active = has_arg(timeout.short_opt,  timeout.long_opt);
  if (timeout.active)
    timeout.arg = get_arg(timeout.short_opt,  timeout.long_opt);

  stratum.active = has_arg(stratum.short_opt,  stratum.long_opt);
                                          
  sievesize.active = has_arg(sievesize.short_opt,  sievesize.long_opt);
  if (sievesize.active)
    sievesize.arg = get_arg(sievesize.short_opt,  sievesize.long_opt);
                                          
  primes.active = has_arg(primes.short_opt,  primes.long_opt);
  if (primes.active)
    primes.arg = get_arg(primes.short_opt,  primes.long_opt);
                                          
  shift.active = has_arg(shift.short_opt,  shift.long_opt);
  if (shift.active)
    shift.arg = get_arg(shift.short_opt,  shift.long_opt);

  cset.active = has_arg(cset.short_opt,  cset.long_opt);
  if (cset.active)
    cset.arg = get_arg(cset.short_opt,  cset.long_opt);

  fermat_threads.active = has_arg(fermat_threads.short_opt,  fermat_threads.long_opt);
  if (fermat_threads.active)
    fermat_threads.arg = get_arg(fermat_threads.short_opt,  fermat_threads.long_opt);


#ifndef CPU_ONLY
  benchmark.active = has_arg(benchmark.short_opt,  benchmark.long_opt);
                                          
  use_gpu.active = has_arg(use_gpu.short_opt,  use_gpu.long_opt);
                                          
  gpu_dev.active = has_arg(gpu_dev.short_opt,  gpu_dev.long_opt);
  if (gpu_dev.active)
    gpu_dev.arg = get_arg(gpu_dev.short_opt,  gpu_dev.long_opt);
                                          
  work_items.active = has_arg(work_items.short_opt,  work_items.long_opt);
  if (work_items.active)
    work_items.arg = get_arg(work_items.short_opt,  work_items.long_opt);
                                          
  queue_size.active = has_arg(queue_size.short_opt,  queue_size.long_opt);
  if (queue_size.active)
    queue_size.arg = get_arg(queue_size.short_opt,  queue_size.long_opt);

  platform.active = has_arg(platform.short_opt,  platform.long_opt);
  if (platform.active)
    platform.arg = get_arg(platform.short_opt,  platform.long_opt);

  n_tests.active = has_arg(n_tests.short_opt,  n_tests.long_opt);
  if (n_tests.active)
    n_tests.arg = get_arg(n_tests.short_opt,  n_tests.long_opt);

  cuda_sieve_proto.active = has_arg(cuda_sieve_proto.short_opt,
                                    cuda_sieve_proto.long_opt);

  bitmap_pool_buffers.active = has_arg(bitmap_pool_buffers.short_opt,
                                       bitmap_pool_buffers.long_opt);
  if (bitmap_pool_buffers.active)
    bitmap_pool_buffers.arg = get_arg(bitmap_pool_buffers.short_opt,
                                      bitmap_pool_buffers.long_opt);

  snapshot_pool_buffers.active = has_arg(snapshot_pool_buffers.short_opt,
                                         snapshot_pool_buffers.long_opt);
  if (snapshot_pool_buffers.active)
    snapshot_pool_buffers.arg = get_arg(snapshot_pool_buffers.short_opt,
                                        snapshot_pool_buffers.long_opt);

  gpu_launch_divisor.active = has_arg(gpu_launch_divisor.short_opt,
                                      gpu_launch_divisor.long_opt);
  if (gpu_launch_divisor.active)
    gpu_launch_divisor.arg = get_arg(gpu_launch_divisor.short_opt,
                                     gpu_launch_divisor.long_opt);

  gpu_launch_wait_ms.active = has_arg(gpu_launch_wait_ms.short_opt,
                                      gpu_launch_wait_ms.long_opt);
  if (gpu_launch_wait_ms.active)
    gpu_launch_wait_ms.arg = get_arg(gpu_launch_wait_ms.short_opt,
                                     gpu_launch_wait_ms.long_opt);
  min_gap.active = has_arg(min_gap.short_opt, min_gap.long_opt);
  if (min_gap.active)
    min_gap.arg = get_arg(min_gap.short_opt, min_gap.long_opt);
#endif    

  calc_ctr.active = has_arg(calc_ctr.short_opt, calc_ctr.long_opt);

  ctr_strength.active = has_arg(ctr_strength.short_opt, ctr_strength.long_opt);
  if (ctr_strength.active)
    ctr_strength.arg = get_arg(ctr_strength.short_opt, ctr_strength.long_opt);

  ctr_primes.active = has_arg(ctr_primes.short_opt, ctr_primes.long_opt);
  if (ctr_primes.active)
    ctr_primes.arg = get_arg(ctr_primes.short_opt, ctr_primes.long_opt);

  ctr_evolution.active = has_arg(ctr_evolution.short_opt, ctr_evolution.long_opt);

  ctr_fixed.active = has_arg(ctr_fixed.short_opt, ctr_fixed.long_opt);
  if (ctr_fixed.active)
    ctr_fixed.arg = get_arg(ctr_fixed.short_opt, ctr_fixed.long_opt);

  ctr_n_ivs.active = has_arg(ctr_n_ivs.short_opt, ctr_n_ivs.long_opt);
  if (ctr_n_ivs.active)
    ctr_n_ivs.arg = get_arg(ctr_n_ivs.short_opt, ctr_n_ivs.long_opt);

  ctr_range.active = has_arg(ctr_range.short_opt, ctr_range.long_opt);
  if (ctr_range.active)
    ctr_range.arg = get_arg(ctr_range.short_opt, ctr_range.long_opt);

  ctr_bits.active = has_arg(ctr_bits.short_opt, ctr_bits.long_opt);
  if (ctr_bits.active)
    ctr_bits.arg = get_arg(ctr_bits.short_opt, ctr_bits.long_opt);

  ctr_merit.active = has_arg(ctr_merit.short_opt, ctr_merit.long_opt);
  if (ctr_merit.active)
    ctr_merit.arg = get_arg(ctr_merit.short_opt, ctr_merit.long_opt);

  ctr_file.active = has_arg(ctr_file.short_opt, ctr_file.long_opt);
  if (ctr_file.active)
    ctr_file.arg = get_arg(ctr_file.short_opt, ctr_file.long_opt);
                                          
  help.active = has_arg(help.short_opt,  help.long_opt);
  license.active = has_arg(license.short_opt,  license.long_opt);
}

/* access or create the only instance of this */
Opts *Opts::get_instance(int argc, char *argv[]) {
  
  pthread_mutex_lock(&creation_mutex);

  /* allow only one creation */
  if (argc != 0 && argv != NULL && only_instance == NULL) {
    only_instance = new Opts(argc, argv);
  }

  pthread_mutex_unlock(&creation_mutex);

  return only_instance;
}


/* get help */
string Opts::get_help()  { 
  stringstream ss;

  ss << "  GapMiner  Copyright (C)  2014  The Gapcoin developers  <info@gapcoin.org>\n\n";
  ss << "Required Options:   \n\n";

  ss << "  " << host.short_opt  << "  " << left << setw(18);
  ss << host.long_opt << "  " << host.description << "\n\n";

  ss << "  " << port.short_opt << "  " << left << setw(18);
  ss << port.long_opt << "  " << port.description << "\n\n";

  ss << "  " << user.short_opt << "  " << left << setw(18);
  ss << user.long_opt << "  " << user.description << "\n\n";

  ss << "  " << pass.short_opt << "  " << left << setw(18);
  ss << pass.long_opt << "  " << pass.description << "\n\n";

  ss << "Additional Options:\n\n";

  ss << "  " << quiet.short_opt << "  " << left << setw(18);
  ss << quiet.long_opt << "  " << quiet.description << "\n\n";

  ss << "  " << extra_vb.short_opt  << "  " << left << setw(18);
  ss << extra_vb.long_opt << "  " << extra_vb.description << "\n\n";

  ss << "  " << share_log.short_opt << "  " << left << setw(18);
  ss << share_log.long_opt << "  " << share_log.description << "\n\n";

  ss << "  " << stats.short_opt << "  " << left << setw(18);
  ss << stats.long_opt << "  " << stats.description << "\n\n";

  ss << "  " << threads.short_opt << "  " << left << setw(18);
  ss << threads.long_opt << "  " << threads.description << "\n\n";

  ss << "  " << pull.short_opt  << "  " << left << setw(18);
  ss << pull.long_opt << "  " << pull.description << "\n\n";

  ss << "  " << timeout.short_opt  << "  " << left << setw(18);
  ss << timeout.long_opt << "  " << timeout.description << "\n\n";

  ss << "  " << stratum.short_opt  << "  " << left << setw(18);
  ss << stratum.long_opt << "  " << stratum.description << "\n\n";

  ss << "  " << sievesize.short_opt  << "  " << left << setw(18);
  ss << sievesize.long_opt << "  " << sievesize.description << "\n\n";

  ss << "  " << primes.short_opt  << "  " << left << setw(18);
  ss << primes.long_opt << "  " << primes.description << "\n\n";

  ss << "  " << shift.short_opt  << "  " << left << setw(18);
  ss << shift.long_opt << "  " << shift.description << "\n\n";

  ss << "  " << cset.short_opt  << "  " << left << setw(18);
  ss << cset.long_opt << "  " << cset.description << "\n\n";

  ss << "  " << fermat_threads.short_opt  << "  " << left << setw(18);
  ss << fermat_threads.long_opt << "  " << fermat_threads.description << "\n\n";

#ifndef CPU_ONLY
  ss << "  " << benchmark.short_opt  << "  " << left << setw(18);
  ss << benchmark.long_opt << "  " << benchmark.description << "\n\n";

  ss << "  " << use_gpu.short_opt  << "  " << left << setw(18);
  ss << use_gpu.long_opt << "  " << use_gpu.description << "\n\n";

  ss << "  " << gpu_dev.short_opt  << "  " << left << setw(18);
  ss << gpu_dev.long_opt << "  " << gpu_dev.description << "\n\n";

  ss << "  " << work_items.short_opt  << "  " << left << setw(18);
  ss << work_items.long_opt << "  " << work_items.description << "\n\n";

  ss << "  " << queue_size.short_opt  << "  " << left << setw(18);
  ss << queue_size.long_opt << "  " << queue_size.description << "\n\n";

  ss << "  " << platform.short_opt  << "  " << left << setw(18);
  ss << platform.long_opt << "  " << platform.description << "\n\n";

  ss << "  " << n_tests.short_opt  << "  " << left << setw(18);
  ss << n_tests.long_opt << "  " << n_tests.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << cuda_sieve_proto.long_opt << "  " << cuda_sieve_proto.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << bitmap_pool_buffers.long_opt << "  " << bitmap_pool_buffers.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << snapshot_pool_buffers.long_opt << "  " << snapshot_pool_buffers.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << gpu_launch_divisor.long_opt << "  " << gpu_launch_divisor.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << gpu_launch_wait_ms.long_opt << "  " << gpu_launch_wait_ms.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << min_gap.long_opt << "  " << min_gap.description << "\n\n";
#endif  

  ss << "      " << left << setw(18);
  ss << calc_ctr.long_opt << "  " << calc_ctr.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << ctr_strength.long_opt << "  " << ctr_strength.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << ctr_primes.long_opt << "  " << ctr_primes.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << ctr_evolution.long_opt << "  " << ctr_evolution.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << ctr_fixed.long_opt << "  " << ctr_fixed.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << ctr_n_ivs.long_opt << "  " << ctr_n_ivs.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << ctr_range.long_opt << "  " << ctr_range.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << ctr_bits.long_opt << "  " << ctr_bits.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << ctr_merit.long_opt << "  " << ctr_merit.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << ctr_file.long_opt << "  " << ctr_file.description << "\n\n";

  ss << "  " << help.short_opt << "  " << left << setw(18);
  ss << help.long_opt << "  " << help.description << "\n\n";

  ss << "  " << license.short_opt << "  " << left << setw(18);
  ss << license.long_opt << "  " << license.description << "\n\n";

  return ss.str();
}
