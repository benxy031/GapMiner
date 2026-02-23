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
stats_csv( NULL, "--stats-csv",      "log mining stats to stats.csv",                 false),
threads(   "-t", "--threads",        "number of mining threads",                      true),
pull(      "-l", "--pull-interval",  "seconds to wait between getwork request",       true),
timeout(   "-m", "--timeout",        "seconds to wait for server to respond",         true),
dump_config(NULL, "--dump-config",   "print effective runtime config",                false),
stratum(   "-c", "--stratum",        "use stratum protocol for connection",           false),
/* Sieving parameters: larger values = more candidates filtered, more memory used.
 * Tune based on available RAM and desired candidate quality */
sievesize( "-s", "--sieve-size",     "the prime sieve size",                          true),
primes(    "-i", "--sieve-primes",   "number of primes for sieving",                  true),
/* Shift determines the bit size of primes to search: bit_size = 256 + shift
 * This translates to decimal digit length via: digits = floor(bit_size * log10(2))
 * Valid range: [14, 1024] (protocol enforced), default: 25
 * Examples: shift=13->81 digits, shift=25->85 digits, shift=128->116 digits
 * Larger shifts = larger primes = more computational work per candidate */
shift(     "-f", "--shift",          "the adder shift",                               true),
/* CRT (Chinese Remainder Theorem) optimization: precomputed residue tables in crt/
 * dramatically accelerate sieve initialization. Use --calc-ctr to generate custom tables
 * optimized for specific shift values and sieve parameters */
cset(      "-r", "--crt",            "use the given Chinese Remainder Theorem file",  true),
fermat_threads("-d", "--fermat-threads", "number of fermat threads wen using the crt",    true),
gap_queue_limit(NULL, "--gap-queue-limit", "max queued gaps before sieving pauses (default 8192)", true),
/* Normally, minimum gap length is auto-calculated based on difficulty target.
 * This option overrides that calculation to force testing of all gaps >= specified length.
 * Useful for targeting specific gap sizes or merit ranges */
min_gaplen(NULL, "--min-gaplen", "minimum gap length to test (overrides auto-calculated min)", true),
/* OPT #10: Batched GMP Updates - reduces multi-precision arithmetic overhead by 256x
 * Enabled by default for 3-5x sieve speedup. Use --disable-batched-gmp to revert to original */
disable_batched_gmp(NULL, "--disable-batched-gmp", "disable batched GMP optimization (not recommended)", false),
gmp_batch_size(NULL, "--gmp-batch-size", "number of gaps per GMP sync (default 256)", true),
/* OPT #11: Multi-Stage Primality Testing - cascade Fermat + Miller-Rabin for 2-3x speedup
 * Stage 1: Quick Fermat (filters ~50%), Stage 2: 3-round MR, Stage 3: Full 25-round MR
 * Enabled by default. Use --disable-multistage-tests to revert to single-stage 25-round MR */
disable_multistage_tests(NULL, "--disable-multistage-tests", "disable multi-stage primality testing (not recommended)", false),
#ifndef CPU_ONLY
/* GPU acceleration options: offload Fermat primality tests to GPU for massive parallelism.
 * Key tuning parameters for RTX 3060: -w 4096-8192, -n 32-64, --queue-size 16384-32768
 * Larger work-items and queue-size = more GPU utilization but higher memory usage */
benchmark( "-b", "--benchmark",      "run a gpu benchmark",                           false),
use_gpu(   "-g", "--use-gpu",        "use the gpu for Fermat testing",                false),
gpu_dev(   "-d", "--gpu-dev",        "the gpu device id",                             true),
group_size( "-G", "--group-size",    "gpu group (block) size (threads per block)",    true),
work_items("-w", "--work-items",     "gpu work items (default 2048)",                 true),
queue_size("-z", "--queue-size",     "the gpu waiting queue size (memory intensive)", true),
platform(  "-a", "--platform",       "opencl platform (amd or nvidia)",               true),
n_tests(   "-n", "--num-gpu-tests",  "the number of test per gap per gpu run",        true),
/* CUDA-specific optimizations: experimental sieve on GPU, Comba multiplication algorithm.
 * Memory pools avoid repeated allocation overhead for high-throughput GPU pipelines */
cuda_sieve_proto(NULL, "--cuda-sieve-proto", "run the experimental CUDA sieve prototype", false),
cuda_comba(NULL, "--cuda-comba", "use Comba Montgomery multiply on CUDA", false),
  cuda_comba_soa(NULL, "--use-comba-soa", "use Comba SoA Montgomery multiply on CUDA", false),
bitmap_pool_buffers(NULL, "--bitmap-pool-buffers", "override bitmap buffer pool size (default queue-size+2)", true),
snapshot_pool_buffers(NULL, "--snapshot-pool-buffers", "override CUDA residue snapshot pool size", true),
/* GPU launch control: divisor affects batch size, wait_ms controls GPU starvation vs latency.
 * Lower divisor = larger batches = better throughput but higher latency */
gpu_launch_divisor(NULL, "--gpu-launch-divisor", "override GPU launch divisor (default 6, lower = faster launches)", true),
  gpu_launch_wait_ms(NULL, "--gpu-launch-wait-ms", "maximum wait in ms before forcing a partial GPU batch (default 50)", true),
#endif
/* CRT calculation/optimization: generate custom Chinese Remainder Theorem tables
 * optimized for specific mining parameters. Uses evolutionary algorithms to find
 * prime residue patterns that maximize sieve efficiency. Advanced users only. */
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

  stats_csv.active = has_arg(stats_csv.short_opt, stats_csv.long_opt);

  threads.active = has_arg(threads.short_opt, threads.long_opt);
  if (threads.active)
    threads.arg = get_arg(threads.short_opt, threads.long_opt);

  pull.active = has_arg(pull.short_opt,  pull.long_opt);
  if (pull.active)
    pull.arg = get_arg(pull.short_opt,  pull.long_opt);
                                          
  timeout.active = has_arg(timeout.short_opt,  timeout.long_opt);
  if (timeout.active)
    timeout.arg = get_arg(timeout.short_opt,  timeout.long_opt);

  dump_config.active = has_arg(dump_config.short_opt, dump_config.long_opt);

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

  gap_queue_limit.active = has_arg(gap_queue_limit.short_opt, gap_queue_limit.long_opt);
  if (gap_queue_limit.active)
    gap_queue_limit.arg = get_arg(gap_queue_limit.short_opt, gap_queue_limit.long_opt);

  min_gaplen.active = has_arg(min_gaplen.short_opt, min_gaplen.long_opt);
  if (min_gaplen.active)
    min_gaplen.arg = get_arg(min_gaplen.short_opt, min_gaplen.long_opt);

  disable_batched_gmp.active = has_arg(disable_batched_gmp.short_opt, disable_batched_gmp.long_opt);

  gmp_batch_size.active = has_arg(gmp_batch_size.short_opt, gmp_batch_size.long_opt);
  if (gmp_batch_size.active)
    gmp_batch_size.arg = get_arg(gmp_batch_size.short_opt, gmp_batch_size.long_opt);

  disable_multistage_tests.active = has_arg(disable_multistage_tests.short_opt, disable_multistage_tests.long_opt);


#ifndef CPU_ONLY
  benchmark.active = has_arg(benchmark.short_opt,  benchmark.long_opt);
                                          
  use_gpu.active = has_arg(use_gpu.short_opt,  use_gpu.long_opt);
                                          
  gpu_dev.active = has_arg(gpu_dev.short_opt,  gpu_dev.long_opt);
  if (gpu_dev.active)
    gpu_dev.arg = get_arg(gpu_dev.short_opt,  gpu_dev.long_opt);
                                          
  group_size.active = has_arg(group_size.short_opt, group_size.long_opt);
  if (group_size.active)
    group_size.arg = get_arg(group_size.short_opt, group_size.long_opt);

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

  cuda_comba.active = has_arg(cuda_comba.short_opt, cuda_comba.long_opt);
  cuda_comba_soa.active = has_arg(cuda_comba_soa.short_opt, cuda_comba_soa.long_opt);

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

  ss << "      " << left << setw(18);
  ss << stats_csv.long_opt << "  " << stats_csv.description << "\n\n";

  ss << "  " << threads.short_opt << "  " << left << setw(18);
  ss << threads.long_opt << "  " << threads.description << "\n\n";

  ss << "  " << pull.short_opt  << "  " << left << setw(18);
  ss << pull.long_opt << "  " << pull.description << "\n\n";

  ss << "  " << timeout.short_opt  << "  " << left << setw(18);
  ss << timeout.long_opt << "  " << timeout.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << dump_config.long_opt << "  " << dump_config.description << "\n\n";

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

  ss << "      " << left << setw(18);
  ss << gap_queue_limit.long_opt << "  " << gap_queue_limit.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << min_gaplen.long_opt << "  " << min_gaplen.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << disable_batched_gmp.long_opt << "  " << disable_batched_gmp.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << gmp_batch_size.long_opt << "  " << gmp_batch_size.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << disable_multistage_tests.long_opt << "  " << disable_multistage_tests.description << "\n\n";

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
  ss << cuda_comba.long_opt << "  " << cuda_comba.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << bitmap_pool_buffers.long_opt << "  " << bitmap_pool_buffers.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << snapshot_pool_buffers.long_opt << "  " << snapshot_pool_buffers.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << gpu_launch_divisor.long_opt << "  " << gpu_launch_divisor.description << "\n\n";

  ss << "      " << left << setw(18);
  ss << gpu_launch_wait_ms.long_opt << "  " << gpu_launch_wait_ms.description << "\n\n";
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
