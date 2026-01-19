/**
 * Header file of GapMiners (simple) option parsing 
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
#ifndef __OPTS_H__
#define __OPTS_H__
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sstream>
#include <iomanip> 
#include <iostream>

using namespace std;

/**
 * Singleton class to parse and store command line parameters
 */
class Opts {

  private:

    class SingleOpt {
      
      public:

      /* short opt */
      const char *short_opt;

      /* long opt */
      const char *long_opt;

      /* description */
      const char *description;

      /* has an argument */
      const bool _has_arg;

      /* was given */
      bool active;

      /* argument if given */
      string arg;

      SingleOpt(const char *short_opt,
                const char *long_opt,
                const char *description,
                const bool _has_arg) :
                short_opt(short_opt),
                long_opt(long_opt),
                description(description),
                _has_arg(_has_arg) {

        this->active = false;
        this->arg    = "";
      }
    };

    SingleOpt host;
    SingleOpt port;
    SingleOpt user;
    SingleOpt pass;
    SingleOpt quiet;
    SingleOpt extra_vb;
    SingleOpt share_log;
    SingleOpt stats;
    SingleOpt threads;
    SingleOpt pull;
    SingleOpt timeout;
    SingleOpt stratum;
    SingleOpt sievesize;
    SingleOpt primes;
    SingleOpt shift;
    SingleOpt cset;
    SingleOpt fermat_threads;
#ifndef CPU_ONLY
    SingleOpt benchmark;
    SingleOpt use_gpu;
    SingleOpt gpu_dev;
    SingleOpt work_items;
    SingleOpt queue_size;
    SingleOpt platform;
    SingleOpt n_tests;
    SingleOpt cuda_sieve_proto;
    SingleOpt bitmap_pool_buffers;
    SingleOpt snapshot_pool_buffers;
  SingleOpt gpu_launch_divisor;
  SingleOpt gpu_launch_wait_ms;
#endif    
    SingleOpt calc_ctr;
    SingleOpt ctr_strength;
    SingleOpt ctr_primes;
    SingleOpt ctr_evolution;
    SingleOpt ctr_fixed;
    SingleOpt ctr_n_ivs;
    SingleOpt ctr_range;
    SingleOpt ctr_bits;
    SingleOpt ctr_merit;
    SingleOpt ctr_file;
    SingleOpt help;
    SingleOpt license;

    Opts(int argc, char *argv[]);
    ~Opts() { }

    /* synchronization mutexes */
    static pthread_mutex_t creation_mutex;

  public:

    /* the only instance of this */
    static Opts *only_instance;

    /* access or create the only instance of this */
    static Opts *get_instance(int argc = 0, char *argv[] = NULL);

    /* read functions */
    bool has_host()             { return host.active;           }
    string get_host()           { return host.arg;              }
                                                                
    bool has_port()             { return port.active;           }
    string get_port()           { return port.arg;              }
                                                                
    bool has_user()             { return user.active;           }
    string get_user()           { return user.arg;              }
                                                                
    bool has_pass()             { return pass.active;           }
    string get_pass()           { return pass.arg;              }
                                                                
    bool has_quiet()            { return quiet.active;          }
                                                                
    bool has_extra_vb()         { return extra_vb.active;       }
                                                                
    bool log_shares()           { return share_log.active;      }

    bool has_stats()            { return stats.active;          }
    string get_stats()          { return stats.arg;             }
                                                                
    bool has_threads()          { return threads.active;        }
    string get_threads()        { return threads.arg;           }
                                                                
    bool has_pull()             { return pull.active;           }
    string get_pull()           { return pull.arg;              }
                                                                
    bool has_timeout()          { return timeout.active;        }
    string get_timeout()        { return timeout.arg;           }
                                                                
    bool has_stratum()          { return stratum.active;        }
                                                                
    bool has_sievesize()        { return sievesize.active;      }
    string get_sievesize()      { return sievesize.arg;         }
                                                                
    bool has_primes()           { return primes.active;         }
    string get_primes()         { return primes.arg;            }
                                                                
    bool has_shift()            { return shift.active;          }
    string get_shift()          { return shift.arg;             }
                                                                
    bool has_cset()             { return cset.active;           }
    string get_cset()           { return cset.arg;              }

    bool has_fermat_threads()   { return fermat_threads.active; }
    string get_fermat_threads() { return fermat_threads.arg;    }
                                                        
#ifndef CPU_ONLY                                                        
    bool has_benchmark()        { return benchmark.active;      }
                                                                
    bool has_use_gpu()          { return use_gpu.active;        }
                                                                
    bool has_gpu_dev()          { return gpu_dev.active;        }
    string get_gpu_dev()        { return gpu_dev.arg;           }
                                                                
    bool has_work_items()       { return work_items.active;     }
    string get_work_items()     { return work_items.arg;        }
                                                                
    bool has_queue_size()       { return queue_size.active;     }
    string get_queue_size()     { return queue_size.arg;        }
                                                                
    bool has_platform()         { return platform.active;       }
    string get_platform()       { return platform.arg;          }
                                                                
    bool has_n_tests()          { return n_tests.active;        }
    string get_n_tests()        { return n_tests.arg;           }

    bool use_cuda_sieve_proto() { return cuda_sieve_proto.active; }

    bool has_bitmap_pool_buffers()   { return bitmap_pool_buffers.active; }
    string get_bitmap_pool_buffers() { return bitmap_pool_buffers.arg;    }

    bool has_snapshot_pool_buffers()   { return snapshot_pool_buffers.active; }
    string get_snapshot_pool_buffers() { return snapshot_pool_buffers.arg;    }

    bool has_gpu_launch_divisor()   { return gpu_launch_divisor.active; }
    string get_gpu_launch_divisor() { return gpu_launch_divisor.arg;    }

    bool has_gpu_launch_wait_ms()   { return gpu_launch_wait_ms.active; }
    string get_gpu_launch_wait_ms() { return gpu_launch_wait_ms.arg;    }
#endif    

    bool has_calc_ctr()         { return calc_ctr.active;       }
    string get_calc_ctr()       { return calc_ctr.arg;          }

    bool has_ctr_strength()     { return ctr_strength.active;   }
    string get_ctr_strength()   { return ctr_strength.arg;      }

    bool has_ctr_primes()       { return ctr_primes.active;     }
    string get_ctr_primes()     { return ctr_primes.arg;        }

    bool has_ctr_evolution()    { return ctr_evolution.active;  }
    string get_ctr_evolution()  { return ctr_evolution.arg;     }

    bool has_ctr_fixed()        { return ctr_fixed.active;      }
    string get_ctr_fixed()      { return ctr_fixed.arg;         }

    bool has_ctr_n_ivs()        { return ctr_n_ivs.active;      }
    string get_ctr_n_ivs()      { return ctr_n_ivs.arg;         }

    bool has_ctr_range()        { return ctr_range.active;      }
    string get_ctr_range()      { return ctr_range.arg;         }

    bool has_ctr_bits()         { return ctr_bits.active;       }
    string get_ctr_bits()       { return ctr_bits.arg;          }

    bool has_ctr_merit()        { return ctr_merit.active;      }
    string get_ctr_merit()      { return ctr_merit.arg;         }
                                                  
    bool has_ctr_file()         { return ctr_file.active;       }
    string get_ctr_file()       { return ctr_file.arg;          }
                                                  
    bool has_help()             { return help.active;           }
                                                                
    bool has_license()          { return license.active;        }
                                                    
    /* get help */                                
    string get_help();                            
};                                              

#endif /* __OPTS_H__ */
