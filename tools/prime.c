/**
 * implementation of the default Sieve of Eratosthenes to generat 
 * a prime table including all primes till a given number starting by 2
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "prime.h"
#include "../../Sort/src/sort.h"


/**
 * Sets the given bit-possition in an byte array
 */
#define set_bit(ary, i) (ary[(i) / 8] |= (1 << ((i) % 8)))
    
/**
 * Unsets the given bit-possition in an byte array
 */
#define unset_bit(ary, i) (ary[(i) / 8] &= ~(1 << ((i) % 8)))

/**
 * returns wether the given bit-possition is an byte array is setted or not
 */
#define bit_at(ary, i) (ary[(i) / 8] & (1 << ((i) % 8)))

/**
 * returns wether the given index is a prime or not
 */
#define is_prime(ary, i) !bit_at(ary, i)

/**
 * generate the X**2
 */
#define POW(X) ((X) * (X))

#define max(x, y) ((x > y) ? x : y)

/**
 * simple binary search to find an index with the next bigger prime
 */
static inline uint64_t find_next_prime(uint64_t *ary, uint64_t len, uint64_t val);

/**
 * first 1001 primes hardcoded
 */
uint64_t first1001prims[] = {
     2,    3,    5,    7,   11,   13,   17,   19,   23,   29,   31,   37,   41, 
    43,   47,   53,   59,   61,   67,   71,   73,   79,   83,   89,   97,  101, 
   103,  107,  109,  113,  127,  131,  137,  139,  149,  151,  157,  163,  167, 
   173,  179,  181,  191,  193,  197,  199,  211,  223,  227,  229,  233,  239, 
   241,  251,  257,  263,  269,  271,  277,  281,  283,  293,  307,  311,  313, 
   317,  331,  337,  347,  349,  353,  359,  367,  373,  379,  383,  389,  397, 
   401,  409,  419,  421,  431,  433,  439,  443,  449,  457,  461,  463,  467, 
   479,  487,  491,  499,  503,  509,  521,  523,  541,  547,  557,  563,  569, 
   571,  577,  587,  593,  599,  601,  607,  613,  617,  619,  631,  641,  643, 
   647,  653,  659,  661,  673,  677,  683,  691,  701,  709,  719,  727,  733, 
   739,  743,  751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823, 
   827,  829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911, 
   919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997, 1009, 
  1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 
  1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 
  1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 
  1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 
  1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 
  1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 
  1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 
  1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 
  1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 
  1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 
  1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 
  2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 
  2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 
  2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 
  2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 
  2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 
  2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 
  2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 
  2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 
  2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 
  2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 
  3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 
  3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 
  3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 
  3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 
  3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 
  3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 
  3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 
  3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 
  3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 
  4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 
  4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 
  4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 
  4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 
  4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 
  4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 
  4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 
  4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 
  4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 
  5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 
  5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 
  5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 
  5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 
  5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 
  5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 
  5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 
  5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 
  5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 
  6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 
  6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 
  6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 
  6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 
  6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 
  6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 
  6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 
  6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 
  6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 
  7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 
  7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 
  7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 
  7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 
  7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 
  7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 
  7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, 7927, 
};

/**
 * save a given PrimeTable to a file
 */
void save_primes(PrimeTable *primes, const char *fname) {

  int fd = open(fname, O_CREAT|O_WRONLY, S_IRWXU|S_IRGRP|S_IROTH);

#ifdef VERBOSE
  printf("\r[II] writing primes...");
#endif

  write(fd, &primes->len, sizeof(uint64_t));
  write(fd, primes->ptr,  sizeof(uint64_t) * primes->len);
  close(fd);

#ifdef VERBOSE
  printf("\n");
#endif
}

/**
 * reads a given PrimeTable from a file
 */
PrimeTable *read_primes(const char *fname) {

  PrimeTable *primes = malloc(sizeof(PrimeTable));

  if (!access(fname, R_OK)) {
    int fd = open(fname, O_RDONLY);

    if (read(fd, &primes->len, sizeof(uint64_t)) == -1) {
      perror("failed to read num primes");
      exit(EXIT_FAILURE);
    }

    primes->ptr = malloc(sizeof(uint64_t) * primes->len);

    if (primes->ptr == NULL) {
      perror("failed to alloc space");
      exit(EXIT_FAILURE);
    }

#ifdef VERBOSE
    printf("[II] reading primes...");
#endif

    if (read(fd, primes->ptr, sizeof(uint64_t) * primes->len) == -1) {
      perror("failed to read primes");
      exit(EXIT_FAILURE);
    }

#ifdef VERBOSE
    printf("\n");
#endif

    close(fd);
  } else
    return NULL;

  return primes;
}
      

/**
 * find and returns the best prime gap in 
 */
PrimeGap *find_biggest_gap(PrimeTable *primes) {

  PrimeGap *gap = malloc(sizeof(PrimeGap));

  gap->start = 2;
  gap->end   = 3;
  gap->val   = 2.0 / log(2);

  uint64_t i;
  for (i = 1; i < (primes->len - 1); i++) {
    
    double cur = ((double) (primes->ptr[i + 1] - primes->ptr[i])) / 
                  log((double) primes->ptr[i]);

    if (cur > gap->val) {
#ifdef VERBOSE
      printf("%5" PRIu64 ": %10" PRIu64 " -> %10" PRIu64 "  =>  %F\n",
             primes->ptr[i + 1] - primes->ptr[i],
             primes->ptr[i],
             primes->ptr[i + 1],
         cur);
#endif
    
      gap->start = primes->ptr[i];
      gap->end   = primes->ptr[i + 1];
      gap->val   = cur; 
    }
  }

  return gap;
}


/**
 * returns the number of sequece numbers in the sive
 */
static uint64_t count_sequence(uint8_t *ary, uint64_t sieve_size, uint64_t x) {

  /* 1 are not counted in the loop */
  uint64_t i, n = 1;

  /** 
   * run the sieve in seps of size 6 
   * (each prime > 3 will be on one of the folloing palces: 6n +/- 1)
   */
  for (i = first1001prims[x]; i < sieve_size; i++ ) {

#ifdef VERBOSE
    if (i % 2134567 == 0)
      printf("counting sieve %" PRIu64 "\n", sieve_size - i);
#endif

    if (is_prime(ary, i))
      n++;
  }

  return n;
}


/**
 * returns the number of primes in the sive
 */
static uint64_t count_sieve(uint8_t *ary, uint64_t sieve_size) {

  /* 2 and 3 are not counted in the loop */
  uint64_t i, n = 2;

  /** 
   * run the sieve in seps of size 6 
   * (each prime > 3 will be on one of the folloing palces: 6n +/- 1)
   */
  for (i = 5; i < sieve_size; i += 4 ) {

#ifdef VERBOSE
    if (i % 2134567 == 0)
      printf("counting sieve %" PRIu64 "\n", sieve_size - i);
#endif

    /* check 6n - 1*/
    if (is_prime(ary, i))
      n++;

    i += 2; /* check 6n + 1 */
    if (is_prime(ary, i))
      n++;
  }

  return n;
}

/**
 * saves the primes and frees the sieve
 */
static PrimeTable *generate_prime_table(uint8_t *ary, uint64_t sieve_size) {

#ifdef VERBOSE
  uint64_t index = 0;
#endif
  
  PrimeTable *table = (PrimeTable *) malloc(sizeof(PrimeTable));

  if (table == NULL)
    perror("failed to alloc space for the prime table");

  table->len = count_sieve(ary, sieve_size);
  table->ptr = (uint64_t *) malloc(sizeof(uint64_t) * table->len);

  /* 2 and 3 are not counted in the loop */
  uint64_t i, n = 2;
  table->ptr[0] = 2;
  table->ptr[1] = 3;

  /** 
   * run the sieve in seps of size 6 
   * (each prime > 3 will be on one of the folloing palces: 6n +/- 1)
   */
  for (i = 5; i < sieve_size; i += 4 ) {

#ifdef VERBOSE
    if (index % 1234567 == 0) {
      printf("generating prime table: %" PRIu64 "\n", sieve_size - i);
    }
    index++;
#endif

    /* check 6n - 1*/
    if (is_prime(ary, i)) {
      table->ptr[n] = i;
      n++;
    }

    i += 2; /* check 6n + 1 */
    if (is_prime(ary, i)) {
      table->ptr[n] = i;
      n++;
    }
  }

  free(ary);
  return table;
}

/**
 * generate the prime table for a prime sequence
 * (skipps n primes) (2 and 3 are always skipped)
 */
static inline void generate_seq_table(uint8_t *ary, 
                                      uint64_t sieve_size, 
                                      uint64_t *primes,
                                      uint64_t n) {
#ifdef VERBOSE
  uint64_t index = 0;
#endif

  /** 
   * loop over the first view primes (they are not included in the sequence
   */
  uint64_t i = 5, s = 0;
  for (; i < first1001prims[n]; i += 4 ) {

#ifdef VERBOSE
    if (index % 1234567 == 0) {
      printf("generating prime table: %" PRIu64 "\n", sieve_size - i);
    }
    index++;
#endif

    /* check 6n - 1*/
    if (is_prime(ary, i) && i >= first1001prims[n]) {
      primes[s] = i;
      s++;
    }

    i += 2; /* check 6n + 1 */
    if (is_prime(ary, i) && i >= first1001prims[n]) {
      primes[s] = i;
      s++;
    }
  }

  /** 
   * collect the rest of the sequence numbers
   * run the sieve in seps of size 6 
   * (each prime > 3 will be on one of the folloing palces: 6n +/- 1)
   */
  for (; i < sieve_size - 2; i += 4 ) {

#ifdef VERBOSE
    if (index % 1234567 == 0) {
      printf("generating prime table: %" PRIu64 "\n", sieve_size - i);
    }
    index++;
#endif

    /* check 6n - 1*/
    if (is_prime(ary, i)) {
      primes[s] = i;
      s++;
    }

    i += 2; /* check 6n + 1 */
    if (is_prime(ary, i)) {
      primes[s] = i;
      s++;
    }
  }

  /** 
   * collect the rest of the sequence numbers
   * run the sieve in seps of size 6 
   * (each prime > 3 will be on one of the folloing palces: 6n +/- 1)
   */
  for (; i < sieve_size; i++ ) {

    /* check 6n - 1*/
    if (is_prime(ary, i)) {
      primes[s] = i;
      s++;
    }
  }
}

/**
 * 
 */
PrimeTable *gen_prime_table(uint64_t sieve_size) {

#ifdef VERBOSE
  uint64_t n = 1;
  uint64_t index = 0;
#endif

  /* bit array for sieveing */
  uint8_t *ary = (uint8_t *) calloc(sieve_size / 8, sizeof(uint8_t));

  if (ary == NULL) 
    perror("failed to allocate space for prime table generation");
  
  uint64_t i, p, limit = (uint64_t) (sqrt((double) sieve_size) + 1);

  /* 0 and 1 ar not primes */
  set_bit(ary, 0);
  set_bit(ary, 1);

  /* sieve 2 and 3 first */
  for (i = 2; i < 4; i++) {
    
#ifdef VERBOSE
    printf("sieveing: %" PRIu64 " n: %" PRIu64 "\n", limit - i, n);
#endif
    
    for (p = POW(i); p < sieve_size; p += i)
      set_bit(ary, p);

  }

  /** 
   * now run the sieve in seps of size 6 
   * (each prime > 3 will be on one of the folloing palces: 6n +/- 1)
   */
  for (i = 5; i < limit; i += 4) {

#ifdef VERBOSE
    if (index % n == 0) {
      printf("sieveing: %" PRIu64 " n: %" PRIu64 "\n", limit - i, n);
      
      n = (uint64_t) (1.0 / ((((double) (limit - i)) / ((double ) limit)) * 
                             (((double) (limit - i)) / ((double ) limit)) *
                             (((double) (limit - i)) / ((double ) limit))));
      
    }
    index++;
#endif

    /* test 6n - 1 */
    if (is_prime(ary, i)) 
      for (p = POW(i); p < sieve_size; p += i) 
        set_bit(ary, p);

    i += 2; /* test 6n + 1 */
    if (is_prime(ary, i)) 
      for (p = POW(i); p < sieve_size; p += i) 
        set_bit(ary, p);
  }

  return generate_prime_table(ary, sieve_size);
}

/**
 * returns true if the given number is a 1cc
 */
static inline char is_1cc(uint64_t *ary, 
                          uint64_t sieve_size, 
                          uint64_t p,
                          uint64_t len) {
  
  uint64_t i;
  for (i = 1; i < len; i++) {

    p = (p << 1) + 1;
    
    if (p > sieve_size || !is_prime(ary, p))
      return 0;
  }

  return 1;
}

/**
 * generates a prime table with all prime chains 
 */
static inline void sieve_1ccs(uint64_t *ary, uint64_t sieve_size, uint64_t len) {

  /** 
   * now run the sieve in seps of size 6 
   * (each prime > 3 will be on one of the folloing palces: 6n +/- 1)
   */
  uint64_t i;
  for (i = 5; i < sieve_size; i += 4) {

    /* test 6n - 1 */
    if (is_prime(ary, i) && !is_1cc(ary, sieve_size, i, len))
      set_bit(ary, i);

    i += 2; /* test 6n + 1 */
    if (is_prime(ary, i) && !is_1cc(ary, sieve_size, i, len))
      set_bit(ary, i);
  }
}

/**
 * returns true if the given number is a 2cc
 */
static inline char is_2cc(uint64_t *ary, 
                          uint64_t sieve_size, 
                          uint64_t p,
                          uint64_t len) {
  
  uint64_t i;
  for (i = 1; i < len; i++) {

    p = (p << 1) - 1;
    
    if (p > sieve_size || !is_prime(ary, p))
      return 0;
  }

  return 1;
}

/**
 * generates a prime table with all prime chains 
 */
static inline void sieve_2ccs(uint64_t *ary, uint64_t sieve_size, uint64_t len) {

  /** 
   * now run the sieve in seps of size 6 
   * (each prime > 3 will be on one of the folloing palces: 6n +/- 1)
   */
  uint64_t i;
  for (i = 5; i < sieve_size; i += 4) {

    /* test 6n - 1 */
    if (is_prime(ary, i) && !is_2cc(ary, sieve_size, i, len))
      set_bit(ary, i);

    i += 2; /* test 6n + 1 */
    if (is_prime(ary, i) && !is_2cc(ary, sieve_size, i, len))
      set_bit(ary, i);
  }
}

/**
 * creat an so called primorial 
 * (a composite number out of the first n primes)
 */
static inline uint64_t primorial(uint64_t n) {

  uint64_t primorial = 1;

  uint64_t i;
  for (i = 0; i < n; i++)
    primorial *= first1001prims[i];

  return primorial;
}


/**
 * test a prime to be coverd by a given sequence
 */
static inline char seq_check_prime(PrimeSequence *seq, uint64_t p) {
  
  char coverd  = 0;
  uint64_t mod = p % seq->primorial;

  // TODO use binary search here, sequence contains all primes till primorial without the first sieved ones
  if (find_next_prime(seq->sequence, seq->len, mod))
    coverd = 1;

//  uint64_t i;
//  for (i = 0; i < seq->len; i++) {
//    if (mod == seq->sequence[i]) {
//      coverd = 1;
//      break;
//    }
//  }
  
  return coverd;
}

/**
 * thest all primes in the given prime table to cover the given sequence
 */
char check_sequence(PrimeSequence *seq, PrimeTable *primes) {
  
  char coverd = 1;

  uint64_t i;
  /* check the primes */
  if (seq->primes.len > 0 && seq->primes.ptr[0] != 2) return 0;

  for (i = 1; i < seq->primes.len && i < primes->len; i++) {

    #ifndef QUIET
    if (i % 12345 == 0)
      printf("Prime Check %" PRIu64 "   \r", seq->primes.len - i);
    #endif
    
    uint64_t index = find_next_prime(primes->ptr, 
                                     primes->len, 
                                     seq->primes.ptr[i]);

    if (primes->ptr[index] != seq->primes.ptr[i]) {
      printf("||===========================||\n"
             "||    Primes Check Failed    ||\n"
             "||===========================||\n\n"
             "Seq-prime: %10" PRIu64 "\n"
             "Org-prime: %10" PRIu64 "\n"
             "Index:     %10" PRIu64 "\n"
             "Seq-len:   %10" PRIu64 "\n"
             "Sequence Primes :",
             seq->primes.ptr[i],
             primes->ptr[index],
             i,
             seq->primes.len);

      uint64_t j;
      for (j = 0; j < seq->primes.len; j++)
        printf(" %" PRIu64 "", seq->primes.ptr[j]);

      printf("\n");
      return 0;
    }
  }
  
  for (i = 0; i < primes->len; i++)
    if (primes->ptr[i] > seq->primorial)
      break;

  for (; i < primes->len; i++) {

    #ifndef QUIET
    if (i % 12345 == 0)
      printf("Sequence Check %" PRIu64 "   \r", primes->len - i);
    #endif

    if (seq_check_prime(seq, primes->ptr[i]) == 0) {

      printf("||===========================||\n"
             "||   Sequence Check Failed   ||\n"
             "||===========================||\n\n"
             "Primorial: %10" PRIu64 "\n"
             "Prime    : %10" PRIu64 "\n"
             "Sequence :",
             seq->primorial,
             primes->ptr[i]);

      uint64_t j;
      for (j = 0; j < seq->len; j++)
        printf(" %" PRIu64 "", seq->sequence[j]);

      printf("\n");
      coverd = 0;
      break;
    }
  }

  return coverd;
}

/**
 * simple binary search to find an index, in an uint64_t array
 */
static inline uint64_t find_index(uint64_t *ary, uint64_t len, uint64_t val) {

  len >>= 1;
  uint64_t i = len;
  while (ary[i] != val) {
    
    if (len > 1)
      len >>= 1;
    
    if (ary[i] < val) i += len;
    else              i -= len;
  }

  return i;
}

/**
 * simple binary search to find an index with the next bigger prime
 */
static inline uint64_t find_next_prime(uint64_t *ary, uint64_t len, uint64_t val) {

  uint64_t orig_len = len;
  len >>= 1;
  uint64_t i = len;
  while (!(ary[i] < val && ary[i + 1] >= val)) {
    
    if (len > 1)
      len >>= 1;
    
    if (ary[i] < val) i += len;
    else              i -= len;

    if (i + 1 >= orig_len) {
      i = orig_len - 1;
      break;
    }
  }

  return i + 1;
}

/**
 * the first prime sequece (all natural numbers)
 */
static inline PrimeSequence *get_first_sequence() {
  
  PrimeSequence *seq = malloc(sizeof(PrimeSequence));
  seq->sequence      = malloc(sizeof(uint64_t));
  seq->primes.ptr    = NULL;

  seq->primorial     = 1;
  seq->n_primorial   = 0;
  seq->sequence[0]   = 0;
  seq->len           = 1;
  seq->primes.len    = 0;

  return seq;
}

/**
 * calculates the n-th primeSequence 
 */
PrimeSequence *generate_prime_sequence(uint64_t n) {

#ifdef VERBOSE
  uint64_t v = 1;
  uint64_t index = 0;
#endif
  
  if (n == 0) return get_first_sequence();

  PrimeSequence *seq = malloc(sizeof(PrimeSequence));
  seq->primorial     = primorial(n);
  seq->n_primorial   = n;

  /* bit array for sieveing */
  uint64_t sieve_size = seq->primorial;
  uint8_t *ary = (uint8_t *) calloc((sieve_size + 8) / 8, sizeof(uint8_t));

  if (ary == NULL) 
    perror("failed to allocate space for prime table generation");
  
  uint64_t i, p, limit = (uint64_t) (sqrt((double) sieve_size) + 1);

  /* 0 and 1 are not primes */
  set_bit(ary, 0);
  set_bit(ary, 1);

  /* sieveall primes included in the primorial */
  for (i = 0; i < n; i++) {
    
#ifdef VERBOSE
    printf("sieveing sequence: %" PRIu64 "\n", n - i);
#endif

    uint64_t prime = first1001prims[i];
    
    for (p = POW(prime); p < sieve_size; p += prime)
      set_bit(ary, p);
  }

  /* count the sequence numbers */
  seq->len         = count_sequence(ary, sieve_size, n);
  seq->sequence    = malloc(sizeof(uint64_t) * seq->len);
  seq->sequence[0] = 1;

  /* collect the numbers for the sequence */
  generate_seq_table(ary, sieve_size, seq->sequence + 1, n);


  /** 
   * Calculate the primes till primorial 
   * (skipp the first already sieved primes)
   * run the sieve in seps of size 6
   * (each prime > 3 will be on one of the folloing palces: 6n +/- 1)
   */
  i = 5;
  for (; i < first1001prims[n]; i += 4) {

#ifdef VERBOSE
    if (index % v == 0) {
      printf("sieveing: %" PRIu64 " n: %" PRIu64 "\n", limit - i, v);
      
      v = (uint64_t) (1.0 / ((((double) (limit - i)) / ((double ) limit)) * 
                             (((double) (limit - i)) / ((double ) limit)) *
                             (((double) (limit - i)) / ((double ) limit))));
      
    }
    index++;
#endif

    /* test 6n - 1 */
    if (is_prime(ary, i) && i >= first1001prims[n]) 
      for (p = POW(i); p < sieve_size; p += i) 
        set_bit(ary, p);

    i += 2; /* test 6n + 1 */
    if (is_prime(ary, i) && i >= first1001prims[n]) 
      for (p = POW(i); p < sieve_size; p += i) 
        set_bit(ary, p);
  }

  /** 
   * Calculate the primes till primorial
   * run the sieve in seps of size 6
   * (each prime > 3 will be on one of the folloing palces: 6n +/- 1)
   */
  for (; i < limit; i += 4) {

#ifdef VERBOSE
    if (index % v == 0) {
      printf("sieveing: %" PRIu64 " n: %" PRIu64 "\n", limit - i, v);
      
      v = (uint64_t) (1.0 / ((((double) (limit - i)) / ((double ) limit)) * 
                             (((double) (limit - i)) / ((double ) limit)) *
                             (((double) (limit - i)) / ((double ) limit))));
      
    }
    index++;
#endif

    /* test 6n - 1 */
    if (is_prime(ary, i)) 
      for (p = POW(i); p < sieve_size; p += i) 
        set_bit(ary, p);

    i += 2; /* test 6n + 1 */
    if (is_prime(ary, i)) 
      for (p = POW(i); p < sieve_size; p += i) 
        set_bit(ary, p);
  }

  seq->primes.len    = count_sequence(ary, sieve_size, 1);
  seq->primes.ptr    = malloc(sizeof(uint64_t) * seq->primes.len);
  seq->primes.ptr[0] = 2; 
  seq->primes.ptr[1] = 3;

  /** collect all primes till primorial */ // TODO hier muss ich einstellen, das beliebig viel primzahlen berechnet werden und dann müssen die primzahlen umgerechnet werden auf den start der sequence, und dann mittels resteclassen, berechnung, so angepasst werden, das ich sieben kann ohne getilt und modulo zu verwenden
  generate_seq_table(ary, sieve_size, seq->primes.ptr + 2, 0);
  free(ary);

  return seq;
}

/**
 * checks if the gien number is a prime
 */
static inline char is_num_prime(uint64_t num, PrimeTable *primes) {

  if (num == 2) return 1;

  // use an HASH here
  return primes->ptr[find_next_prime(primes->ptr, primes->len, num)] == num;
}

/**
 * factorisation of a given number uint64_t
 */
FactorNum *factorisize_int(uint64_t num, PrimeTable *primes) {
  
  FactorNum *fac  = malloc(sizeof(FactorNum));
  fac->factors    = malloc(sizeof(uint64_t) * 64);
  fac->len        = 0;
  fac->factors[0] = num;

  while (!is_num_prime(fac->factors[fac->len], primes)) {

    uint64_t i;
    for (i = 0; i < primes->len; i++) {
      
      if (fac->factors[fac->len] % primes->ptr[i] == 0) {
        
        fac->len++;
        fac->factors[fac->len]     = fac->factors[fac->len - 1] / primes->ptr[i];
        fac->factors[fac->len - 1] = primes->ptr[i];
        break;
      }
    }
  }

  fac->len++;
  return fac;
}

/*
static inline char uint64_bigger(uint64_t a, uint64_t b) { return a >  b; }
static inline char uint64_smaler(uint64_t a, uint64_t b) { return a <  b; }
static inline char uint64_equals(uint64_t a, uint64_t b) { return a == b; }

 * calculates the next sequence from a given sequence, by sieveing,
 * all composite numbers in the given sequence till the double of the next
 * sequence, and then calculate the next sequence and its composites
 *
PrimeSequence *next_sequence(PrimeSequence *seq) {

  if ((UINT64_MAX >> 1) / seq->primes.ptr[seq->n_primorial] <
      seq->primorial) {
    printf("[EE] max primsequence reatched %" PRIu64 "\n", seq->n_primorial);
  }
  
  PrimeSequence *next = malloc(sizeof(PrimeSequence));
  next->n_primorial   = seq->n_primorial + 1;
  next->primorial     = primorial(&seq->primes, next->n_primorial);

  uint64_t end   = 2 * next->primorial;
  uint64_t start = 2 * seq->primorial;

  / max number of primes till the double of the next sequence /
  uint64_t n_primes = ((end - start) / seq->primorial) * seq->len;
                      

  / the prime sequence till 2 * next->primorial /
  uint64_t *sequence = malloc(sizeof(uint64_t) * n_primes);

  / fill the sequence /
  uint64_t i, div, mod;
  for (i = 0, div = 1, mod = 0; i < n_primes; i++, mod++) {
    
    mod %= seq->len;
    if (mod == 0) div++;

    sequence[i] = div * seq->primorial + seq->sequence[mod];
  }

  / now remove the composite numbers from the sequence /
  while (ARY_LAST(seq->composites) < end) {
    
    uint64_t cur;
    ARY_PULL(seq->composites, cur);

    / calculate the possition of the composite /
    uint64_t pos = ((cur / seq->primorial) - 2) * seq->len +
                   find_index(seq->sequence, seq->len, (cur % seq->primorial));

    / unset the possition /
    sequence[pos] = 0;
  }

  / count the primes /
  uint64_t real_primes = 0;
  for (i = 0; i < n_primes; i++)
    if (sequence[i] != 0)
      real_primes++;

  next->primes.len = real_primes + seq->primes.len;
  next->primes.ptr = malloc(sizeof(uint64_t) * (next->primes.len));
  memcpy(next->primes.ptr, seq->primes.ptr, sizeof(uint64_t) * seq->primes.len);

  / save the primes /
  uint64_t j;
  for (i = 0, j = 0; i < n_primes; i++) {
    if (sequence[i] != 0) {
      next->primes.ptr[seq->primes.len + j] = sequence[i];
      j++;
    }
  }

  uint64_t start_index = find_next_prime(next->primes.ptr, 
                                         next->primes.len, 
                                         next->primorial);

  / calculate the length of the next sequence /
  next->len      = next->primes.len - start_index;
  next->sequence = malloc(sizeof(uint64_t) * next->len);

  / calculate te next sequence /
  for (j = 0, i = start_index; i < next->primes.len; j++, i++)
    next->sequence[j] = next->primes.ptr[i] - next->primorial;


  ARY_INIT(uint64_t, next->composites, ARY_LEN(seq->composites) * 2);
  ARY_CLONE(next->composites, seq->composites);

  / calculate the new composites /
  for (i = next->n_primorial; i < next->primes.len; i++) {

    j = max(seq->primes.len, i);

    if (UINT64_MAX / next->primes.ptr[i] < next->primes.ptr[j]) break;

    for (; j < next->primes.len; j++) {
      if (UINT64_MAX / next->primes.ptr[i] < next->primes.ptr[j]) break;

      ARY_PUSH(next->composites, next->primes.ptr[i] * next->primes.ptr[j]);
    }
  }

  / sort the composites so that we can always pop the last /
  QUICK_INSERT_SORT_MAX(uint64_t, 
                        next->composites.ptr, 
                        ARY_LEN(next->composites), 
                        uint64_bigger,
                        uint64_smaler, 
                        uint64_equals, 
                        20);

  return next;
}
*/
