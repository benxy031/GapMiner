/**
 * smal definitin of an large array > 1 MB
 * using an underlying field
 *
 * usefull wehen meory is fragmented and it fails 
 * to malloc large memory blocks
 */
#ifndef __LONG_ARRAY_H__
#define __LONG_ARRAY_H__

#include <time.h>
#include <inttypes.h>
#include <math.h>

/**
 * defines an new LongArray struct with a given type as content
 * and a given Name
 */
#define DEFINE_LONG_ARRAY(TYPE, NAME)   \
typedef struct {                        \
  TYPE **ptr;                           \
  uint64_t length;                      \
  uint64_t col_len;                     \
  uint64_t max_rows;                    \
  uint64_t cur_rows;                    \
  uint64_t max_len;                     \
} NAME


/**
 * Initialize an given Array with an given max length
 * also initialize rand
 *
 * Note it can grow lager. max length is for the block size
 *      block_length = sqrt(max)
 *      you should give the expected maximum length
 */
#define LONG_ARY_INIT(TYPE, ARY, MAX_LEN)                               \
  do {                                                                  \
    (ARY).col_len = (ARY).max_rows = (uint64_t) sqrt(MAX_LEN) + 1;      \
      (ARY).ptr = (TYPE **) calloc((ARY).col_len, sizeof(TYPE *));        \
    (ARY).ptr[0] = (TYPE *) malloc(sizeof(TYPE) * (ARY).col_len);       \
    (ARY).length = 0;                                                   \
    (ARY).max_len = MAX_LEN;                                            \
    (ARY).cur_rows = 1;                                                 \
    srand(time(NULL) * rand());                                         \
  } while (0)

/**
 * Makro for accressing an Long Arry index
 */
#define LONG_ARY_AT(ARY, INDEX)                                         \
    ((ARY).ptr[(INDEX) / (ARY).col_len][(INDEX) % (ARY).col_len])
  
/**
 * Makro for the address of an Long Array Element at index
 */
#define LONG_ARY_PTR_AT(ARY, INDEX)                                     \
  ((ARY).ptr[(INDEX) / (ARY).col_len] + ((INDEX) % (ARY).col_len))

/**
 * Makro for Accressing the Last Element of the given Array
 */
#define LONG_ARY_LAST(ARY) LONG_ARY_AT(ARY, ((ARY).length - 1))

/**
 * Insert an given element at the end of a given Array
 * (mallocs space if neccesary)
 *
 * NOTE the Arry content type should be the same as the type of 
 *      the given element
 */
#define LONG_ARY_ADD(ARY, E)                                                  \
  do {                                                                        \
    if ((ARY).length >= (ARY).max_len) {                                      \
      (ARY).max_len *= 2;                                                     \
      (ARY).ptr = realloc((ARY).ptr, sizeof((ARY).ptr[0]) * (ARY).max_len);   \
    }                                                                         \
    if ((ARY).length / (ARY).col_len == (ARY).cur_rows) {                     \
      (ARY).ptr[(ARY).cur_rows] = malloc(sizeof((ARY).ptr[0][0])              \
                                         * (ARY).col_len);                    \
      (ARY).cur_rows++;                                                       \
    }                                                                         \
                                                                              \
    LONG_ARY_AT(ARY, (ARY).length) = E;                                       \
    (ARY).length++;                                                           \
  } while (0)

/**
 * Reserves Space for one more Element if nedded
 *
 * NOTE the Arry content type should be the same as the type of 
 * the given element and TYPE
 */
#define LONG_ARY_ADD_SPACE(ARY)                                               \
  do {                                                                        \
    if ((ARY).length >= (ARY).max_len) {                                      \
      (ARY).max_len *= 2;                                                     \
      (ARY).ptr = realloc((ARY).ptr, sizeof((ARY).ptr[0]) * (ARY).max_len);   \
    }                                                                         \
    if ((ARY).length / (ARY).col_len == (ARY).cur_rows) {                     \
      (ARY).ptr[(ARY).cur_rows] = malloc(sizeof((ARY).ptr[0][0])              \
                                         * (ARY).col_len);                    \
      (ARY).cur_rows++;                                                       \
    }                                                                         \
  } while (0)

/**
 * Removes and Returns the Last Element of a given Array if possible
 *
 * NOTE the Arry content type should be the same as the type of 
 *      the given element
 * NOTE it returns NULL if there is no more element in the Array
 */
#define LONG_ARY_PULL(ARY, VALUE)                                         \
  do {                                                                    \
    if ((ARY).length > 0) {                                               \
      (ARY).length--;                                                     \
      VALUE = LONG_ARY_AT(ARY, (ARY).length);                             \
    } else {                                                              \
      printf("[ERROR] no more elements in Array, in %s at Line %d\n",     \
             __FILE__, __LINE__);                                         \
      exit(1);                                                            \
    }                                                                     \
  } while (0)                                                            

/**
 * Extract a random element from a given array
 * and saves it into a given value if possible
 *
 * note it changes the order of the given array
 * NOTE the Arry content type should be the same as th type of 
 *      the given value
 *      It returns Null if the is no more element in the given Array
 */
#define LONG_ARY_EXTRACT(ARY, VALUE)                                      \
  do {                                                                    \
    if ((ARY).length > 0) {                                               \
      uint64_t ary_ext_i = RAND_INDEX(ARY);                               \
      VALUE = LONG_ARY_AT(ARY, ary_ext_i);                                \
      (ARY).length--;                                                     \
      LONG_ARY_AT(ARY, ary_ext_i) = LONG_ARY_AT(ARY, (ARY).length);       \
    } else {                                                              \
      printf("[ERROR] no more elements in Array, in %s at Line %d\n",     \
             __FILE__, __LINE__);                                         \
      exit(1);                                                            \
    }                                                                     \
  } while (0)                                                            

/**
 * Frees an given Long Array
 */
#define LONG_ARY_FREE(ARY)                                                  \
  do {                                                                      \
    uint64_t laryfree_i;                                                    \
    if ((ARY).ptr != NULL) {                                                \
      for (laryfree_i = 0; laryfree_i < (ARY).cur_rows; laryfree_i++) {     \
        if ((ARY).ptr[laryfree_i] != NULL) {                                \
          free((ARY).ptr[laryfree_i]);                                      \
          (ARY).ptr[laryfree_i] = NULL;                                     \
        }                                                                   \
      }                                                                     \
    }                                                                       \
    if ((ARY).ptr != NULL) {                                                \
      free((ARY).ptr);                                                      \
      (ARY).ptr = NULL;                                                     \
    }                                                                       \
  } while (0)


/**
 * Clones an given LongArray into DST.
 *
 * also initializes it
 */
#define LONG_ARY_CLONE(DST, SRC)                                              \
  do {                                                                        \
    if ((DST).ptr == NULL                ||                                   \
        (DST).col_len  != (SRC).col_len)                                      \
      LARY_FREE(DST);                                                         \
                                                                              \
    (DST).length   = (SRC).length;                                            \
    (DST).col_len  = (SRC).col_len;                                           \
    (DST).max_rows = (SRC).max_rows;                                          \
    (DST).cur_rows = (SRC).cur_rows;                                          \
    (DST).max_len  = (SRC).max_len;                                           \
      (DST).ptr      = calloc((DST).col_len, sizeof((SRC).ptr[0]));             \
                                                                              \
    uint64_t lary_clone_i;                                                    \
    for (lary_clone_i = 0;                                                    \
         lary_clone_i < (DST).cur_rows;                                       \
         lary_clone_i++) {                                                    \
                                                                              \
      if ((DST).ptr[lary_clone_i] == NULL)                                    \
        (DST).ptr[lary_clone_i] = malloc(sizeof((SRC).ptr[0][0])              \
                                                * (DST).col_len);             \
      memcpy((DST).ptr[lary_clone_i],                                         \
             (SRC).ptr[lary_clone_i],                                         \
             sizeof((SRC).ptr[0][0]) * (DST).col_len);                        \
    }                                                                         \
  } while (0)

/**
 * Dublicates and returns an given LongArray
 */
#define LONG_ARY_DUB(SRC)                                                     \
  ({                                                                          \
    typeof(SRC) _tmp_lary_;                                                   \
    LONG_ARY_INIT(typeof((SRC).ptr[0], _tmp_lary_, (SRC).max_len);            \
                                                                              \
    LARY_CLONE(_tmp_lary_, SRC);                                              \
    _tmp_lary_;                                                               \
  })

/**
 * Returns the Length of an given Long Array
 */
#define LONG_ARY_LENGTH(ARY) (ARY).length

/**
 * access the un initialized element after the last added element
 */
#define LONG_ARY_AT_LENGTH(ARY) LONG_ARY_AT(ARY, ((ARY).length))

/**
 * clears an LARY (sets it size to zero)
 */
#define LONG_ARY_CLEAR(ARY) (ARY).length = 0

/**
 * Some shorter Markros
 */
#ifndef LARY
#define DEF_LARY(TYPE, NAME)            DEFINE_LONG_ARRAY(TYPE, NAME)
#define LARY_INIT(TYPE, ARY, MAX_LEN)   LONG_ARY_INIT(TYPE, ARY, MAX_LEN)
#define LARY_AT(ARY, INDEX)             LONG_ARY_AT(ARY, INDEX)
#define LARY_PTR(ARY, INDEX)            LONG_ARY_PTR_AT(ARY, INDEX)
#define LARY_RAND_PULL(ARY, VALUE)      LONG_ARY_EXTRACT(ARY, VALUE)
#define LARY_ADD(ARY, E)                LONG_ARY_ADD(ARY, E)
#define LARY_PUSH(ARY, E)               LONG_ARY_ADD(ARY, E)
#define LARY_PULL(ARY, VALUE)           LONG_ARY_PULL(ARY, VALUE)
#define LARY_FREE(ARY)                  LONG_ARY_FREE(ARY)
#define LARY_CLONE(DST, SRC)            LONG_ARY_CLONE(DST, SRC)
#define LARY_DUB(DST, SRC)              LONG_ARY_DUB(DST, SRC)
#define LARY_LAST(ARY)                  LONG_ARY_LAST(ARY)
#define LARY_AT_LEN(ARY)                LONG_ARY_AT_LENGTH(ARY)
#define LARY_ADD_SPACE(ARY)             LONG_ARY_ADD_SPACE(ARY)
#define LARY_LEN(ARY)                   LONG_ARY_LENGTH(ARY)                        
#define LARY_CLEAR(ARY)                 LONG_ARY_CLEAR(ARY)
#endif

#endif // __LONG_ARRAY_H__
