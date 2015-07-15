#ifndef SPLATT_UTIL_H
#define SPLATT_UTIL_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define rand_val splatt_rand_val
/**
* @brief Generate a random val_t in the range [0, 1].
*
* @return A pseudo-random val_t.
*/
val_t rand_val(void);


#define rand_idx splatt_rand_idx
/**
* @brief Generate a random idx_t in the range [0, RAND_MAX << 16].
*
* @return A pseudo-random idx_t.
*/
idx_t rand_idx(void);


#define fill_rand splatt_fill_rand
/**
* @brief Fill a val_t array with random values.
*
* @param vals The array of values to fill
* @param nelems The length of the array.
*/
void fill_rand(
  val_t * const restrict vals,
  idx_t const nelems);


#define bytes_str splatt_bytes_str
/**
* @brief Return a string describing a human-readable number of bytes.
*
* @param bytes The number of bytes to describe
*
* @return The human-readable string. NOTE: this string needs to be freed!
*/
char * bytes_str(
  idx_t const bytes);

#endif
