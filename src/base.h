#ifndef SPLATT_BASE_H
#define SPLATT_BASE_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../include/splatt.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <time.h>


/******************************************************************************
 * DEFINES
 *****************************************************************************/
#define MAX_NMODES SPLATT_MAX_NMODES

/* alias splatt types */
#define val_t splatt_val_t
#define idx_t splatt_idx_t

#define SS_MIN(x,y) ((x) < (y) ? (x) : (y))
#define SS_MAX(x,y) ((x) > (y) ? (x) : (y))


/******************************************************************************
 * DEFAULTS
 *****************************************************************************/
static double const DEFAULT_TOL = 1e-5;

static idx_t const DEFAULT_NFACTORS = 16;
static idx_t const DEFAULT_ITS = 50;
static idx_t const DEFAULT_MPI_DISTRIBUTION = MAX_NMODES+1;

#define SPLATT_MPI_FINE (MAX_NMODES + 1)

static int const DEFAULT_WRITE = 0;
static int const DEFAULT_TILE = 0;

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * MEMORY ALLOCATION
 *****************************************************************************/

/**
* @brief Allocate 'bytes' memory, 64-bit aligned. Returns a pointer to memory.
*
* @param bytes The number of bytes to allocate.
*
* @return The allocated memory.
*/
void * splatt_malloc(
    size_t const bytes);


/**
* @brief Free memory allocated by splatt_malloc().
*
* @param ptr The pointer to free.
*/
void splatt_free(
    void * ptr);

#ifdef __cplusplus
}
#endif

#endif
