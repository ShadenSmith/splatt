#ifndef SPLATT_BASE_H
#define SPLATT_BASE_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../include/splatt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>


/******************************************************************************
 * DEFINES
 *****************************************************************************/
/* alias splatt types */
#define val_t splatt_val_t
#define idx_t splatt_idx_t

#define SS_MIN(x,y) ((x) < (y) ? (x) : (y))
#define SS_MAX(x,y) ((x) > (y) ? (x) : (y))


/******************************************************************************
 * DEFAULTS
 *****************************************************************************/
static double const DEFAULT_TOL = 1e-4;

static idx_t const DEFAULT_NFACTORS = 10;
static idx_t const DEFAULT_ITS = 50;
static idx_t const DEFAULT_THREADS = 1;
static idx_t const DEFAULT_MPI_DISTRIBUTION = 0;

static int const DEFAULT_WRITE = 1;
static int const DEFAULT_TILE = 0;

#endif
