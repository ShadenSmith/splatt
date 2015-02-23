#ifndef SPLATT_BASE_H
#define SPLATT_BASE_H

#include "../include/splatt.h"

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* alias splatt types */
#define val_t splatt_val_t
#define idx_t splatt_idx_t

#define MAX_NMODES ((idx_t)8)

#define SS_MIN(x,y) ((x) < (y) ? (x) : (y))
#define SS_MAX(x,y) ((x) > (y) ? (x) : (y))

#endif
