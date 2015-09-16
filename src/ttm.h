#ifndef SPLATT_TTM_H
#define SPLATT_TTM_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "ftensor.h"
#include "thd_info.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void ttm_splatt(
    splatt_csf_t const * const ft,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    idx_t const nthreads);

#endif
