#ifndef SPLATT_SPLATT_H
#define SPLATT_SPLATT_H

#include <stddef.h>

typedef double splatt_val_t;
#define SS_VAL "f"
#define SS_MPI_VAL MPI_DOUBLE

typedef unsigned long splatt_idx_t;
#define SS_IDX "lu"
#define SS_MPI_IDX MPI_UNSIGNED_LONG


/**
* @brief Compute the CPD using alternating least squares.
*
* @param rank The rank of the decomposition to perform.
* @param nmodes The number of modes in the tensor. Optimizations are currently
*               only present for nmodes=3.
* @param nnz The number of nonzeros in the tensor.
* @param inds The nonzero indices of the tensor. The nth nonzero can be found
*             at inds[0][n], inds[1][n], ... , inds[m][n]. These indices
*             WILL be rearranged during computation (for sorting, etc.).
* @param vals The nonzero values of the tensor. These values WILL be rearranged
*             during computation (for sorting. etc.).
* @param mats The factor matrices, pre-initialized. Layout is assumed to be
*             row-major.
* @param lambda The scaling factors extracted from mats.
*/
void cpd_als(
    splatt_idx_t const rank,
    splatt_idx_t const nmodes,
    splatt_idx_t const nnz,
    splatt_idx_t ** const inds,
    splatt_val_t * const vals,
    splatt_val_t ** const mats,
    splatt_val_t * const lambda);

#endif
