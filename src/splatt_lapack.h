#ifndef SPLATT_LAPACK_H
#define SPLATT_LAPACK_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "base.h"




/******************************************************************************
 * TYPES / MACROS
 *****************************************************************************/

/* Define prefix/postfix for BLAS/LAPACK functions. */
#if SPLATT_VAL_TYPEWIDTH == 32
  #define SPLATT_BLAS(func) s ## func ## _
#else
  #define SPLATT_BLAS(func) d ## func ## _
#endif



/******************************************************************************
 * PROTOTYPES
 *****************************************************************************/

/* Cholesky factorization */
void SPLATT_BLAS(potrf)(
    char *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_blas_int *);
void SPLATT_BLAS(potrs)(
    char *, splatt_blas_int *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_blas_int *);

/* Rank-k update. */
void SPLATT_BLAS(syrk)(
    char *,
    char *,
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_val_t *,
    splatt_blas_int *);

/* LU */
void SPLATT_BLAS(getrf)(
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_blas_int *);
void SPLATT_BLAS(getrs)(
    char *,
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_blas_int *);


/* SVD solve */
void SPLATT_BLAS(gelss)(
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_blas_int *);

#endif
