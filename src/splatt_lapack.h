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

/* Matrix multiplications */
void SPLATT_BLAS(gemv)(
    char *,
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_val_t const *,
    splatt_blas_int *,
    splatt_val_t const *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_val_t *,
    splatt_blas_int *);

void SPLATT_BLAS(gemm)(
    char *,
    char *,
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_val_t const *,
    splatt_blas_int *,
    splatt_val_t const *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_val_t *,
    splatt_blas_int *);




/* Cholesky factorization */
void SPLATT_BLAS(potrf)(
    char *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_blas_int *);

void SPLATT_BLAS(potrs)(
    char *,
    splatt_blas_int *,
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


void SPLATT_BLAS(gesdd)(
    char *,
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_val_t *,
    splatt_blas_int *,
    splatt_blas_int *,
    splatt_blas_int *);

void SPLATT_BLAS(bdsqr)(
    char * uplo,
    splatt_blas_int * N,
    splatt_blas_int * NCVT,
    splatt_blas_int * NRU,
    splatt_blas_int * NCC,
    splatt_val_t * D,
    splatt_val_t * E,
    splatt_val_t * VT,
    splatt_blas_int * LDVT,
    splatt_val_t * U,
    splatt_blas_int * LDU,
    splatt_val_t * C,
    splatt_blas_int * LDC,
    splatt_val_t * work,
    splatt_blas_int * info);

#endif
