

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../admm.h"


/******************************************************************************
 * TYPES
 *****************************************************************************/


typedef struct
{
  val_t lambda;
  matrix_t * transpose_buffer;
  matrix_t * banded;
  int      * pivot;
} smooth_ws;


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Initialize the banded matrix B^T * B, where B is a tri-diagonal matrix
*        with 2's on the diagonal and 1's on the sub/super-diagonals. The
*        result is a banded matrix with bandwidth 2, stored col-major:
*
*                                 0   0   0   0   0
*                                 0   0   0   0   0
*                                 5  -4   1   0   0
*               rho * (lambda *  -4   6  -4   1   0  + diag(rho))
*                                 1  -4   6  -4   1
*                                 0   1  -4   6  -4
*                                 0   0   1  -4   5
*
*        The additional 2 rows at the top are for fill-in during LU
*        factorization.
*
*        This routine then computes the LU factorization of this matrix using
*        DGBTRF.
*
* @param[out] smooth The smoothness workspace to initialize.
* @param I The dimension of the mode with smoothness.
* @param rho The current penalty term of the ADMM iteration.
*/
static void p_form_banded(
    smooth_ws * const smooth,
    idx_t const I,
    val_t const rho)
{
  val_t * vals = smooth->banded->vals;

  val_t const lambda = smooth->lambda;

  /* first column is special */
  vals[2+2+0] =  (5. * lambda) + rho;
  vals[2+2+1] = -4. * lambda;
  vals[2+2+2] =  1. * lambda;

  /* all columns except the last */
  idx_t const nrows = smooth->banded->I; /* account for extra rows */
  for(idx_t i=1; i < I-1; ++i) {
    vals += nrows;

    /* offset into current column */
    if(i > 1) {
      vals[2+0] =  1. * lambda;
    }

    vals[2+1] = -4. * lambda;
    vals[2+2] = (6. * lambda) + rho; /* add rho to diagonal */
    vals[2+3] = -4. * lambda;

    if(i < I-2) {
      vals[2+4] =  1. * lambda;
    }
  }

  /* last column is special, too */
  vals += nrows;
  vals[2+0] =  1. * lambda;
  vals[2+1] = -4. * lambda;
  vals[2+2] = (5. * lambda) + rho;

  /* compute the LU factorization */
  int nbands = 2;
  int M = (int) I;
  int N = (int) I;
  int KL = (int) nbands;
  int KU = (int) nbands;
  int lda = (int) (2 * nbands) + nbands + 1;
  int info = 0;
  LAPACK_DGBTRF(&M, &N, &KL, &KU, smooth->banded->vals, &lda, smooth->pivot, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGBTRF returned %d\n", info);
  }
}


void splatt_smooth_init(
    splatt_val_t * vals,
    splatt_idx_t const nrows,
    splatt_idx_t const ncols,
    void * data)
{
  smooth_ws * ws = data;

  /* This will be a matrix stored in LAPACK banded format. We allocate
   * diagonal + upper/lower bands + another 2 bands for LU fill-in. */
  int const nbands = 2;
  ws->banded = mat_alloc(1 + (nbands * 3), nrows);
  ws->banded->rowmajor = 0;
  ws->pivot = splatt_malloc(nrows* sizeof(*ws->pivot));

  ws->transpose_buffer = mat_alloc(nrows, ncols);
}




/**
* @brief Apply the proximity operator to enforce column smoothness. This solves
*        a banded linear system and performs a few matrix transposes.
*
* @param[out] primal The row-major matrix to update.
* @param nrows The number of rows in primal.
* @param ncols The number of columns in primal.
* @param offset Not used.
* @param data Workspace.
* @param rho Multiplier on the regularization.
* @param should_parallelize If true, parallelize.
*/
void splatt_smooth_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize)
{
  assert(offset == 0);

  smooth_ws * ws = data;

  val_t * const restrict buf = ws->transpose_buffer->vals;

  /* form the banded matrix and compute its LU factorization */
  p_form_banded(ws, nrows, rho);

  /* transpose the RHS (primal) */
  #pragma omp parallel if(should_parallelize)
  {
    for(idx_t j=0; j < ncols; ++j) {
      #pragma omp for schedule(static) nowait
      for(idx_t i=0; i < nrows; ++i) {
        idx_t const old = j + (i*ncols);
        idx_t const new = i + (j*nrows);
        buf[new] = primal[old];
      }
    }
  } /* end omp parallel */

  /* solve the linear system of equations */
  char trans = 'N';
  int N = (int) nrows;
  int KL = 2;
  int KU = 2;
  int nrhs = (int) ncols;
  int lda = (int) (2 * KL) + KU + 1;
  int ldb = N;
  int info;

  LAPACK_DGBTRS(&trans, &N, &KL, &KU, &nrhs, ws->banded->vals, &lda,
      ws->pivot, buf, &ldb, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGBTRS returned %d\n", info);
  }

  /* now transpose back and multiply by rho */
  #pragma omp parallel for schedule(static) if(should_parallelize)
  for(idx_t i=0; i < nrows; ++i) {
    for(idx_t j=0; j < ncols; ++j) {
      idx_t const old = i + (j*nrows);
      idx_t const new = j + (i*ncols);
      primal[new] = buf[old] * rho;
    }
  }
}


/**
* @brief Free the smoothness workspace.
*
* @param data The data to free.
*/
void splatt_smooth_free(
    void * data)
{
  smooth_ws * ws = data;

  mat_free(ws->banded);
  mat_free(ws->transpose_buffer);
  splatt_free(ws->pivot);
  splatt_free(ws);
}




/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

splatt_error_type splatt_register_smooth(
    splatt_cpd_opts * opts,
    splatt_val_t const multiplier,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes)
{
  for(idx_t m=0; m < num_modes; ++m) {
    idx_t const mode = modes_included[m];
    splatt_cpd_constraint * con = splatt_alloc_constraint(SPLATT_CON_ADMM);

    con->prox_func = splatt_smooth_prox;
    con->init_func = splatt_smooth_init;
    con->free_func = splatt_smooth_free;

    sprintf(con->description, "SMOOTH-COL");

    smooth_ws * ws = splatt_malloc(sizeof(*ws));
    ws->lambda = multiplier;

    con->data = ws;

    splatt_register_constraint(opts, mode, con);
  }

  return SPLATT_SUCCESS;
}


