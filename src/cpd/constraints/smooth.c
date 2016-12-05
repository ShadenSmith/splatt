
#if 0

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../../base.h"
#include "../../matrix.h"


/******************************************************************************
 * TYPES
 *****************************************************************************/


typedef struct
{
  val_t lambda;
  matrix_t * banded;
  int      * pivot;
} reg_smooth_ws;




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
*           penalty * (lambda *  -4   6  -4   1   0  + diag(penalty))
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
* @param penalty The current penalty term of the ADMM iteration.
*/
static void p_form_banded(
    reg_smooth_ws * const smooth,
    idx_t const I,
    val_t const penalty)
{
  val_t * vals = smooth->banded->vals;

  val_t const lambda = smooth->lambda;

  /* first column is special */
  vals[2+2+0] =  (5. * lambda) + penalty;
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
    vals[2+2] = (6. * lambda) + penalty; /* rho gets added to diagonal */
    vals[2+3] = -4. * lambda;

    if(i < I-2) {
      vals[2+4] =  1. * lambda;
    }
  }

  /* last column is special, too */
  vals += nrows;
  vals[2+0] =  1. * lambda;
  vals[2+1] = -4. * lambda;
  vals[2+2] = (5. * lambda) + penalty;

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



static void p_init_smooth(
    reg_smooth_ws * const smooth,
    idx_t const I)
{
  /* This will be a matrix stored in LAPACK banded format. We allocate
   * diagonal + upper/lower bands + another 2 bands for LU fill-in. */
  int const nbands = 2;
  smooth->banded = mat_alloc(1 + (nbands * 3), I);
  smooth->banded->rowmajor = 0;
  smooth->pivot = splatt_malloc(I * sizeof(*smooth->pivot));
}




/**
* @brief Apply the proximity operator to enforce column smoothness. This solves
*        a banded linear system and performs a few matrix transposes.
*
* @param[out] mat_primal The primal variable (factor matrix) to update.
* @param penalty The current penalty parameter (rho).
* @param ws CPD workspace data -- used for regularization parameter.
* @param mode The mode we are updating.
* @param data Constraint-specific data -- a reg_smooth_ws struct pointer.
* @param is_chunked Whether this is a chunk of a larger matrix.
*/
static void p_proximity_smooth(
    matrix_t  * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t const * const mat_dual,
    val_t const penalty,
    void const * const data,
    bool const is_chunked)
{
  assert(!is_chunked);

  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict matv = mat_primal->vals;
  val_t       * const restrict auxl = mat_auxil->vals;
  val_t const * const restrict dual = mat_dual->vals;

  reg_smooth_ws * smooth = (reg_smooth_ws *) data;

  /* allocate if this is the first time */
  if(smooth->banded == NULL) {
    p_init_smooth(smooth, I);
  }

  /* form the banded matrix and compute its LU factorization */
  p_form_banded(smooth, I, penalty);

  /* form and transpose the RHS */
  #pragma omp parallel
  {
    for(idx_t j=0; j < J; ++j) {
      #pragma omp for schedule(static) nowait
      for(idx_t i=0; i < I; ++i) {
        idx_t const old = j + (i*J);
        idx_t const new = i + (j*I);
        matv[new] = auxl[old] - dual[old];
      }
    }
  } /* end omp parallel */

  /* solve the linear system of equations */
  char trans = 'N';
  int N = (int) I;
  int KL = 2;
  int KU = 2;
  int nrhs = (int) J;
  int lda = (int) (2 * KL) + KU + 1;
  int ldb = N;
  int info;

  LAPACK_DGBTRS(&trans, &N, &KL, &KU, &nrhs, smooth->banded->vals, &lda,
      smooth->pivot, matv, &ldb, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGBTRS returned %d\n", info);
  }

  /* now transpose back and multiply by penalty */
  #pragma omp parallel for schedule(static)
  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      idx_t const old = i + (j*I);
      idx_t const new = j + (i*J);
      auxl[new] = matv[old] * penalty;
    }
  }

  /* now copy back to mat_primal... */
  par_memcpy(matv, auxl, I * J * sizeof(*matv));
}




void splatt_cpd_reg_smooth(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode,
    splatt_val_t const scale)
{
  /* MAX_NMODES will simply apply regularization to all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_reg_smooth(cpd_opts, m, scale);
    }
    return;
  }

  splatt_cpd_con_clear(cpd_opts, mode);

  cpd_opts->unconstrained = false;
  cpd_opts->constraints[mode].which = SPLATT_REG_SMOOTHNESS;

  reg_smooth_ws * smooth = splatt_malloc(sizeof(*smooth));
  smooth->lambda = scale;
  smooth->banded = NULL; /* to be allocated when we know the dimension */
  cpd_opts->constraints[mode].data = (void *) smooth;
}


#endif
