



/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "admm.h"
#include "../util.h"
#include "../io.h"



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




/******************************************************************************
 * PROXIMITY OPERATORS
 *
 *                Used to enforce ADMM constraints.
 *
 * This class of functions operate on (H\tilde^T - U), the difference between
 * the auxiliary and the dual variables. All proximity functions are expected
 * to take the same function signature.
 *****************************************************************************/


/**
* @brief Perform Lasso regularization via soft thresholding.
*
* @param[out] mat_primal The primal variable (factor matrix) to update.
* @param penalty The current penalty parameter (rho).
* @param ws CPD workspace data -- used for regularization parameter.
* @param mode The mode we are updating.
* @param data Constraint-specific data -- we store lambda in this.
* @param is_chunked Whether this is a chunk of a larger matrix.
*/
static void p_proximity_l1(
    matrix_t  * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t const * const mat_dual,
    val_t const penalty,
    void const * const data,
    bool const is_chunked)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxl = mat_auxil->vals;
  val_t const * const restrict dual = mat_dual->vals;

  val_t const lambda = *((val_t *) data);
  val_t const mult = lambda / penalty;

  #pragma omp parallel for schedule(static) if(!is_chunked)
  for(idx_t x=0; x < I * J; ++x) {
    val_t const v = auxl[x] - dual[x];

    /* TODO: could this be done faster? */
    if(v > mult) {
      matv[x] = v - mult;
    } else if(v < -mult) {
      matv[x] = v + mult;
    } else {
      matv[x] = 0.;
    }
  }
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




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Compute the auxiliary matrix before the Cholesky solve. This function
*        computes: mat_mttkrp + (penalty .* (mat_primal - mat_dual)).
*
* @param mat_primal The primal variable.
* @param mat_mttkrp The latest MTTKRP result.
* @param mat_dual The dual variable.
* @param penalty The penalty parameter, 'rho'. This could also be used during
*                l2 (Tikhonov) regularization.
* @param[out] mat_auxil The auxiliary matrix.
* @param is_chunked Whether this is a chunk of a larger matrix.
*/
static void p_setup_auxiliary(
    matrix_t const * const mat_primal,
    matrix_t const * const mat_mttkrp,
    matrix_t const * const mat_dual,
    val_t const penalty,
    matrix_t * const mat_auxil,
    bool const is_chunked)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict aux    = mat_auxil->vals;

  val_t const * const restrict mttkrp = mat_mttkrp->vals;
  val_t const * const restrict primal = mat_primal->vals;
  val_t const * const restrict dual   = mat_dual->vals;

  #pragma omp parallel for schedule(static) if(!is_chunked)
  for(idx_t x=0; x < I * J; ++x) {
    aux[x] = mttkrp[x] + penalty * (primal[x] + dual[x]);
  }
}



/**
* @brief Update the dual variable after updating the primal and auxiliary
*        variables. The squared Frobenius norm of the new dual is returned.
*        This function performs: mat_dual += mat_primal - mat_auxil.
*
* @param mat_primal The newest primal variable.
* @param mat_auxil The newest auxiliary variable.
* @param[out] mat_dual The dual variable to update.
* @param is_chunked Whether this is a chunk of a larger matrix.
*
* @return The norm of the new dual; || mat_dual ||_F^2.
*/
static val_t p_update_dual(
    matrix_t const * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t * const mat_dual,
    bool const is_chunked)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict dual = mat_dual->vals;
  val_t const * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxl = mat_auxil->vals;

  val_t norm = 0.;

  #pragma omp parallel for schedule(static) reduction(+:norm) if(!is_chunked)
  for(idx_t x=0; x < I * J; ++x) {
    dual[x] += matv[x] - auxl[x];
    norm += dual[x] * dual[x];
  }

  return norm;
}



/**
* @brief Calculate the primal and dual residuals before the ADMM convergence
*        check.
*
* @param mat_primal The primal variable (the factor we are updating).
* @param mat_auxil The auxiliary matrix; ideally mat_auxil^T = mat_primal.
* @param mat_init The initial matrix factor (at the start of this iteration).
* @param[out] primal_norm The norm of the primal variable; norm(mat_primal)^2.
* @param[out] primal_resid The residual of the primal variable;
*             norm(mat_primal - mat_auxil)^2.
* @param[out] dual_resid The dual residual; norm(mat_primal - mat_init)^2.
* @param is_chunked Whether this is a chunk of a larger matrix.
*/
static void p_calc_residual(
    matrix_t const * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t const * const mat_init,
    val_t * primal_norm,
    val_t * primal_resid,
    val_t * dual_resid,
    bool const is_chunked)
{
  val_t const * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxv = mat_auxil->vals;
  val_t const * const restrict init = mat_init->vals;

  idx_t const nrows = mat_primal->I;
  idx_t const ncols = mat_primal->J;

  val_t p_norm  = 0;
  val_t p_resid = 0;
  val_t d_resid = 0;

#define ROW_CONVERGE
#ifdef ROW_CONVERGE

  #pragma omp parallel for reduction(max:p_norm, p_resid, d_resid) \
      if(!is_chunked)
  for(idx_t i=0; i < nrows; ++i) {
    val_t row_p_norm  = 0;
    val_t row_p_resid = 0;
    val_t row_d_resid = 0;

    for(idx_t j=0; j < ncols; ++j) {
      idx_t const index = j + (i*ncols);
      val_t const pdiff = matv[index] - auxv[index];
      val_t const ddiff = matv[index] - init[index];
      row_p_norm  += matv[index] * matv[index];
      row_p_resid += pdiff * pdiff;
      row_d_resid += ddiff * ddiff;
    }

    /* save the row with the largest primal residual */
    if(row_p_resid > p_resid) {
      p_norm  = row_p_norm;
      p_resid = row_p_resid;
      d_resid = row_d_resid;
    }
  }

#else
  #pragma omp parallel for reduction(+:p_norm, p_resid, d_resid) \
      if(!is_chunked)
  for(idx_t i=0; i < nrows; ++i) {
    for(idx_t j=0; j < ncols; ++j) {
      idx_t const index = j + (i*ncols);
      val_t const pdiff = matv[index] - auxv[index];
      val_t const ddiff = matv[index] - init[index];

      p_norm  += matv[index] * matv[index];
      p_resid += pdiff * pdiff;
      d_resid += ddiff * ddiff;
    }
  }
#endif

  *primal_norm  = p_norm;
  *primal_resid = p_resid;
  *dual_resid   = d_resid;
}



/**
* @brief Optimally compute the new primal variable when no regularization or
*        only L2 regularization is present.
*
* @param[out] primal The matrix to update.
* @param ws CPD workspace.
* @param which_reg Which regularization we are using.
* @param cdata The constraint data -- 'lambda' if L2 regularization is used.
*/
static void p_admm_optimal_regs(
    matrix_t * const primal,
    cpd_ws * const ws,
    splatt_con_type which_reg,
    void const * const cdata)
{
  /* Add to the diagonal for L2 regularization. */
  if(which_reg == SPLATT_REG_L2) {
    val_t const reg = *((val_t *) cdata);
    mat_add_diag(ws->gram, reg);
  }

  mat_cholesky(ws->gram);

  /* Copy and then solve directly against MTTKRP */
  size_t const bytes = primal->I * primal->J * sizeof(*primal->vals);
  par_memcpy(primal->vals, ws->mttkrp_buf->vals, bytes);
  mat_solve_cholesky(ws->gram, primal);
}




static idx_t p_admm_iterate_chunk(
    matrix_t * primal,
    matrix_t * auxil,
    matrix_t * dual,
    matrix_t * cholesky,
    matrix_t * mttkrp_buf,
    matrix_t * init_buf,
    idx_t mode,
    idx_t which_reg,
    val_t const rho,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const rank = primal->J;

  /* for checking convergence */
  val_t primal_norm     = 0.;
  val_t dual_norm       = 0.;
  val_t primal_residual = 0.;
  val_t dual_residual   = 0.;

  void const * const cdata = cpd_opts->constraints[mode].data;

  bool const chunked = (primal->I != ws->duals[mode]->I);
  //cpd_opts->chunk_sizes[mode];

  /* foreach inner iteration */
  idx_t it;
  for(it=0; it < cpd_opts->max_inner_iterations; ++it) {
    /* save starting point for convergence check */
    size_t const bytes = primal->I * rank * sizeof(*primal->vals);
    if(chunked) {
      memcpy(init_buf->vals, primal->vals, bytes);
    } else {
      par_memcpy(init_buf->vals, primal->vals, bytes);
    }

    /* auxiliary = MTTKRP + (rho .* (primal + dual)) */
    p_setup_auxiliary(primal, mttkrp_buf, dual, rho, auxil, chunked);

    /* Cholesky against auxiliary */
    mat_solve_cholesky(ws->gram, auxil);

    /* APPLY CONSTRAINT / REGULARIZATION */
    /* primal = proximity(auxiliary) */
    switch(which_reg) {
    case SPLATT_CON_NONNEG:
      p_proximity_nonneg(primal, auxil, dual, rho, cdata, chunked);
      break;
    case SPLATT_REG_L1:
      p_proximity_l1(primal, auxil, dual, rho, cdata, chunked);
      break;
    case SPLATT_REG_SMOOTHNESS:
      p_proximity_smooth(primal, auxil, dual, rho, cdata, chunked);
      break;

    case SPLATT_CON_NONE:
    case SPLATT_REG_L2:
      /* XXX: NONE and L2 should have been caught already. */
      assert(false);
      break;
    } /* proximity operatior */

    /* update dual: U += (primal - auxiliary) */
    dual_norm = p_update_dual(primal, auxil, dual, chunked);

    /* check ADMM convergence */
    p_calc_residual(primal, auxil, init_buf,
        &primal_norm, &primal_residual, &dual_residual, chunked);

    /* converged? */
    if((primal_residual <= cpd_opts->inner_tolerance * primal_norm) &&
       (dual_residual   <= cpd_opts->inner_tolerance * dual_norm)) {
      ++it;
      break;
    }
  }

  return it;
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

val_t admm_inner(
    idx_t mode,
    matrix_t * * mats,
    val_t * const restrict column_weights,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  timer_start(&timers[TIMER_ADMM]);

  idx_t const rank = mats[mode]->J;

  /* (A^T * A) .* (B^T * B) .* .... ) */
  mat_form_gram(ws->aTa, ws->gram, ws->nmodes, mode);

  splatt_con_type const which_reg = cpd_opts->constraints[mode].which;
  void const * const cdata        = cpd_opts->constraints[mode].data;

  /* these can be solved optimally without ADMM iterations */
  if(which_reg == SPLATT_CON_NONE || which_reg == SPLATT_REG_L2) {
    p_admm_optimal_regs(mats[mode], ws, which_reg, cdata);

    /* Absorb columns into column_weights if no constraints are applied */
    if(cpd_opts->unconstrained) {
      mat_normalize(mats[mode], column_weights, MAT_NORM_2, NULL, ws->thds);
    }
    return 0.;
  }

  /* Add penalty to diagonal -- value taken from AO-ADMM paper */
  val_t const rho = mat_trace(ws->gram) / (val_t) rank;
  mat_add_diag(ws->gram, rho);

  /* Compute Cholesky factorization to use for forward/backward solves each
   * ADMM iteration */
  mat_cholesky(ws->gram);

  /* Compute number of chunks */
  idx_t num_chunks = 1;
  idx_t const chunk_size = cpd_opts->chunk_sizes[mode];
  if(chunk_size > 0) {
    num_chunks = mats[mode]->I / chunk_size + (mats[mode]->I % chunk_size > 0);
  }

  idx_t it = 0;
  #pragma omp parallel for schedule(dynamic) reduction(+:it) if(num_chunks > 1)
  for(idx_t c=0; c < num_chunks; ++c) {
    idx_t const start = c * chunk_size;
    idx_t const stop = (c == num_chunks-1) ? mats[mode]->I : (c+1)*chunk_size;
    idx_t const offset = start * rank;
    idx_t const nrows = stop - start;

    /* sub-matrix chunks */
    matrix_t primal;
    matrix_t auxil;
    matrix_t dual;
    matrix_t mttkrp;
    matrix_t init_buf;

    /* extract all the workspaces */
    mat_fillptr(&primal, mats[mode]->vals + offset, nrows, rank,
        mats[mode]->rowmajor);
    mat_fillptr(&auxil, ws->auxil->vals + offset, nrows, rank,
        ws->auxil->rowmajor);
    mat_fillptr(&dual, ws->duals[mode]->vals + offset, nrows, rank,
        ws->duals[mode]->rowmajor);
    mat_fillptr(&mttkrp, ws->mttkrp_buf->vals + offset, nrows, rank,
        ws->mttkrp_buf->rowmajor);
    mat_fillptr(&init_buf, ws->mat_init->vals + offset, nrows, rank,
        ws->mat_init->rowmajor);

    /* Run ADMM to convergence and record total ADMM its per row. */
    it += nrows * p_admm_iterate_chunk(&primal, &auxil, &dual, ws->gram,
        &mttkrp, &init_buf, mode, which_reg, rho, ws, cpd_opts, global_opts);
  } /* foreach chunk */

  timer_stop(&timers[TIMER_ADMM]);

  /* return average # iterations */
  return (val_t) it / (val_t) mats[mode]->I;
}



void splatt_cpd_reg_l1(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode,
    splatt_val_t const scale)
{
  /* MAX_NMODES will simply apply regularization to all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_reg_l1(cpd_opts, m, scale);
    }
    return;
  }

  splatt_cpd_con_clear(cpd_opts, mode);

  cpd_opts->unconstrained = false;
  cpd_opts->constraints[mode].which = SPLATT_REG_L1;
  val_t * lambda = splatt_malloc(sizeof(*lambda));
  *lambda = scale;
  cpd_opts->constraints[mode].data = (void *) lambda;
}


void splatt_cpd_reg_l2(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode,
    splatt_val_t const scale)
{
  /* MAX_NMODES will simply apply regularization to all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_reg_l2(cpd_opts, m, scale);
    }
    return;
  }

  splatt_cpd_con_clear(cpd_opts, mode);

  cpd_opts->unconstrained = false;
  cpd_opts->constraints[mode].which = SPLATT_REG_L2;
  val_t * lambda = splatt_malloc(sizeof(*lambda));
  *lambda = scale;
  cpd_opts->constraints[mode].data = (void *) lambda;
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


void splatt_cpd_con_clear(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode)
{
  /* MAX_NMODES will operate on all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_con_clear(cpd_opts, m);
    }
    return;
  }

  reg_smooth_ws * smooth = NULL;

  switch(cpd_opts->constraints[mode].which) {
    /* no-ops */
    case SPLATT_CON_NONE:
    case SPLATT_CON_NONNEG:
      break;

    case SPLATT_REG_L1:
      splatt_free(cpd_opts->constraints[mode].data);
      break;
    case SPLATT_REG_L2:
      splatt_free(cpd_opts->constraints[mode].data);
      break;
    case SPLATT_REG_SMOOTHNESS:
      smooth = cpd_opts->constraints[mode].data;
      mat_free(smooth->banded);
      splatt_free(smooth->pivot);
      splatt_free(smooth);
      break;
  }

  /* clear just in case */
  cpd_opts->constraints[mode].which = SPLATT_CON_NONE;
  cpd_opts->constraints[mode].data = NULL;

  /* check if no constraints */
  cpd_opts->unconstrained = true;
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    if(cpd_opts->constraints[mode].which != SPLATT_CON_NONE) {
      cpd_opts->unconstrained = false;
      break;
    }
  }
}


