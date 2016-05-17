



/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "admm.h"
#include "../io.h"
#include "../util.h"






/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static void p_setup_auxiliary(
    idx_t const mode,
    matrix_t * * mats,
    cpd_ws * const ws,
    val_t const penalty)
{
  idx_t const I = mats[mode]->I;
  idx_t const J = mats[mode]->J;

  val_t       * const restrict aux    = ws->auxil[mode]->vals;
  val_t const * const restrict mttkrp = ws->mttkrp_buf->vals;
  val_t const * const restrict primal = mats[mode]->vals;
  val_t const * const restrict dual   = ws->duals[mode]->vals;

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      idx_t const x = j + (i*J);
      aux[x] = mttkrp[x] + penalty * (primal[x] + dual[x]);
    }
  }
}


static void p_apply_proxr(
    idx_t const mode,
    matrix_t * * mats,
    cpd_ws * const ws)
{
  idx_t const I = mats[mode]->I;
  idx_t const J = mats[mode]->J;

  val_t       * const restrict matv = mats[mode]->vals;
  val_t const * const restrict auxl = ws->auxil[mode]->vals;
  val_t const * const restrict dual = ws->duals[mode]->vals;

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      idx_t const x = j + (i*J);
      val_t const v = auxl[x] - dual[x];

      matv[x] = (v > 0.) ? v : 0.;
    }
  }
}


static void p_update_dual(
    idx_t const mode,
    matrix_t * * mats,
    cpd_ws * const ws)
{
  idx_t const I = mats[mode]->I;
  idx_t const J = mats[mode]->J;

  val_t       * const restrict dual = ws->duals[mode]->vals;
  val_t const * const restrict matv = mats[mode]->vals;
  val_t const * const restrict auxl = ws->auxil[mode]->vals;

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      idx_t const x = j + (i*J);
      dual[x] += matv[x] - auxl[x];
    }
  }
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

idx_t admm_inner(
    idx_t mode,
    matrix_t * * mats,
    val_t * const restrict column_weights,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const dim  = mats[mode]->I;
  idx_t const rank = mats[mode]->J;

  /* (A^T * A) .* (B^T * B) .* .... ) */
  mat_form_gram(ws->aTa, ws->gram, ws->nmodes, mode);

#if 0
  /* no constraints / regularization */
  mat_cholesky(ws->gram);
  mat_solve_cholesky(ws->gram, mats[mode]);
  mat_normalize(mats[mode], column_weights, MAT_NORM_2, NULL, ws->thds);
  return 1;
#else

  /* Add penalty to diagonal -- value taken from AO-ADMM paper */
  val_t const rho = mat_trace(ws->gram) / (val_t) rank;
  for(idx_t i=0; i < rank; ++i) {
    ws->gram->vals[i + (i*rank)] += rho;
    column_weights[i] = 1.;
  }

  mat_cholesky(ws->gram);

  matrix_t * mat_init = mat_alloc(dim, rank);

  idx_t it;
  for(it=0; it < cpd_opts->max_inner_iterations; ++it) {
    memcpy(mat_init->vals, mats[mode]->vals, dim * rank * sizeof(val_t));

    /* setup auxiliary: MTTKRP + (rho .* (primal + dual)) */
    p_setup_auxiliary(mode, mats, ws, rho);

    /* Cholesky against auxiliary */
    mat_solve_cholesky(ws->gram, ws->auxil[mode]);

    /* mats[mode] = prox(auxiliary) */
    p_apply_proxr(mode, mats, ws);

    /* update dual: U += (mats[mode] - auxiliary) */
    p_update_dual(mode, mats, ws);

    /* check ADMM convergence */
    val_t primal_residual = 0.;
    val_t dual_residual = 0.;
    val_t prim_norm = 0.;
    val_t dual_norm = 0.;

    val_t const * const restrict matv = mats[mode]->vals;
    val_t const * const restrict auxv = ws->auxil[mode]->vals;
    val_t const * const restrict dual = ws->duals[mode]->vals;
    val_t const * const restrict init = mat_init->vals;

    for(idx_t i=0; i < dim; ++i) {
      for(idx_t j=0; j < rank; ++j) {
        idx_t const x = j + (i*rank);

        val_t const pdiff = matv[x] - auxv[x];
        val_t const ddiff = matv[x] - init[x];
        primal_residual += pdiff * pdiff;
        dual_residual   += ddiff * ddiff;

        prim_norm += matv[x] * matv[x];
        dual_norm += dual[x] * dual[x];
      }
    }

    val_t const eps = cpd_opts->inner_tolerance;
    if((primal_residual <= eps * prim_norm) &&
       (dual_residual   <= eps * dual_norm)) {
      break;
    }
  } /* foreach ADMM iteration */

  mat_free(mat_init);
  return it;
#endif
}
