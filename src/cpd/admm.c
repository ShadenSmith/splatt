



/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "admm.h"
#include "../util.h"





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
* @brief Project onto the non-negative orthant.
*
* @param[out] mat_primal The primal variable (factor matrix) to update.
* @param penalty The current penalty parameter (rho) -- not used.
* @param ws CPD workspace data -- not used.
* @param mode The mode we are updating -- not used.
* @param data Constraint-specific data -- not used.
*/
static void p_proximity_nonneg(
    matrix_t  * const mat_primal,
    val_t const penalty,
    cpd_ws * const ws,
    idx_t const mode,
    void const * const data)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxl = ws->auxil->vals;
  val_t const * const restrict dual = ws->duals[mode]->vals;

  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < I * J; ++x) {
    val_t const v = auxl[x] - dual[x];
    matv[x] = (v > 0.) ? v : 0.;
  }
}


/**
* @brief Perform Lasso regularization via soft thresholding.
*
* @param[out] mat_primal The primal variable (factor matrix) to update.
* @param penalty The current penalty parameter (rho).
* @param ws CPD workspace data -- used for regularization parameter.
* @param mode The mode we are updating.
* @param data Constraint-specific data -- we store lambda in this.
*/
static void p_proximity_l1(
    matrix_t  * const mat_primal,
    val_t const penalty,
    cpd_ws * const ws,
    idx_t const mode,
    void const * const data)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxl = ws->auxil->vals;
  val_t const * const restrict dual = ws->duals[mode]->vals;

  val_t const lambda = *((val_t *) data);
  val_t const mult = lambda / penalty;

  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < I * J; ++x) {
    val_t const v = auxl[x] - dual[x];

    /* Could this be done faster */
    if(v > mult) {
      matv[x] = v - mult;
    } else if(v < -mult) {
      matv[x] = v + mult;
    } else {
      matv[x] = 0.;
    }
  }
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
*/
static void p_setup_auxiliary(
    matrix_t const * const mat_primal,
    matrix_t const * const mat_mttkrp,
    matrix_t const * const mat_dual,
    val_t const penalty,
    matrix_t * const mat_auxil)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict aux    = mat_auxil->vals;

  val_t const * const restrict mttkrp = mat_mttkrp->vals;
  val_t const * const restrict primal = mat_primal->vals;
  val_t const * const restrict dual   = mat_dual->vals;

  #pragma omp parallel for schedule(static)
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
*
* @return The norm of the new dual; || mat_dual ||_F^2.
*/
static val_t p_update_dual(
    matrix_t const * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t * const mat_dual)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict dual = mat_dual->vals;
  val_t const * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxl = mat_auxil->vals;

  val_t dual_norm = 0.;

  #pragma omp parallel for schedule(static) reduction(+:dual_norm)
  for(idx_t x=0; x < I * J; ++x) {
    dual[x] += matv[x] - auxl[x];
    dual_norm += dual[x] * dual[x];
  }

  return dual_norm;
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
*/
static void p_calc_residual(
    matrix_t const * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t const * const mat_init,
    val_t * primal_norm,
    val_t * primal_resid,
    val_t * dual_resid)
{
  val_t const * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxv = mat_auxil->vals;
  val_t const * const restrict init = mat_init->vals;

  idx_t const nrows = mat_primal->I;
  idx_t const ncols = mat_primal->J;

  val_t p_norm = 0.;
  val_t p_resid = 0.;
  val_t d_resid = 0.;

  #pragma omp parallel for reduction(+: p_norm, p_resid, d_resid)
  for(idx_t x=0; x < nrows * ncols; ++x) {
    val_t const pdiff = matv[x] - auxv[x];
    val_t const ddiff = matv[x] - init[x];

    p_norm  += matv[x] * matv[x];
    p_resid += pdiff * pdiff;
    d_resid += ddiff * ddiff;
  }

  *primal_norm  = p_norm;
  *primal_resid = p_resid;
  *dual_resid   = d_resid;
}



/**
* @brief Optimally compute the new primal variable when no regularization or
*        only L2 regularization is present.
*
* @param[out] primal The matrix to update.
* @param[out] column_weights Column weights to absorb into if no
*                            regularization.
* @param ws CPD workspace.
* @param which_reg Which regularization we are using.
* @param cdata The constraint data -- 'lambda' if L2 regularization is used.
*/
static void p_admm_optimal_regs(
    matrix_t * const primal,
    val_t * const restrict column_weights,
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

  /* Absorb columns into column_weights */
  if(which_reg == SPLATT_CON_NONE) {
    mat_normalize(primal, column_weights, MAT_NORM_2, NULL, ws->thds);
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
  timer_start(&timers[TIMER_ADMM]);

  idx_t const rank = mats[mode]->J;

  /* (A^T * A) .* (B^T * B) .* .... ) */
  mat_form_gram(ws->aTa, ws->gram, ws->nmodes, mode);

  splatt_con_type const which_reg = cpd_opts->constraints[mode].which;
  void const * const cdata        = cpd_opts->constraints[mode].data;

  /* these can be solved optimally without ADMM iterations */
  if(which_reg == SPLATT_CON_NONE || which_reg == SPLATT_REG_L2) {
    p_admm_optimal_regs(mats[mode], column_weights, ws, which_reg, cdata);
    return 0;
  }

  /* Add penalty to diagonal -- value taken from AO-ADMM paper */
  val_t const rho = mat_trace(ws->gram) / (val_t) rank;
  mat_add_diag(ws->gram, rho);

  /* Compute Cholesky factorization to use for forward/backward solves each
   * ADMM iteration */
  mat_cholesky(ws->gram);

  /* for checking convergence */
  val_t primal_norm     = 0.;
  val_t dual_norm       = 0.;
  val_t primal_residual = 0.;
  val_t dual_residual   = 0.;

  idx_t it;
  for(it=0; it < cpd_opts->max_inner_iterations; ++it) {
    /* save starting point for convergence check */
    size_t const bytes = mats[mode]->I * rank * sizeof(*(*mats)->vals);
    par_memcpy(ws->mat_init->vals, mats[mode]->vals, bytes);

    /* auxiliary = MTTKRP + (rho .* (primal + dual)) */
    p_setup_auxiliary(mats[mode], ws->mttkrp_buf, ws->duals[mode], rho,
        ws->auxil);

    /* Cholesky against auxiliary */
    mat_solve_cholesky(ws->gram, ws->auxil);

    /* APPLY CONSTRAINT / REGULARIZATION */
    /* mats[mode] = proximity(auxiliary) */
    switch(which_reg) {

    case SPLATT_REG_L1:
      p_proximity_l1(mats[mode], rho, ws, mode, cdata);
      break;

    case SPLATT_CON_NONNEG:
      p_proximity_nonneg(mats[mode], rho, ws, mode, cdata);
      break;

    case SPLATT_CON_NONE:
    case SPLATT_REG_L2:
      /* XXX: NONE and L2 should have been caught already. */
      assert(false);
      break;
    } /* proximity operatior */

    /* update dual: U += (mats[mode] - auxiliary) */
    dual_norm = p_update_dual(mats[mode], ws->auxil,
        ws->duals[mode]);

    /* check ADMM convergence */
    p_calc_residual(mats[mode], ws->auxil, ws->mat_init,
        &primal_norm, &primal_residual, &dual_residual);

    /* converged? */
    if((primal_residual <= cpd_opts->inner_tolerance * primal_norm) &&
       (dual_residual   <= cpd_opts->inner_tolerance * dual_norm)) {
      ++it;
      break;
    }
  } /* foreach ADMM iteration */

  timer_stop(&timers[TIMER_ADMM]);
  return it;
}




