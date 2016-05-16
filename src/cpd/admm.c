



/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "admm.h"
#include "../io.h"






/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static void p_setup_gram(
    matrix_t * const gram,
    splatt_cpd_opts const * const cpd_opts)
{
  /* add regularization, etc. */

}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

idx_t admm_inner(
    idx_t mode,
    matrix_t * * mats,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  matrix_t * const gram = ws->aTa[mode];
  idx_t const rank = gram->J;

  p_setup_gram(gram, cpd_opts);

  mat_cholesky(gram);
  mat_solve_cholesky(gram, mats[mode]);

  idx_t it;
  for(it=0; it < cpd_opts->max_inner_iterations; ++it) {
  }

  return it;
}
