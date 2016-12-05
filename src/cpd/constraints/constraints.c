
/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../admm.h"




/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

splatt_cpd_constraint * splatt_alloc_constraint(
    splatt_con_solve_type const solve_type)
{
  splatt_cpd_constraint * con = splatt_malloc(sizeof(*con));

  con->solve_type = solve_type;

  /* zero out structures */
  memset(&(con->hints), 0, sizeof(con->hints));

  con->data = NULL;

  /* function pointers */
  con->init_func = NULL;
  con->prox_func = NULL;
  con->clos_func = NULL;
  con->post_func = NULL;
  con->free_func = NULL;

  asprintf(&(con->description), "UNCONSTRAINED");

  return con;
}


void splatt_register_constraint(
    splatt_cpd_opts * const opts,
    splatt_idx_t const mode,
    splatt_cpd_constraint * con)
{
  /* first clear out any constraint that already existed */
  splatt_free_constraint(opts->constraints[mode]);

  /* now save the new one */
  opts->constraints[mode] = con;
}


void splatt_free_constraint(
    splatt_cpd_constraint * con)
{
  if(con == NULL) {
    return;
  }

  /* Allow constraint to clean up after itself. */
  if(con->free_func != NULL) {
    con->free_func(con->data);
  }

  free(con->description);

  /* Now just delete pointer. */
  splatt_free(con);
}


void cpd_init_constraints(
    splatt_cpd_opts * const opts,
    matrix_t * * primals,
    idx_t const nmodes)
{
  for(idx_t m=0; m < nmodes; ++m) {
    if(opts->constraints[m]->init_func != NULL) {
      idx_t const nrows = primals[m]->I;
      idx_t const ncols = primals[m]->J;
      void * data = opts->constraints[m]->data;
      opts->constraints[m]->init_func(primals[m]->vals, nrows, ncols, data);
    }
  }
}


void cpd_finalize_constraints(
    splatt_cpd_opts * const opts,
    matrix_t * * primals,
    idx_t const nmodes)
{
  for(idx_t m=0; m < nmodes; ++m) {
    if(opts->constraints[m]->post_func != NULL) {
      idx_t const nrows = primals[m]->I;
      idx_t const ncols = primals[m]->J;
      void * data = opts->constraints[m]->data;
      opts->constraints[m]->post_func(primals[m]->vals, nrows, ncols, data);
    }
  }
}
