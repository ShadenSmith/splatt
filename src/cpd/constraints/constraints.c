
/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../../base.h"




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
    splatt_cpd_constraint const * const con)
{

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

