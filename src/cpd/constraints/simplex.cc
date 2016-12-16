

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

extern "C" {
#include "../admm.h"
}

#include <math.h>
#include <algorithm>
#include <functional>



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Project a vector onto the probability simplex.
*
* @param projected The vector to project.
* @param buffer An allocated buffer of length N.
* @param N The length of 'projected' and 'buffer'.
*/
static void p_project_vector(
    val_t * const restrict projected,
    val_t * const restrict buffer,
    idx_t const N)
{
  /* sort buffer into non-increasing order */
  for(idx_t j=0; j < N; ++j) {
    buffer[j] = projected[j];
  }
  std::sort(buffer, buffer + N, std::greater<val_t>());

  val_t running_sum = -1.; /* only -1 once */
  idx_t pivot;
  for(pivot = 0; pivot < N; ++pivot) {
    running_sum += buffer[pivot];

    val_t const v = buffer[pivot] - (running_sum / ((val_t) (pivot+1)));

    /* pivot is j-1, so adjust accordingly */
    if(v <= 0.) {
      running_sum -= buffer[pivot];
      break;
    }
  }
  val_t const theta = running_sum / ((val_t) pivot);

  /* update row */
  for(idx_t j=0; j < N; ++j) {
    projected[j] -= theta;
    projected[j] = SS_MAX(projected[j], 0.);
  }
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


extern "C"
/**
* @brief The proximal update for a row simplex constraint. Since rows are
*        assumed to be small, the naive (ncols * log(ncols)) algorithm is used.
*
* @param[out] primal The row-major matrix to update.
* @param nrows The number of rows in primal.
* @param ncols The number of columns in primal.
* @param offset Not used.
* @param data Not used.
* @param rho Not used.
* @param should_parallelize If true, parallelize.
*/
void splatt_rowsimp_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize)
{
  #pragma omp parallel if(should_parallelize)
  {
    val_t * row_buf = (val_t *) splatt_malloc(ncols * sizeof(*row_buf));

    #pragma omp for
    for(idx_t i=0; i < nrows; ++i) {
      val_t * const restrict row = primal + (i * ncols);
      p_project_vector(row, row_buf, ncols);
    }

    splatt_free(row_buf);
  } /* end omp parallel */
}


extern "C"
/**
* @brief The proximal update for a column simplex constraint.
*         
*   TODO: This should probably use the linear-time randomized algorithm.
*
* @param[out] primal The row-major matrix to update.
* @param nrows The number of rows in primal.
* @param ncols The number of columns in primal.
* @param offset Not used.
* @param data Not used.
* @param rho Not used.
* @param should_parallelize If true, parallelize.
*/
void splatt_colsimp_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize)
{
  /*
   * TODO: handle the case of parallelism when ncols < nthreads
   */

  #pragma omp parallel if(should_parallelize)
  {
    val_t * col_buf = (val_t *) splatt_malloc(nrows * sizeof(*col_buf));
    val_t * prj_buf = (val_t *) splatt_malloc(nrows * sizeof(*prj_buf));

    #pragma omp for
    for(idx_t j=0; j < ncols; ++j) {

      /* first copy column nto col_buf */
      for(idx_t i=0; i < nrows; ++i) {
        col_buf[i] = primal[j + (i*ncols)];
      }

      /* project the column */
      p_project_vector(col_buf, prj_buf, nrows);

      /* now write back */
      for(idx_t i=0; i < nrows; ++i) {
        primal[j + (i*ncols)] = col_buf[i];
      }
    }

    splatt_free(prj_buf);
    splatt_free(col_buf);
  } /* end omp parallel */
}



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


splatt_error_type splatt_register_rowsimp(
    splatt_cpd_opts * opts,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes)
{
  for(idx_t m = 0; m < num_modes; ++m) {
    idx_t const mode = modes_included[m];

    splatt_cpd_constraint * con = splatt_alloc_constraint(SPLATT_CON_ADMM);

    /* only fill the details that are used */
    con->prox_func = splatt_rowsimp_prox;

    /* set hints to assist optimizations */
    con->hints.row_separable     = true;
    con->hints.sparsity_inducing = true;

    sprintf(con->description, "ROW-SIMPLEX");

    /* add to the CPD factorization */
    splatt_register_constraint(opts, mode, con);
  }

  return SPLATT_SUCCESS;
}


splatt_error_type splatt_register_colsimp(
    splatt_cpd_opts * opts,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes)
{
  for(idx_t m = 0; m < num_modes; ++m) {
    idx_t const mode = modes_included[m];

    splatt_cpd_constraint * con = splatt_alloc_constraint(SPLATT_CON_ADMM);

    /* only fill the details that are used */
    con->prox_func = splatt_colsimp_prox;

    /* set hints to assist optimizations */
    con->hints.sparsity_inducing = true;

    sprintf(con->description, "COL-SIMPLEX");

    /* add to the CPD factorization */
    splatt_register_constraint(opts, mode, con);
  }

  return SPLATT_SUCCESS;
}

