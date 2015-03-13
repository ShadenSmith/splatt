
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "mpi.h"
#include "../mttkrp.h"
#include "../timer.h"
#include "../thd_info.h"
#include "../tile.h"

#include <omp.h>

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Flush the updated values in globalmat to our local representation.
*
* @param tt The tensor we are operating on.
* @param localmat The local matrix to update.
* @param globalmat The recently updated global matrix.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the factor matrices.
* @param mode The mode we are operating on.
*/
static void __flush_glob_to_local(
  sptensor_t const * const tt,
  matrix_t * const localmat,
  matrix_t const * const globalmat,
  rank_info const * const rinfo,
  idx_t const nfactors,
  idx_t const mode)
{
  idx_t const m = mode;
  val_t const * const restrict gmatv = globalmat->vals;
  val_t * const restrict matv = localmat->vals;

  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const mat_end = rinfo->mat_end[m];
  idx_t const start = rinfo->ownstart[m];
  idx_t const end = rinfo->ownend[m];

  /* now add partials to my global matrix */
  if(tt->indmap[m] == NULL) {
    for(idx_t i=start; i < end; ++i) {
      assert(i >= mat_start && i < mat_end);
      idx_t const row = i - mat_start;
      for(idx_t f=0; f < nfactors; ++f) {
        matv[f+(i*nfactors)] = gmatv[f+(row*nfactors)];
      }
    }
  } else {
    idx_t const * const restrict indmap = tt->indmap[m];
    for(idx_t i=start; i < end; ++i) {
      assert(indmap[i] >= mat_start && indmap[i] < mat_end);
      idx_t const row = indmap[i] - mat_start;
      for(idx_t f=0; f < nfactors; ++f) {
        matv[f+(i*nfactors)] = gmatv[f+(row*nfactors)];
      }
    }
  }
}


/**
* @brief Do an all-to-all communication of exchanging updated rows with other
*        ranks. We send globmats[mode] to the needing ranks and receive other
*        ranks' globmats entries which we store in mats[mode].
*
* @param tt The tensor we are operating on.
* @param nbr2globs_buf Buffer at least as large as as there are rows to send
*                      (for each rank).
* @param nbr2local_buf Buffer at least as large as there are rows to receive.
* @param localmat Local factor matrix which receives updated values.
* @param globalmat Global factor matrix (owned by me) which is sent to ranks.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the factor matrices.
* @param mode The mode to exchange along.
*/
static void __update_rows(
  sptensor_t const * const tt,
  val_t * const restrict nbr2globs_buf,
  val_t * const restrict nbr2local_buf,
  matrix_t * const localmat,
  matrix_t * const globalmat,
  rank_info const * const rinfo,
  idx_t const nfactors,
  idx_t const mode)
{
  timer_start(&timers[TIMER_MPI]);
  idx_t const m = mode;
  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const * const restrict nbr2globs_inds = rinfo->nbr2globs_inds[m];
  val_t const * const restrict gmatv = globalmat->vals;

  /* first prepare all rows that I own and need to send */
  for(idx_t s=0; s < rinfo->nnbr2globs[m]; ++s) {
    idx_t const row = nbr2globs_inds[s] - mat_start;
    for(idx_t f=0; f < nfactors; ++f) {
      nbr2globs_buf[f+(s*nfactors)] = gmatv[f+(row*nfactors)];
    }
  }

  /* grab ptr/disp from rinfo. nbr2local and local2nbr will have the same
   * structure so we just reuse those */
  int const * const restrict nbr2globs_ptr = rinfo->nbr2globs_ptr[m];
  int const * const restrict nbr2local_ptr = rinfo->local2nbr_ptr[m];
  int const * const restrict nbr2globs_disp = rinfo->nbr2globs_disp[m];
  int const * const restrict nbr2local_disp = rinfo->local2nbr_disp[m];

  timer_start(&timers[TIMER_MPI_IDLE]);
  MPI_Barrier(rinfo->layer_comm[m]);
  timer_stop(&timers[TIMER_MPI_IDLE]);

  /* exchange rows */
  timer_start(&timers[TIMER_MPI_COMM]);
  MPI_Alltoallv(nbr2globs_buf, nbr2globs_ptr, nbr2globs_disp, SS_MPI_VAL,
                nbr2local_buf, nbr2local_ptr, nbr2local_disp, SS_MPI_VAL,
                rinfo->layer_comm[m]);
  timer_stop(&timers[TIMER_MPI_COMM]);

  /* now write incoming nbr2locals to my local matrix */
  idx_t const * const restrict local2nbr_inds = rinfo->local2nbr_inds[m];
  val_t * const restrict matv = localmat->vals;
  for(idx_t r=0; r < rinfo->nlocal2nbr[m]; ++r) {
    idx_t const row = local2nbr_inds[r];
    for(idx_t f=0; f < nfactors; ++f) {
      matv[f+(row*nfactors)] = nbr2local_buf[f+(r*nfactors)];
    }
  }

  /* ensure the local matrix is up to date too */
  __flush_glob_to_local(tt, localmat, globalmat, rinfo, nfactors, m);
  timer_stop(&timers[TIMER_MPI]);
}



/**
* @brief Do a reduction (sum) of all neighbor partial products which I own.
*        Updates are written to globalmat.
*
* @param local2nbr_buf A buffer at least as large as nlocal2nbr.
* @param nbr2globs_buf A buffer at least as large as nnbr2globs.
* @param localmat My local matrix containing partial products for other ranks.
* @param globalmat The global factor matrix to update.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the matrices.
* @param mode The mode to operate on.
*/
static void __reduce_rows(
  val_t * const restrict local2nbr_buf,
  val_t * const restrict nbr2globs_buf,
  matrix_t const * const localmat,
  matrix_t * const globalmat,
  rank_info const * const rinfo,
  idx_t const nfactors,
  idx_t const mode)
{
  timer_start(&timers[TIMER_MPI]);
  idx_t const m = mode;

  val_t const * const restrict matv = localmat->vals;
  idx_t const * const restrict local2nbr_inds = rinfo->local2nbr_inds[m];

  /* copy my partial products into the sendbuf */
  for(idx_t s=0; s < rinfo->nlocal2nbr[m]; ++s) {
    idx_t const row = local2nbr_inds[s];
    for(idx_t f=0; f < nfactors; ++f) {
      local2nbr_buf[f + (s*nfactors)] = matv[f + (row*nfactors)];
    }
  }

  /* grab ptr/disp from rinfo. nbr2local and local2nbr will have the same
   * structure so we just reuse those */
  int const * const restrict nbr2globs_ptr = rinfo->nbr2globs_ptr[m];
  int const * const restrict nbr2local_ptr = rinfo->local2nbr_ptr[m];
  int const * const restrict nbr2globs_disp = rinfo->nbr2globs_disp[m];
  int const * const restrict nbr2local_disp = rinfo->local2nbr_disp[m];

  timer_start(&timers[TIMER_MPI_IDLE]);
  MPI_Barrier(rinfo->layer_comm[m]);
  timer_stop(&timers[TIMER_MPI_IDLE]);

  timer_start(&timers[TIMER_MPI_COMM]);
  /* exchange rows */
  MPI_Alltoallv(local2nbr_buf, nbr2local_ptr, nbr2local_disp, SS_MPI_VAL,
                nbr2globs_buf, nbr2globs_ptr, nbr2globs_disp, SS_MPI_VAL,
                rinfo->layer_comm[m]);
  timer_stop(&timers[TIMER_MPI_COMM]);


  /* now add received rows to globmats */
  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const * const restrict nbr2globs_inds = rinfo->nbr2globs_inds[m];
  val_t * const restrict gmatv = globalmat->vals;
  for(idx_t r=0; r < rinfo->nnbr2globs[m]; ++r) {
    idx_t const row = nbr2globs_inds[r] - mat_start;
    for(idx_t f=0; f < nfactors; ++f) {
      gmatv[f+(row*nfactors)] += nbr2globs_buf[f+(r*nfactors)];
    }
  }
  timer_stop(&timers[TIMER_MPI]);
}



/**
* @brief Add my own partial products to the global matrix that I own.
*
* @param tt The tensor I am computing on.
* @param localmat The local matrix containing my partial products.
* @param globmat The global factor matrix I am writing to.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the matrices.
* @param mode The mode I am operating on.
*/
static void __add_my_partials(
  sptensor_t const * const tt,
  matrix_t const * const localmat,
  matrix_t * const globmat,
  rank_info const * const rinfo,
  idx_t const nfactors,
  idx_t const mode)
{
  idx_t const m = mode;
  val_t * const restrict gmatv = globmat->vals;
  val_t const * const restrict matv = localmat->vals;

  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const mat_end = rinfo->mat_end[m];
  idx_t const start = rinfo->ownstart[m];
  idx_t const end = rinfo->ownend[m];

  memset(gmatv, 0, globmat->I * nfactors * sizeof(val_t));

  /* now add partials to my global matrix */
  if(tt->indmap[m] == NULL) {
    for(idx_t i=start; i < end; ++i) {
      assert(i >= mat_start && i < mat_end);
      idx_t const row = i - mat_start;
      for(idx_t f=0; f < nfactors; ++f) {
        gmatv[f+(row*nfactors)] = matv[f+(i*nfactors)];
      }
    }
  } else {
    idx_t const * const restrict indmap = tt->indmap[m];
    for(idx_t i=start; i < end; ++i) {
      assert(indmap[i] >= mat_start && indmap[i] < mat_end);
      idx_t const row = indmap[i] - mat_start;
      for(idx_t f=0; f < nfactors; ++f) {
        gmatv[f+(row*nfactors)] = matv[f+(i*nfactors)];
      }
    }
  }
}




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mpi_cpd(
  sptensor_t * const tt,
  matrix_t ** mats,
  matrix_t ** globmats,
  rank_info * const rinfo,
  cpd_opts const * const opts)
{
  MPI_Barrier(rinfo->comm_3d);

  idx_t const nfactors = opts->rank;

  idx_t maxdim = 0;
  idx_t maxlocal2nbr = 0;
  idx_t maxnbr2globs = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    maxlocal2nbr = SS_MAX(maxlocal2nbr, rinfo->nlocal2nbr[m]);
    maxnbr2globs = SS_MAX(maxnbr2globs, rinfo->nnbr2globs[m]);
    maxdim = SS_MAX(globmats[m]->I, maxdim);
  }
  maxlocal2nbr *= nfactors;
  maxnbr2globs *= nfactors;

  val_t * local2nbr_buf = (val_t *) malloc(maxlocal2nbr * sizeof(val_t));
  val_t * nbr2globs_buf = (val_t *) malloc(maxnbr2globs * sizeof(val_t));
  matrix_t * m1 = mat_alloc(maxdim, nfactors);

  /* exchange initial matrices */
  for(idx_t m=1; m < tt->nmodes; ++m) {
    __update_rows(tt, nbr2globs_buf, local2nbr_buf, mats[m], globmats[m],
        rinfo, nfactors, m);
  }

  /* allocate tensor */
  ftensor_t * ft = ften_alloc(tt, opts->tile);

  /* setup thread structures */
  omp_set_num_threads(opts->nthreads);
  thd_info * thds;
  if(opts->tile) {
    thds = thd_init(opts->nthreads, nfactors * sizeof(val_t) + 64,
      TILE_SIZES[0] * nfactors * sizeof(val_t) + 64);
  } else {
    thds = thd_init(opts->nthreads, nfactors * sizeof(val_t) + 64, 0);
  }

  sp_timer_t itertime;

  timer_start(&timers[TIMER_CPD]);
  for(idx_t it=0; it < opts->niters; ++it) {
    timer_fstart(&itertime);
    for(idx_t m=0; m < tt->nmodes; ++m) {
      mats[MAX_NMODES]->I = ft->dims[m];
      m1->I = globmats[m]->I;

      /* M1 = X * (C o B) */
      timer_start(&timers[TIMER_MTTKRP]);
      mttkrp_splatt(ft, mats, m, thds, opts->nthreads);
      timer_stop(&timers[TIMER_MTTKRP]);

      /* add my partial multiplications to globmats[m] */
      __add_my_partials(tt, mats[MAX_NMODES], m1, rinfo, nfactors, m);

      /* incorporate neighbors' partials */
      __reduce_rows(local2nbr_buf, nbr2globs_buf, mats[MAX_NMODES],
          m1, rinfo, nfactors, m);

      memcpy(globmats[m]->vals, m1->vals, m1->I * m1->J * sizeof(val_t));

      /* send updated rows to neighbors */
      __update_rows(tt, nbr2globs_buf, local2nbr_buf, mats[m], globmats[m],
          rinfo, nfactors, m);

      //timer_start(&timers[TIMER_MPI_IDLE]);
      //MPI_Barrier(rinfo->comm_3d);
      //timer_stop(&timers[TIMER_MPI_IDLE]);
    } /* foreach mode */

    timer_stop(&itertime);
    if(rinfo->rank == 0) {
      printf("    its = %3"SS_IDX" (%0.3fs)  fit = %0.3f\n", it+1,
          itertime.seconds, 0.1);
    }
  } /* foreach iteration */

  /* clean up */
  ften_free(ft);
  mat_free(m1);
  free(local2nbr_buf);
  free(nbr2globs_buf);

  MPI_Barrier(rinfo->comm_3d);
  timer_stop(&timers[TIMER_CPD]);
}


