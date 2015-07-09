
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../splatt_mpi.h"
#include "../mttkrp.h"
#include "../timer.h"
#include "../thd_info.h"
#include "../tile.h"

#include <math.h>
#include <omp.h>


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Flush the updated values in globalmat to our local representation.
*
* @param localmat The local matrix to update.
* @param globalmat The recently updated global matrix.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the factor matrices.
* @param mode The mode we are operating on.
*/
static void __flush_glob_to_local(
  idx_t const * const indmap,
  matrix_t * const localmat,
  matrix_t const * const globalmat,
  rank_info const * const rinfo,
  idx_t const nfactors,
  idx_t const mode)
{
  idx_t const m = mode;

  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const mat_end = rinfo->mat_end[m];
  idx_t const start = rinfo->ownstart[m];
  idx_t const nowned = rinfo->nowned[m];

  assert(start + nowned <= localmat->I);

  memcpy(localmat->vals + (start*nfactors),
         globalmat->vals,
         nowned * nfactors * sizeof(val_t));
}


/**
* @brief Do a reduction (sum) of all neighbor partial products which I own.
*        Updates are written to globalmat.
*        This version accomplishes the communication with an MPI_Alltoallv().
*
* @param local2nbr_buf A buffer at least as large as nlocal2nbr.
* @param nbr2globs_buf A buffer at least as large as nnbr2globs.
* @param localmat My local matrix containing partial products for other ranks.
* @param globalmat The global factor matrix to update.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the matrices.
* @param m The mode to operate on.
*/
static void __reduce_rows_all2all(
  val_t * const restrict local2nbr_buf,
  val_t * const restrict nbr2globs_buf,
  matrix_t const * const localmat,
  matrix_t * const globalmat,
  rank_info const * const rinfo,
  idx_t const nfactors,
  idx_t const m)
{
  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const * const restrict local2nbr_inds = rinfo->local2nbr_inds[m];
  idx_t const * const restrict nbr2globs_inds = rinfo->nbr2globs_inds[m];
  val_t const * const restrict matv = localmat->vals;
  val_t * const restrict gmatv = globalmat->vals;

  /* copy my partial products into the sendbuf */
  #pragma omp parallel for
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
  MPI_Alltoallv(local2nbr_buf, nbr2local_ptr, nbr2local_disp, SPLATT_MPI_VAL,
                nbr2globs_buf, nbr2globs_ptr, nbr2globs_disp, SPLATT_MPI_VAL,
                rinfo->layer_comm[m]);
  timer_stop(&timers[TIMER_MPI_COMM]);

  int const lrank = rinfo->layer_rank[m];
  int const lsize = rinfo->layer_size[m];

  /* Now add received partial products. We can parallelize the additions from
   * each process. */
  #pragma omp parallel
  for(int p=1; p < lsize; ++p) {
    int const porig = (p + lrank) % lsize;
    /* The number of rows to recv from porig */
    int const nrecvs = nbr2globs_ptr[porig] / nfactors;
    int const disp  = nbr2globs_disp[porig] / nfactors;

    #pragma omp for
    for(int r=disp; r < disp + nrecvs; ++r) {
      idx_t const row = nbr2globs_inds[r] - mat_start;
      for(idx_t f=0; f < nfactors; ++f) {
        gmatv[f+(row*nfactors)] += nbr2globs_buf[f+(r*nfactors)];
      }
    }
  } /* end recvs */
}


static void __reduce_rows_point2point(
  val_t * const restrict local2nbr_buf,
  val_t * const restrict nbr2globs_buf,
  matrix_t const * const localmat,
  matrix_t * const globalmat,
  rank_info * const rinfo,
  idx_t const nfactors,
  idx_t const m)
{
  int const lrank = rinfo->layer_rank[m];
  int const lsize = rinfo->layer_size[m];

  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const * const restrict local2nbr_inds = rinfo->local2nbr_inds[m];
  idx_t const * const restrict nbr2globs_inds = rinfo->nbr2globs_inds[m];
  val_t const * const restrict matv = localmat->vals;
  val_t * const restrict gmatv = globalmat->vals;

  #pragma omp parallel default(shared)
  {
    /* SENDS */
    for(int p=1; p < lsize; ++p) {
      /* destination process -- starting from p+1 helps avoid contention */
      int const pdest = (p + lrank) % lsize;
      /* The number of rows to send to pdest */
      int const nsends = rinfo->local2nbr_ptr[m][pdest] / nfactors;
      int const disp  = rinfo->local2nbr_disp[m][pdest] / nfactors;
      if(nsends == 0) {
        continue;
      }

      /* first prepare all rows that I own and need to send */
      #pragma omp for
      for(int s=disp; s < disp+nsends; ++s) {
        idx_t const row = local2nbr_inds[s];
        for(idx_t f=0; f < nfactors; ++f) {
          local2nbr_buf[f + (s*nfactors)] = matv[f + (row*nfactors)];
        }
      }

      /* do the actual communication */
      #pragma omp master
      {
        timer_start(&timers[TIMER_MPI_COMM]);
        MPI_Isend(&(local2nbr_buf[disp*nfactors]), nsends*nfactors, SPLATT_MPI_VAL,
            pdest, 0, rinfo->layer_comm[m], &(rinfo->req));
        timer_stop(&timers[TIMER_MPI_COMM]);
      }
    } /* end sends */


    /* RECVS */
    for(int p=1; p < lsize; ++p) {
      int const porig = (p + lrank) % lsize;
      /* The number of rows to recv from porig */
      int const nrecvs = rinfo->nbr2globs_ptr[m][porig] / nfactors;
      int const disp  = rinfo->nbr2globs_disp[m][porig] / nfactors;
      if(nrecvs == 0) {
        continue;
      }

      /* do the actual communication */
      #pragma omp master
      {
        timer_start(&timers[TIMER_MPI_COMM]);
        MPI_Recv(&(nbr2globs_buf[disp*nfactors]), nrecvs*nfactors, SPLATT_MPI_VAL,
            porig, 0, rinfo->layer_comm[m], &(rinfo->status));
        timer_stop(&timers[TIMER_MPI_COMM]);
      }

      /* wait until recv is done */
      #pragma omp barrier

      /* now add partial products */
      #pragma omp for
      for(int r=disp; r < disp + nrecvs; ++r) {
        idx_t const row = nbr2globs_inds[r] - mat_start;
        for(idx_t f=0; f < nfactors; ++f) {
          gmatv[f+(row*nfactors)] += nbr2globs_buf[f+(r*nfactors)];
        }
      }
    } /* end recvs */
  } /* end omp parallel */
}


/**
* @brief Exchange updated factor rows with all MPI ranks in the same layer.
*        This version accomplishes the communication with individual MPI_Isend
*        and MPI_Recv.
*        We send globmats[mode] to the needing ranks and receive other ranks'
*        globmats entries which we store in mats[mode].
*
* @param nbr2globs_buf Buffer at least as large as as there are rows to send
*                      (for each rank).
* @param nbr2local_buf Buffer at least as large as there are rows to receive.
* @param localmat Local factor matrix which receives updated values.
* @param globalmat Global factor matrix (owned by me) which is sent to ranks.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the factor matrices.
* @param mode The mode to exchange along.
*/
static void __update_rows_point2point(
  val_t * const nbr2globs_buf,
  val_t * const nbr2local_buf,
  matrix_t * const localmat,
  matrix_t * const globalmat,
  rank_info * const rinfo,
  idx_t const nfactors,
  idx_t const mode)
{
  idx_t const m = mode;
  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const * const nbr2globs_inds = rinfo->nbr2globs_inds[m];
  idx_t const * const local2nbr_inds = rinfo->local2nbr_inds[m];
  val_t const * const gmatv = globalmat->vals;
  val_t * const matv = localmat->vals;

  int const lrank = rinfo->layer_rank[m];
  int const lsize = rinfo->layer_size[m];

  #pragma omp parallel default(shared)
  {
    /* SENDS */
    for(int p=1; p < lsize; ++p) {
      /* destination process -- starting from p+1 helps avoid contention */
      int const pdest = (p + lrank) % lsize;
      /* The number of rows to send to pdest */
      int const nsends = rinfo->nbr2globs_ptr[m][pdest] / nfactors;
      int const disp = rinfo->nbr2globs_disp[m][pdest] / nfactors;

      if(nsends == 0) {
        continue;
      }

      /* first prepare all rows that I own and need to send */
      #pragma omp for
      for(int s=disp; s < disp+nsends; ++s) {
        idx_t const row = nbr2globs_inds[s] - mat_start;
        for(idx_t f=0; f < nfactors; ++f) {
          nbr2globs_buf[f+(s*nfactors)] = gmatv[f+(row*nfactors)];
        }
      }

      /* do the actual communication */
      #pragma omp master
      {
        timer_start(&timers[TIMER_MPI_COMM]);
        MPI_Isend(&(nbr2globs_buf[disp*nfactors]), nsends*nfactors, SPLATT_MPI_VAL,
            pdest, 0, rinfo->layer_comm[m], &(rinfo->req));
        timer_stop(&timers[TIMER_MPI_COMM]);
      }
    } /* end sends */

    /* RECVS */
    for(int p=1; p < lsize; ++p) {
      int const porig = (p + lrank) % lsize;
      /* The number of rows to recv from porig */
      int const nrecvs = rinfo->local2nbr_ptr[m][porig] / nfactors;
      int const disp = rinfo->local2nbr_disp[m][porig] / nfactors;

      if(nrecvs == 0) {
        continue;
      }

      /* do the actual communication */
      #pragma omp master
      {
        timer_start(&timers[TIMER_MPI_COMM]);
        MPI_Recv(&(nbr2local_buf[disp*nfactors]), nrecvs*nfactors, SPLATT_MPI_VAL,
            porig, 0, rinfo->layer_comm[m], &(rinfo->status));
        timer_stop(&timers[TIMER_MPI_COMM]);
      }

      /* wait until recv is done */
      #pragma omp barrier

      /* now write incoming nbr2locals to my local matrix */
      #pragma omp for
      for(int r=disp; r < disp + nrecvs; ++r) {
        idx_t const row = local2nbr_inds[r];
        for(idx_t f=0; f < nfactors; ++f) {
          matv[f+(row*nfactors)] = nbr2local_buf[f+(r*nfactors)];
        }
      }
    } /* end recvs */
  } /* end omp parallel */
}


/**
* @brief Exchange updated factor rows with all MPI ranks in the same layer.
*        This version accomplishes the communication with an MPI_Alltoallv().
*        We send globmats[mode] to the needing ranks and receive other ranks'
*        globmats entries which we store in mats[mode].
*
* @param nbr2globs_buf Buffer at least as large as as there are rows to send
*                      (for each rank).
* @param nbr2local_buf Buffer at least as large as there are rows to receive.
* @param localmat Local factor matrix which receives updated values.
* @param globalmat Global factor matrix (owned by me) which is sent to ranks.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the factor matrices.
* @param mode The mode to exchange along.
*/
static void __update_rows_all2all(
  val_t * const nbr2globs_buf,
  val_t * const nbr2local_buf,
  matrix_t * const localmat,
  matrix_t * const globalmat,
  rank_info * const rinfo,
  idx_t const nfactors,
  idx_t const mode)
{
  idx_t const m = mode;
  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const * const nbr2globs_inds = rinfo->nbr2globs_inds[m];
  idx_t const * const local2nbr_inds = rinfo->local2nbr_inds[m];
  val_t const * const gmatv = globalmat->vals;

  #pragma omp parallel
  {
    /* first prepare all rows that I own and need to send */
    #pragma omp for
    for(idx_t s=0; s < rinfo->nnbr2globs[m]; ++s) {
      idx_t const row = nbr2globs_inds[s] - mat_start;
      for(idx_t f=0; f < nfactors; ++f) {
        nbr2globs_buf[f+(s*nfactors)] = gmatv[f+(row*nfactors)];
      }
    }

    #pragma omp master
    {
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
      MPI_Alltoallv(nbr2globs_buf, nbr2globs_ptr, nbr2globs_disp, SPLATT_MPI_VAL,
                    nbr2local_buf, nbr2local_ptr, nbr2local_disp, SPLATT_MPI_VAL,
                    rinfo->layer_comm[m]);
      timer_stop(&timers[TIMER_MPI_COMM]);
    }

    /* wait for communication to complete */
    #pragma omp barrier

    /* now write incoming nbr2locals to my local matrix */
    val_t * const matv = localmat->vals;
    #pragma omp for
    for(idx_t r=0; r < rinfo->nlocal2nbr[m]; ++r) {
      idx_t const row = local2nbr_inds[r];
      for(idx_t f=0; f < nfactors; ++f) {
        matv[f+(row*nfactors)] = nbr2local_buf[f+(r*nfactors)];
      }
    }
  } /* end omp parallel */
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mpi_update_rows(
  idx_t const * const indmap,
  val_t * const nbr2globs_buf,
  val_t * const nbr2local_buf,
  matrix_t * const localmat,
  matrix_t * const globalmat,
  rank_info * const rinfo,
  idx_t const nfactors,
  idx_t const mode,
  splatt_comm_type const which)
{
  timer_start(&timers[TIMER_MPI_UPDATE]);

  switch(which) {
  case SPLATT_POINT2POINT:
    __update_rows_point2point(nbr2globs_buf, nbr2local_buf, localmat,
        globalmat, rinfo, nfactors, mode);
    break;

  case SPLATT_ALL2ALL:
    __update_rows_all2all(nbr2globs_buf, nbr2local_buf, localmat, globalmat,
        rinfo, nfactors, mode);
    break;

  case SPLATT_SPARSEREDUCE:
    break;
  }

  /* ensure the local matrix is up to date too */
  __flush_glob_to_local(indmap, localmat, globalmat, rinfo, nfactors, mode);
  timer_stop(&timers[TIMER_MPI_UPDATE]);
}


void mpi_reduce_rows(
  val_t * const restrict local2nbr_buf,
  val_t * const restrict nbr2globs_buf,
  matrix_t const * const localmat,
  matrix_t * const globalmat,
  rank_info * const rinfo,
  idx_t const nfactors,
  idx_t const mode,
  splatt_comm_type const which)
{
  timer_start(&timers[TIMER_MPI_REDUCE]);

  switch(which) {
  case SPLATT_POINT2POINT:
    __reduce_rows_point2point(local2nbr_buf, nbr2globs_buf, localmat,
        globalmat, rinfo, nfactors, mode);
    break;

  case SPLATT_ALL2ALL:
    __reduce_rows_all2all(local2nbr_buf, nbr2globs_buf, localmat, globalmat,
        rinfo, nfactors, mode);
    break;

  case SPLATT_SPARSEREDUCE:
    break;
  }
  timer_stop(&timers[TIMER_MPI_REDUCE]);
}


void mpi_add_my_partials(
  idx_t const * const indmap,
  matrix_t const * const localmat,
  matrix_t * const globmat,
  rank_info const * const rinfo,
  idx_t const nfactors,
  idx_t const mode)
{
  timer_start(&timers[TIMER_MPI_PARTIALS]);
  idx_t const m = mode;

  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const mat_end = rinfo->mat_end[m];
  idx_t const start = rinfo->ownstart[m];
  idx_t const nowned = rinfo->nowned[m];

  memset(globmat->vals, 0, globmat->I * nfactors * sizeof(val_t));

  idx_t const goffset = (indmap == NULL) ?
      start - mat_start : indmap[start] - mat_start;

  memcpy(globmat->vals + (goffset * nfactors),
         localmat->vals + (start * nfactors),
         nowned * nfactors * sizeof(val_t));
  timer_stop(&timers[TIMER_MPI_PARTIALS]);
}


void mpi_time_stats(
  rank_info const * const rinfo)
{
  double max_mttkrp, avg_mttkrp;
  double max_mpi, avg_mpi;
  double max_idle, avg_idle;
  double max_com, avg_com;

  timers[TIMER_MPI].seconds =
      timers[TIMER_MPI_ATA].seconds
      + timers[TIMER_MPI_REDUCE].seconds
      + timers[TIMER_MPI_PARTIALS].seconds
      + timers[TIMER_MPI_NORM].seconds
      + timers[TIMER_MPI_UPDATE].seconds
      + timers[TIMER_MPI_FIT].seconds;

  /* get avg times */
  MPI_Reduce(&timers[TIMER_MTTKRP].seconds, &avg_mttkrp, 1, MPI_DOUBLE,
      MPI_SUM, 0, rinfo->comm_3d);
  MPI_Reduce(&timers[TIMER_MPI].seconds, &avg_mpi, 1, MPI_DOUBLE,
      MPI_SUM, 0, rinfo->comm_3d);
  MPI_Reduce(&timers[TIMER_MPI_IDLE].seconds, &avg_idle, 1, MPI_DOUBLE,
      MPI_SUM, 0, rinfo->comm_3d);
  MPI_Reduce(&timers[TIMER_MPI_COMM].seconds, &avg_com, 1, MPI_DOUBLE,
      MPI_SUM, 0, rinfo->comm_3d);

  /* get max times */
  MPI_Reduce(&timers[TIMER_MTTKRP].seconds, &max_mttkrp, 1, MPI_DOUBLE,
      MPI_MAX, 0, rinfo->comm_3d);
  MPI_Reduce(&timers[TIMER_MPI].seconds, &max_mpi, 1, MPI_DOUBLE,
      MPI_MAX, 0, rinfo->comm_3d);
  MPI_Reduce(&timers[TIMER_MPI_IDLE].seconds, &max_idle, 1, MPI_DOUBLE,
      MPI_MAX, 0, rinfo->comm_3d);
  MPI_Reduce(&timers[TIMER_MPI_COMM].seconds, &max_com, 1, MPI_DOUBLE,
      MPI_MAX, 0, rinfo->comm_3d);

  /* set avg times */
  timers[TIMER_MTTKRP].seconds   = avg_mttkrp / rinfo->npes;
  timers[TIMER_MPI].seconds      = avg_mpi    / rinfo->npes;
  timers[TIMER_MPI_IDLE].seconds = avg_idle   / rinfo->npes;
  timers[TIMER_MPI_COMM].seconds = avg_com    / rinfo->npes;
  /* set max times */
  timers[TIMER_MTTKRP_MAX].seconds   = max_mttkrp;
  timers[TIMER_MPI_MAX].seconds      = max_mpi;
  timers[TIMER_MPI_IDLE_MAX].seconds = max_idle;
  timers[TIMER_MPI_COMM_MAX].seconds = max_com;
}

