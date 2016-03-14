
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../splatt_mpi.h"
#include "../kruskal.h"
#include "../mttkrp.h"
#include "../timer.h"
#include "../thd_info.h"
#include "../tile.h"
#include "../util.h"

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
static void p_flush_glob_to_local(
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

  par_memcpy(localmat->vals + (start*nfactors),
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
static void p_reduce_rows_all2all(
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

/**
* @brief Do a reduction (sum) of all neighbor partial products which I own.
*        Updates are written to globalmat.
*        This version accomplishes the communication with an MPI_{Irecv,Isend}.
*
* @param local2nbr_buf A buffer at least as large as nlocal2nbr.
* @param nbr2globs_buf A buffer at least as large as nnbr2globs.
* @param localmat My local matrix containing partial products for other ranks.
* @param globalmat The global factor matrix to update.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the matrices.
* @param m The mode to operate on.
*/
static void p_reduce_rows_point2point(
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

  /* IRECVS */
  for(int p=1; p < lsize; ++p) {
    int const porig = (p + lrank) % lsize;
    /* The number of rows to recv from porig */
    int const nrecvs = rinfo->nbr2globs_ptr[m][porig] / nfactors;
    int const disp  = rinfo->nbr2globs_disp[m][porig] / nfactors;
    if(nrecvs == 0) {
      continue;
    }

    /* do the actual communication */
    timer_start(&timers[TIMER_MPI_COMM]);
    MPI_Irecv(&(nbr2globs_buf[disp*nfactors]), nrecvs*nfactors, SPLATT_MPI_VAL,
        porig, 0, rinfo->layer_comm[m], rinfo->recv_reqs + porig);
    timer_stop(&timers[TIMER_MPI_COMM]);
  }


  #pragma omp parallel default(shared)
  {
    /* ISENDS */
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
            pdest, 0, rinfo->layer_comm[m], rinfo->send_reqs + pdest);
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

      /* Wait for receive to complete */
      #pragma omp master
      {
        timer_start(&timers[TIMER_MPI_COMM]);
        MPI_Wait(rinfo->recv_reqs + porig, MPI_STATUS_IGNORE);
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
static void p_update_rows_point2point(
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


  /* IRECVS */
  for(int p=1; p < lsize; ++p) {
    int const porig = (p + lrank) % lsize;
    /* The number of rows to recv from porig */
    int const nrecvs = rinfo->local2nbr_ptr[m][porig] / nfactors;
    int const disp = rinfo->local2nbr_disp[m][porig] / nfactors;

    if(nrecvs == 0) {
      continue;
    }

    /* do the actual communication */
    timer_start(&timers[TIMER_MPI_COMM]);
    MPI_Irecv(&(nbr2local_buf[disp*nfactors]), nrecvs*nfactors, SPLATT_MPI_VAL,
        porig, 0, rinfo->layer_comm[m], rinfo->recv_reqs + porig);
    timer_stop(&timers[TIMER_MPI_COMM]);
  }


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

      /* wait for the actual communication */
      #pragma omp master
      {
        timer_start(&timers[TIMER_MPI_COMM]);
        MPI_Wait(rinfo->recv_reqs + porig, MPI_STATUS_IGNORE);
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
static void p_update_rows_all2all(
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

double mpi_cpd_als_iterate(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  matrix_t ** globmats,
  val_t * const lambda,
  idx_t const nfactors,
  rank_info * const rinfo,
  double const * const opts)
{
  idx_t const nmodes = tensors[0].nmodes;
  idx_t const nthreads = (idx_t) opts[SPLATT_OPTION_NTHREADS];

  /* Setup thread structures. + 64 bytes is to avoid false sharing. */
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 3,
    (nfactors * nfactors * sizeof(val_t)) + 64,
    (TILE_SIZES[0] * nfactors * sizeof(val_t)) + 64,
    (nmodes * nfactors * sizeof(val_t)) + 64);

  matrix_t * m1 = mats[MAX_NMODES];

  /* Extract MPI communication structures */
  idx_t maxdim = 0;
  idx_t maxlocal2nbr = 0;
  idx_t maxnbr2globs = 0;
  for(idx_t m=0; m < nmodes; ++m) {
    maxlocal2nbr = SS_MAX(maxlocal2nbr, rinfo->nlocal2nbr[m]);
    maxnbr2globs = SS_MAX(maxnbr2globs, rinfo->nnbr2globs[m]);
    maxdim = SS_MAX(globmats[m]->I, maxdim);
  }
  maxlocal2nbr *= nfactors;
  maxnbr2globs *= nfactors;

  val_t * local2nbr_buf = (val_t *) splatt_malloc(maxlocal2nbr * sizeof(val_t));
  val_t * nbr2globs_buf = (val_t *) splatt_malloc(maxnbr2globs * sizeof(val_t));
  if(rinfo->decomp != SPLATT_DECOMP_COARSE) {
    m1 = mat_alloc(maxdim, nfactors);
  }

  /* Exchange initial matrices */
  for(idx_t m=1; m < nmodes; ++m) {
    mpi_update_rows(rinfo->indmap[m], nbr2globs_buf, local2nbr_buf, mats[m],
        globmats[m], rinfo, nfactors, m, opts[SPLATT_OPTION_COMM]);
  }

  matrix_t * m1ptr = m1; /* for restoring m1 */

  /* Initialize first A^T * A mats. We redundantly do the first because it
   * makes communication easier. */
  matrix_t * aTa[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    aTa[m] = mat_alloc(nfactors, nfactors);
    mat_aTa(globmats[m], aTa[m], rinfo, thds, nthreads);
  }
  /* used as buffer space */
  aTa[MAX_NMODES] = mat_alloc(nfactors, nfactors);

  /* Compute input tensor norm */
  double oldfit = 0;
  double fit = 0;
  val_t mynorm = csf_frobsq(tensors);

  val_t ttnormsq = 0;
  MPI_Allreduce(&mynorm, &ttnormsq, 1, SPLATT_MPI_VAL, MPI_SUM, rinfo->comm_3d);

  /* setup timers */
  reset_cpd_timers(rinfo);
  MPI_Barrier(rinfo->comm_3d);
  sp_timer_t itertime;
  sp_timer_t modetime[MAX_NMODES];
  timer_start(&timers[TIMER_CPD]);

  idx_t const niters = (idx_t) opts[SPLATT_OPTION_NITER];
  for(idx_t it=0; it < niters; ++it) {
    timer_fstart(&itertime);
    for(idx_t m=0; m < nmodes; ++m) {
      timer_fstart(&modetime[m]);
      mats[MAX_NMODES]->I = tensors[0].dims[m];

      /* M1 = X * (C o B) */
      timer_start(&timers[TIMER_MTTKRP]);
      mttkrp_csf(tensors, mats, m, thds, opts);
      timer_stop(&timers[TIMER_MTTKRP]);

      m1->I = globmats[m]->I;
      m1ptr->I = globmats[m]->I;

      if(rinfo->decomp != SPLATT_DECOMP_COARSE && rinfo->layer_size[m] > 1) {
        m1 = m1ptr;
        /* add my partial multiplications to globmats[m] */
        mpi_add_my_partials(rinfo->indmap[m], mats[MAX_NMODES], m1, rinfo,
            nfactors, m);
        /* incorporate neighbors' partials */
        mpi_reduce_rows(local2nbr_buf, nbr2globs_buf, mats[MAX_NMODES], m1,
            rinfo, nfactors, m, opts[SPLATT_OPTION_COMM]);
      } else {
        /* skip the whole process */
        m1 = mats[MAX_NMODES];
      }

      /* invert normal equations (Cholesky factorization) for new factor */
      par_memcpy(globmats[m]->vals, m1->vals, m1->I * nfactors * sizeof(val_t));
      mat_solve_normals(m, nmodes, aTa, globmats[m], 0.);

      /* normalize columns and extract lambda */
      if(it == 0) {
        mat_normalize(globmats[m], lambda, MAT_NORM_2, rinfo, thds, nthreads);
      } else {
        mat_normalize(globmats[m], lambda, MAT_NORM_MAX, rinfo, thds,nthreads);
      }

      /* send updated rows to neighbors */
      mpi_update_rows(rinfo->indmap[m], nbr2globs_buf, local2nbr_buf, mats[m],
          globmats[m], rinfo, nfactors, m, opts[SPLATT_OPTION_COMM]);

      /* update A^T*A */
      mat_aTa(globmats[m], aTa[m], rinfo, thds, nthreads);
      timer_stop(&modetime[m]);
    } /* foreach mode */

    fit = kruskal_calc_fit(nmodes, rinfo, thds, ttnormsq, lambda, globmats,
        m1, aTa);
    timer_stop(&itertime);

    if(rinfo->rank == 0 &&
        opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_NONE) {
      printf("  its = %3"SPLATT_PF_IDX" (%0.3fs)  fit = %0.5f  delta = %+0.4e\n",
          it+1, itertime.seconds, fit, fit - oldfit);
      if(opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_LOW) {
        for(idx_t m=0; m < nmodes; ++m) {
          printf("     mode = %1"SPLATT_PF_IDX" (%0.3fs)\n", m+1,
              modetime[m].seconds);
        }
      }
    }
    if(it > 0 && fabs(fit - oldfit) < opts[SPLATT_OPTION_TOLERANCE]) {
      break;
    }
    oldfit = fit;
  }
  timer_stop(&timers[TIMER_CPD]);

  if(rinfo->rank == 0 &&
      opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_NONE) {
    printf("Final fit: %0.5f\n", fit);
  }

  /* POST PROCESSING */
  /* normalize each mat and adjust lambda */
  val_t * tmp = (val_t *) splatt_malloc(nfactors * sizeof(val_t));
  for(idx_t m=0; m < nmodes; ++m) {
    mat_normalize(globmats[m], tmp, MAT_NORM_2, rinfo, thds, nthreads);
    for(idx_t f=0; f < nfactors; ++f) {
      lambda[f] *= tmp[f];
    }
  }
  free(tmp);

  /* CLEAN UP */
  for(idx_t m=0; m < nmodes; ++m) {
    mat_free(aTa[m]);
  }
  mat_free(aTa[MAX_NMODES]);

  thd_free(thds, nthreads);
  if(rinfo->decomp != SPLATT_DECOMP_COARSE) {
    mat_free(m1ptr);
  }
  free(local2nbr_buf);
  free(nbr2globs_buf);

  mpi_time_stats(rinfo);

  return fit;
}


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
  case SPLATT_COMM_POINT2POINT:
    p_update_rows_point2point(nbr2globs_buf, nbr2local_buf, localmat,
        globalmat, rinfo, nfactors, mode);
    break;

  case SPLATT_COMM_ALL2ALL:
    p_update_rows_all2all(nbr2globs_buf, nbr2local_buf, localmat, globalmat,
        rinfo, nfactors, mode);
    break;
  }

  /* ensure the local matrix is up to date too */
  p_flush_glob_to_local(indmap, localmat, globalmat, rinfo, nfactors, mode);
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
  case SPLATT_COMM_POINT2POINT:
    p_reduce_rows_point2point(local2nbr_buf, nbr2globs_buf, localmat,
        globalmat, rinfo, nfactors, mode);
    break;

  case SPLATT_COMM_ALL2ALL:
    p_reduce_rows_all2all(local2nbr_buf, nbr2globs_buf, localmat, globalmat,
        rinfo, nfactors, mode);
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

  par_memcpy(globmat->vals + (goffset * nfactors),
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

