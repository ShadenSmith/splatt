
/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "completion.h"
#include "../csf.h"
#include "../tile.h"
#include "../util.h"

#include <math.h>
#include <omp.h>



/* It is faster to continually add/subtract to residual instead of recomputing
 * predictions. In practice this is fine, but set to 1 if we want to see. */
#ifndef MEASURE_DRIFT
#define MEASURE_DRIFT 0
#endif

/* Use hardcoded 3-mode kernels when possible. Results in small speedups. */
#ifndef USE_3MODE_OPT
#define USE_3MODE_OPT 1
#endif


#define GRAB_SPARSITY(tile_id) \
  csf_sparsity * const pt = csf->pt + (tile_id);\
  idx_t const * const restrict sptr = pt->fptr[0];\
  idx_t const * const restrict fptr = pt->fptr[1];\
  idx_t const * const restrict fids = pt->fids[1];\
  idx_t const * const restrict inds = pt->fids[2];\
  val_t * const restrict residual = pt->vals;\

#define GRAB_CONST_FACTORS \
  idx_t const I = model->dims[csf->dim_perm[0]];\
  idx_t const J = model->dims[csf->dim_perm[1]];\
  idx_t const K = model->dims[csf->dim_perm[2]];\
  val_t const * const restrict avals = model->factors[csf->dim_perm[0]]+(f*I);\
  val_t const * const restrict bvals = model->factors[csf->dim_perm[1]]+(f*J);\
  val_t const * const restrict cvals = model->factors[csf->dim_perm[2]]+(f*K);\




/******************************************************************************
 * MPI FUNCTIONS
 *
 * CCD++ is a column-major method, so the CPD communication will not work here.
 *****************************************************************************/

#ifdef SPLATT_USE_MPI


static void p_reduce_partials_all2all(
    tc_model * const model,
    tc_ws * const ws,
    idx_t const mode,
    idx_t const col)
{
  rank_info * const rinfo = ws->rinfo;
  idx_t const m = mode;

  if(ws->rinfo->layer_size[m] == 1) {
    return;
  }

  val_t * const restrict numer = ws->numerator;
  val_t * const restrict denom = ws->denominator;

  /* Just do an MPI_Allreduce on dense modes */
  if(ws->isdense[m]) {
    #pragma omp master
    {
      MPI_Request req_numer;
      MPI_Request req_denom;

      timer_start(&timers[TIMER_MPI_COMM]);
      MPI_Iallreduce(MPI_IN_PLACE, numer, model->dims[m], SPLATT_MPI_VAL,
          MPI_SUM, rinfo->layer_comm[m], &req_numer);
      MPI_Iallreduce(MPI_IN_PLACE, denom, model->dims[m], SPLATT_MPI_VAL,
          MPI_SUM, rinfo->layer_comm[m], &req_denom);

      MPI_Wait(&req_numer, MPI_STATUS_IGNORE);
      MPI_Wait(&req_denom, MPI_STATUS_IGNORE);

      timer_stop(&timers[TIMER_MPI_COMM]);
    }
    #pragma omp barrier
    return;
  }

  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const * const restrict local2nbr_inds = rinfo->local2nbr_inds[m];
  idx_t const * const restrict nbr2globs_inds = rinfo->nbr2globs_inds[m];

  /* grab buffers */
  idx_t const nsends = rinfo->nlocal2nbr[m] * 2;
  val_t * const restrict local2nbr_buf  = ws->local2nbr_buf;
  val_t * const restrict nbr2globs_buf  = ws->nbr2globs_buf;

  /* grab ptr/disp from rinfo. nbr2local and local2nbr will have the same
   * structure so we just reuse those */
  int * const restrict nbr2globs_ptr = rinfo->nbr2globs_ptr[m];
  int * const restrict nbr2local_ptr = rinfo->local2nbr_ptr[m];
  int * const restrict nbr2globs_disp = rinfo->nbr2globs_disp[m];
  int * const restrict nbr2local_disp = rinfo->local2nbr_disp[m];

  /* fill outgoing buffer */
  #pragma omp for
  for(idx_t s=0; s < nsends; s += 2) {
    idx_t const row = local2nbr_inds[s/2];
    local2nbr_buf[s]    = numer[row];
    local2nbr_buf[s+1]  = denom[row];
  }

  int const lrank = rinfo->layer_rank[m];
  int const lsize = rinfo->layer_size[m];

  /* temporarily increase comvol by 2 */
  #pragma omp for
  for(int p=0; p < lsize; ++p) {
    nbr2local_ptr[p]  *= 2;
    nbr2local_disp[p] *= 2;
    nbr2globs_ptr[p]  *= 2;
    nbr2globs_disp[p] *= 2;
  }

  /* exchange numerator and denominator */
  #pragma omp master
  {
    timer_start(&timers[TIMER_MPI_COMM]);
    MPI_Alltoallv(local2nbr_buf, nbr2local_ptr, nbr2local_disp, SPLATT_MPI_VAL,
                  nbr2globs_buf, nbr2globs_ptr, nbr2globs_disp, SPLATT_MPI_VAL,
                  rinfo->layer_comm[m]);
    timer_stop(&timers[TIMER_MPI_COMM]);
  }
  #pragma omp barrier

  /* Now add received partial products. We can parallelize the additions from
   * each process. */
  for(int p=1; p < lsize; ++p) {
    int const porig = (p + lrank) % lsize;
    /* The number of rows to recv from porig */
    int const nrecvs = nbr2globs_ptr[porig];
    int const disp   = nbr2globs_disp[porig];

    #pragma omp for
    for(int r=disp; r < disp + nrecvs; r += 2) {
      idx_t const row = nbr2globs_inds[r/2];
      numer[row] += nbr2globs_buf[r];
      denom[row] += nbr2globs_buf[r+1];
    }
  } /* end recvs */

  /* fix comvol */
  #pragma omp for
  for(int p=0; p < lsize; ++p) {
    nbr2local_ptr[p]  /= 2;
    nbr2local_disp[p] /= 2;
    nbr2globs_ptr[p]  /= 2;
    nbr2globs_disp[p] /= 2;
  }
}




static void p_update_col_all2all(
    tc_model * const model,
    tc_ws * const ws,
    idx_t const mode,
    idx_t const col)
{
  rank_info * const rinfo = ws->rinfo;
  idx_t const m = mode;

  idx_t const nglobrows = model->globmats[m]->I;
  val_t const * const restrict gmatv = model->globmats[m]->vals +
      (col * nglobrows);

  /* ensure local info is up to date */
  assert(rinfo->ownstart[m] + rinfo->nowned[m] <= model->dims[m]);
  val_t * const restrict matv = model->factors[m] + (col * model->dims[m]);
  par_memcpy(matv + rinfo->ownstart[m], gmatv, rinfo->nowned[m]*sizeof(*matv));

  if(rinfo->layer_size[mode] == 1) {
    return;
  }

  /* first prepare all values that I own and need to send */
  idx_t const mat_start = rinfo->mat_start[m];
  idx_t const * const restrict nbr2globs_inds = rinfo->nbr2globs_inds[m];
  idx_t const * const restrict local2nbr_inds = rinfo->local2nbr_inds[m];

  idx_t const nsends = rinfo->nnbr2globs[m];
  idx_t const nrecvs = rinfo->nlocal2nbr[m];

  val_t * const restrict nbr2globs_buf = ws->nbr2globs_buf;
  val_t * const restrict nbr2local_buf = ws->local2nbr_buf;

  /* foreach send */
  #pragma omp for
  for(idx_t s=0; s < nsends; ++s) {
    idx_t const row = nbr2globs_inds[s] - mat_start;
    nbr2globs_buf[s] = gmatv[row];
  }

  /* grab ptr/disp from rinfo. nbr2local and local2nbr will have the same
   * structure so we just reuse those */
  int const * const restrict nbr2globs_ptr = rinfo->nbr2globs_ptr[m];
  int const * const restrict nbr2local_ptr = rinfo->local2nbr_ptr[m];
  int const * const restrict nbr2globs_disp = rinfo->nbr2globs_disp[m];
  int const * const restrict nbr2local_disp = rinfo->local2nbr_disp[m];

  /* exchange entries */
  #pragma omp master
  {
    timer_start(&timers[TIMER_MPI_COMM]);
    MPI_Alltoallv(nbr2globs_buf, nbr2globs_ptr, nbr2globs_disp, SPLATT_MPI_VAL,
                  nbr2local_buf, nbr2local_ptr, nbr2local_disp, SPLATT_MPI_VAL,
                  rinfo->layer_comm[m]);
    timer_stop(&timers[TIMER_MPI_COMM]);
  }
  #pragma omp barrier

  /* now write incoming values to my local matrix */
  #pragma omp for
  for(idx_t r=0; r < nrecvs; ++r) {
    idx_t const row = local2nbr_inds[r];
    matv[row] = nbr2local_buf[r];
  }
}


static void p_init_mpi(
    sptensor_t const * const train,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t maxlocal2nbr = 0;
  idx_t maxnbr2globs = 0;

  /* recompute this stuff with nfactors = 1 due to column-major layout */
  for(idx_t m=0; m < train->nmodes; ++m) {
    mpi_find_owned(train, m, ws->rinfo);
    mpi_compute_ineed(ws->rinfo, train, m, 1, 3);

    maxlocal2nbr = SS_MAX(maxlocal2nbr, ws->rinfo->nlocal2nbr[m]);
    maxnbr2globs = SS_MAX(maxnbr2globs, ws->rinfo->nnbr2globs[m]);
  }

  ws->local2nbr_buf  = splatt_malloc(2*maxlocal2nbr * sizeof(val_t));
  ws->nbr2globs_buf  = splatt_malloc(2*maxnbr2globs * sizeof(val_t));

  /* get initial factors */
  #pragma omp parallel
  for(idx_t m=0; m < train->nmodes; ++m) {
    for(idx_t f=0; f < model->rank; ++f) {
      p_update_col_all2all(model, ws, m, f);
    }
  }
}



#endif






/******************************************************************************
 * TYPES
 *****************************************************************************/

typedef enum
{
  NODE_ROOT,
  NODE_INTL,
  NODE_LEAF
} node_type;



/**
* @brief Determine what time of node mode 'm' is.
*
*        TODO: Accept 'opts' as a parameter and integrate into MTTKRP and other
*              computations.
*
* @param csf The CSF tensor.
* @param m The mode.
*
* @return Root, intl, or leaf.
*/
static inline node_type which_depth(
    splatt_csf const * const csf,
    idx_t const m)
{
  node_type which;
  idx_t const depth = csf_mode_depth(m, csf->dim_perm, csf->nmodes);
  if(depth == 0) {
    which = NODE_ROOT;
  } else if(depth == csf->nmodes - 1) {
    which = NODE_LEAF;
  } else {
    which = NODE_INTL;
  }

  return which;
}



/******************************************************************************
 * UPDATING FUNCTIONS - for updating residual and factors
 *****************************************************************************/


static val_t p_update_residual3(
    splatt_csf const * const csf,
    idx_t const f,
    tc_model const * const model,
    tc_ws * const ws,
    val_t const mult)
{
  #pragma omp master
  timer_start(&ws->resid_time);

  idx_t const nfactors = model->rank;

  idx_t const I = model->dims[csf->dim_perm[0]];
  idx_t const J = model->dims[csf->dim_perm[1]];
  idx_t const K = model->dims[csf->dim_perm[2]];
  val_t const * const restrict avals = model->factors[csf->dim_perm[0]]+(f*I);
  val_t const * const restrict bvals = model->factors[csf->dim_perm[1]]+(f*J);
  val_t const * const restrict cvals = model->factors[csf->dim_perm[2]]+(f*K);

  val_t myloss = 0;

  /* update residual in parallel */
  #pragma omp for schedule(dynamic, 1)
  for(idx_t tile=0; tile < csf->ntiles; ++tile) {
    GRAB_SPARSITY(tile)

    for(idx_t i=0; i < pt->nfibs[0]; ++i) {
      idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
      val_t const aval = avals[a_id];

      for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
        val_t const bval = bvals[fids[fib]];
        for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj){
          val_t const cval = cvals[inds[jj]];

          residual[jj] += mult * aval * bval * cval;
          myloss += residual[jj] * residual[jj];
        }
      } /* foreach fiber */
    } /* foreach slice */
  } /* foreach tile */

  #pragma omp master
  timer_stop(&ws->resid_time);
  return myloss;
}


static val_t p_update_residual(
    splatt_csf const * const csf,
    idx_t const f,
    tc_model const * const model,
    tc_ws * const ws,
    val_t const mult)
{
  idx_t const nmodes = csf->nmodes;
#if USE_3MODE_OPT
  if(nmodes == 3) {
    return p_update_residual3(csf, f, model, ws, mult);
  }
#endif

  #pragma omp master
  timer_start(&ws->resid_time);

  /* grab factors */
  val_t const * restrict mats[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (val_t const * restrict) model->factors[csf->dim_perm[m]] +
        (f * model->dims[csf->dim_perm[m]]);
  }
  val_t const * const restrict lastmat = mats[nmodes-1];

  val_t myloss = 0;

  #pragma omp for schedule(dynamic, 1)
  for(idx_t tile=0; tile < csf->ntiles; ++tile) {

    /* grab sparsity structure */
    csf_sparsity const * const pt = csf->pt + tile;
    val_t * const restrict residual = pt->vals;
    idx_t const * const restrict * fp = (idx_t const * const *) pt->fptr;
    idx_t const * const restrict * fids = (idx_t const * const *) pt->fids;
    idx_t const * const restrict inds = fids[nmodes-1];

    idx_t idxstack[MAX_NMODES];
    val_t predictbuf[MAX_NMODES];

    /* foreach outer slice */
    for(idx_t i=0; i < pt->nfibs[0]; ++i) {
      idx_t out_id = (fids[0] == NULL) ? i : fids[0][i];

      /* initialize first prediction portion */
      predictbuf[0] = mats[0][out_id];

      /* clear out stale data */
      idxstack[0] = i;
      for(idx_t m=1; m < nmodes-1; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* process each subtree */
      idx_t depth = 0;
      while(idxstack[1] < fp[0][i+1]) {
        /* move down to nnz node while computing predicted value */
        for(; depth < nmodes-2; ++depth) {
          predictbuf[depth+1] = predictbuf[depth] *
              mats[depth+1][fids[depth+1][idxstack[depth+1]]];
        }

        /* process all nonzeros [start, end) */
        idx_t const start = fp[depth][idxstack[depth]];
        idx_t const end   = fp[depth][idxstack[depth]+1];
        for(idx_t jj=start; jj < end; ++jj) {
          /* computed predicted value and update residual */
          val_t const p = predictbuf[depth] * lastmat[inds[jj]];
          residual[jj] += mult * p;
          myloss += residual[jj] * residual[jj];
        }

        /* now move up to the next unprocessed subtree */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* foreach fiber subtree */
    } /* foreach outer slice */
  } /* foreach tile */

  #pragma omp master
  timer_stop(&ws->resid_time);

  return myloss;
}



/******************************************************************************
 * PROCESSING FUNCTIONS - for computing numerator/denominator
 *****************************************************************************/
static void p_process_root3(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const f,
    tc_model const * const model,
    tc_ws * const ws,
    val_t * const restrict numer,
    val_t * const restrict denom)
{
  idx_t const nfactors = model->rank;

  GRAB_SPARSITY(tile)
  /* empty tile, just return */
  if(residual == NULL) {
    return;
  }

  GRAB_CONST_FACTORS

  for(idx_t i=0; i < pt->nfibs[0]; ++i) {
    idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];

    /* process each fiber */
    for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
      val_t const bval = bvals[fids[fib]];

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj) {
        val_t const cval = cvals[inds[jj]];

        val_t const sgrad = bval * cval;
        numer[a_id] += residual[jj] * sgrad;
        denom[a_id] += sgrad * sgrad;
      }
    } /* foreach fiber */
  } /* foreach slice */
}


static void p_process_root(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const f,
    tc_model const * const model,
    tc_ws * const ws,
    val_t * const restrict numer,
    val_t * const restrict denom)
{
  idx_t const nmodes = csf->nmodes;
#if USE_3MODE_OPT
  if(nmodes == 3) {
    p_process_root3(csf, tile, f, model, ws, numer, denom);
    return;
  }
#endif

  /* grab sparsity structure */
  csf_sparsity const * const pt = csf->pt + tile;
  val_t const * const restrict residual = pt->vals;
  if(residual == NULL) {
    return;
  }
  idx_t const * const * const restrict fp = (idx_t const * const *) pt->fptr;
  idx_t const * const * const restrict fids = (idx_t const * const *) pt->fids;
  idx_t const * const restrict inds = fids[nmodes-1];

  idx_t idxstack[MAX_NMODES];
  val_t predictbuf[MAX_NMODES];

  /* grab factors */
  val_t const * restrict mats[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (val_t const * const restrict) model->factors[csf->dim_perm[m]] +
        (f * model->dims[csf->dim_perm[m]]);
  }
  val_t const * const restrict lastmat = mats[nmodes-1];

  /* foreach outer slice */
  for(idx_t i=0; i < pt->nfibs[0]; ++i) {
    idx_t const out_id = (fids[0] == NULL) ? i : fids[0][i];

    predictbuf[0] = 1.;

    /* clear out stale data */
    idxstack[0] = i;
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* process each subtree */
    idx_t depth = 0;
    while(idxstack[1] < fp[0][i+1]) {
      /* move down to nnz node while computing predicted value */
      for(; depth < nmodes-2; ++depth) {
        predictbuf[depth+1] = predictbuf[depth] *
            mats[depth+1][fids[depth+1][idxstack[depth+1]]];
      }

      /* process all nonzeros [start, end) */
      idx_t const start = fp[depth][idxstack[depth]];
      idx_t const end   = fp[depth][idxstack[depth]+1];
      for(idx_t jj=start; jj < end; ++jj) {
        val_t const lastval = lastmat[inds[jj]];
        val_t const sgrad = predictbuf[depth] * lastval;
        numer[out_id] += residual[jj] * sgrad;
        denom[out_id] += sgrad * sgrad;
      }

      /* now move up to the next unprocessed subtree */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* foreach fiber subtree */
  } /* foreach outer slice */
}



static void p_process_intl3(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const f,
    tc_model const * const model,
    tc_ws * const ws,
    val_t * const restrict numer,
    val_t * const restrict denom)
{
  idx_t const nfactors = model->rank;

  GRAB_SPARSITY(tile)
  if(residual == NULL) {
    return;
  }
  GRAB_CONST_FACTORS

  for(idx_t i=0; i < pt->nfibs[0]; ++i) {
    idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];

    /* grab the top-level row to update */
    val_t const aval = avals[a_id];

    /* process each fiber */
    for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
      idx_t const b_id = fids[fib];
      val_t const bval = bvals[b_id];

      val_t const predict = aval * bval;

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj) {
        val_t const cval = cvals[inds[jj]];

        val_t const sgrad = aval * cval;
        numer[b_id] += residual[jj] * sgrad;
        denom[b_id] += sgrad * sgrad;
      }
    } /* foreach fiber */
  } /* foreach slice */
}


static void p_process_intl(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const f,
    idx_t const outdepth,
    tc_model const * const model,
    tc_ws * const ws,
    val_t * const restrict numer,
    val_t * const restrict denom)
{
  idx_t const nmodes = csf->nmodes;
#if USE_3MODE_OPT
  if(nmodes == 3) {
    p_process_intl3(csf, tile, f, model, ws, numer, denom);
    return;
  }
#endif

  /* grab sparsity structure */
  csf_sparsity const * const pt = csf->pt + tile;
  val_t const * const restrict residual = pt->vals;
  if(residual == NULL) {
    return;
  }
  idx_t const * const * const restrict fp = (idx_t const * const *) pt->fptr;
  idx_t const * const * const restrict fids = (idx_t const * const *) pt->fids;
  idx_t const * const restrict inds = fids[nmodes-1];

  idx_t idxstack[MAX_NMODES];
  val_t predictbuf[MAX_NMODES];

  /* grab factors */
  val_t const * restrict mats[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (val_t const * const restrict) model->factors[csf->dim_perm[m]] +
        (f * model->dims[csf->dim_perm[m]]);
  }
  val_t const * const restrict lastmat = mats[nmodes-1];

  /* foreach outer slice */
  for(idx_t i=0; i < pt->nfibs[0]; ++i) {
    idx_t const top_id = (fids[0] == NULL) ? i : fids[0][i];

    predictbuf[0] = mats[0][top_id];

    /* clear out stale data */
    idxstack[0] = i;
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* process each subtree */
    idx_t depth = 0;
    while(idxstack[1] < fp[0][i+1]) {
      /* move down to the output level */
      for(; depth < outdepth; ++depth) {
        predictbuf[depth+1] = predictbuf[depth] *
            mats[depth+1][fids[depth+1][idxstack[depth+1]]];
      }

      /* grab output idx and skip predictbuf at this depth */
      idx_t const out_id = fids[outdepth][idxstack[outdepth]];
      predictbuf[outdepth] = predictbuf[outdepth-1];

      /* move down to nnz node while computing predicted value */
      for(; depth < nmodes-2; ++depth) {
        predictbuf[depth+1] = predictbuf[depth] *
            mats[depth+1][fids[depth+1][idxstack[depth+1]]];
      }

      /* process all nonzeros [start, end) */
      idx_t const start = fp[depth][idxstack[depth]];
      idx_t const end   = fp[depth][idxstack[depth]+1];
      for(idx_t jj=start; jj < end; ++jj) {
        val_t const sgrad = predictbuf[depth] * lastmat[inds[jj]];
        numer[out_id] += residual[jj] * sgrad;
        denom[out_id] += sgrad * sgrad;
      }

      /* now move up to the next unprocessed subtree */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* foreach fiber subtree */
  } /* foreach outer slice */

}


static void p_process_leaf3(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const f,
    tc_model const * const model,
    tc_ws * const ws,
    val_t * const restrict numer,
    val_t * const restrict denom)
{
  idx_t const nfactors = model->rank;

  GRAB_SPARSITY(tile)
  if(residual == NULL) {
    return;
  }
  GRAB_CONST_FACTORS

  for(idx_t i=0; i < pt->nfibs[0]; ++i) {
    idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];

    /* grab the top-level row to update */
    val_t const aval = avals[a_id];

    /* process each fiber */
    for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
      val_t const bval = bvals[fids[fib]];

      val_t const predict = aval * bval;

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj) {
        idx_t const c_id = inds[jj];
        val_t const cval = cvals[c_id];

        val_t const sgrad = aval * bval;
        numer[c_id] += residual[jj] * predict;
        denom[c_id] += sgrad * sgrad;
      }
    } /* foreach fiber */
  } /* foreach slice */
}



static void p_process_leaf(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const f,
    tc_model const * const model,
    tc_ws * const ws,
    val_t * const restrict numer,
    val_t * const restrict denom)
{
  idx_t const nmodes = csf->nmodes;
#if USE_3MODE_OPT
  if(nmodes == 3) {
    p_process_leaf3(csf, tile, f, model, ws, numer, denom);
    return;
  }
#endif

  /* grab sparsity structure */
  csf_sparsity const * const pt = csf->pt + tile;
  val_t const * const restrict residual = pt->vals;
  if(residual == NULL) {
    return;
  }
  idx_t const * const * const restrict fp = (idx_t const * const *) pt->fptr;
  idx_t const * const * const restrict fids = (idx_t const * const *) pt->fids;
  idx_t const * const restrict inds = fids[nmodes-1];

  idx_t idxstack[MAX_NMODES];
  val_t predictbuf[MAX_NMODES];

  /* grab factors */
  val_t const * restrict mats[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (val_t const * const restrict) model->factors[csf->dim_perm[m]] +
        (f * model->dims[csf->dim_perm[m]]);
  }

  /* foreach outer slice */
  for(idx_t i=0; i < pt->nfibs[0]; ++i) {
    idx_t const top_id = (fids[0] == NULL) ? i : fids[0][i];

    predictbuf[0] = mats[0][top_id];

    /* clear out stale data */
    idxstack[0] = i;
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* process each subtree */
    idx_t depth = 0;
    while(idxstack[1] < fp[0][i+1]) {
      /* move down to nnz node while computing predicted value */
      for(; depth < nmodes-2; ++depth) {
        predictbuf[depth+1] = predictbuf[depth] *
            mats[depth+1][fids[depth+1][idxstack[depth+1]]];
      }

      /* process all nonzeros [start, end) */
      val_t const sgrad = predictbuf[depth];
      idx_t const start = fp[depth][idxstack[depth]];
      idx_t const end   = fp[depth][idxstack[depth]+1];
      for(idx_t jj=start; jj < end; ++jj) {
        idx_t const out_id = inds[jj];
        numer[out_id] += residual[jj] * sgrad;
        denom[out_id] += sgrad * sgrad;
      }

      /* now move up to the next unprocessed subtree */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* foreach fiber subtree */
  } /* foreach outer slice */
}


/******************************************************************************
 * MODE-UPDATE FUNCTIONS - drivers for updating a full factor column
 *****************************************************************************/


static void p_densemode_ccd_update(
    splatt_csf const * const csf,
    idx_t const m,
    idx_t const f,
    tc_model * const model,
    tc_ws * const ws,
    thd_info * thd_densefactors,
    int const tid)
{
  idx_t const dim = model->dims[m];

  /* save numerator/denominator ptrs */
  #pragma omp master
  {
    timer_start(&ws->dense_time);
    SPLATT_VPTR_SWAP(thd_densefactors[0].scratch[0], ws->numerator);
    SPLATT_VPTR_SWAP(thd_densefactors[0].scratch[1], ws->denominator);
  }
  #pragma omp barrier

  val_t * const restrict numer = thd_densefactors[tid].scratch[0];
  val_t * const restrict denom = thd_densefactors[tid].scratch[1];
  memset(numer, 0, dim * sizeof(*numer));
  memset(denom, 0, dim * sizeof(*denom));

  /* which routine to call? */
  node_type const which = which_depth(csf, m);
  idx_t const depth = csf_mode_depth(m, csf->dim_perm, csf->nmodes);

  /* Compute thread-local numerators/denominators */
  #pragma omp for schedule(dynamic, 1)
  for(idx_t tile=0; tile < csf->ntiles; ++tile) {
    switch(which) {
    case NODE_ROOT:
      p_process_root(csf, tile, f, model, ws, numer, denom);
      break;
    case NODE_INTL:
      p_process_intl(csf, tile, f, depth, model, ws, numer, denom);
      break;
    case NODE_LEAF:
      p_process_leaf(csf, tile, f, model, ws, numer, denom);
      break;
    }
  } /* foreach tile */

  /* aggregate numerator and denominator */
  thd_reduce(thd_densefactors, 0, dim, REDUCE_SUM);
  thd_reduce(thd_densefactors, 1, dim, REDUCE_SUM);

  /* return aggregated numer/denom to ws */
  #pragma omp master
  {
    SPLATT_VPTR_SWAP(thd_densefactors[0].scratch[0], ws->numerator);
    SPLATT_VPTR_SWAP(thd_densefactors[0].scratch[1], ws->denominator);
    timer_stop(&ws->dense_time);
  }
  #pragma omp barrier
}



static void p_sparsemode_ccd_update(
    splatt_csf const * const csf,
    idx_t const m,
    idx_t const f,
    tc_model * const model,
    tc_ws * const ws,
    int const tid)
{
  idx_t const dim = model->dims[m];

  val_t * const restrict numer = ws->numerator;
  val_t * const restrict denom = ws->denominator;

  #pragma omp barrier
  #pragma omp master
  timer_start(&ws->sparse_time);

  /* initialize numerator/denominator */
  #pragma omp for schedule(static)
  for(idx_t i=0; i < dim; ++i) {
    numer[i] = 0;
    denom[i] = 0;
  }

  /* which routine to call? */
  node_type const which = which_depth(csf, m);
  idx_t const depth = csf_mode_depth(m, csf->dim_perm, csf->nmodes);

  /* Compute numerator/denominator. Distribute tile layer to threads
   *  to avoid locks. */
  #pragma omp for schedule(dynamic, 1)
  for(idx_t t=0; t < csf->tile_dims[m]; ++t) {
    idx_t tile = get_next_tileid(TILE_BEGIN, csf->tile_dims, csf->nmodes,
        m, t);
    while(tile != TILE_END) {
      /* process tile */
      switch(which) {
      case NODE_ROOT:
        p_process_root(csf, tile, f, model, ws, numer, denom);
        break;
      case NODE_INTL:
        p_process_intl(csf, tile, f, depth, model, ws, numer, denom);
        break;
      case NODE_LEAF:
        p_process_leaf(csf, tile, f, model, ws, numer, denom);
        break;
      }

      /* move on to text tile in my layer */
      tile = get_next_tileid(tile, csf->tile_dims, csf->nmodes, m, t);
    }
  } /* foreach tile */

  #pragma omp master
  timer_stop(&ws->sparse_time);
}



/**
* @brief Finalize the new f-th column of factors[m] after computing the new
*        numerator/denominator.
*
* @param model The model to update
* @param ws Workspace data.
* @param m The mode to update.
* @param f The column to update.
*/
static inline void p_compute_newcol(
    tc_model * const model,
    tc_ws * const ws,
    idx_t const m,
    idx_t const f)
{
  #pragma omp barrier
  #pragma omp master
  timer_start(&ws->newcol_time);

#ifdef SPLATT_USE_MPI
  idx_t const dim = model->globmats[m]->I;
  val_t * const restrict avals = model->globmats[m]->vals + (f * dim);
  idx_t const offset = ws->rinfo->ownstart[m];
#else
  idx_t const dim = model->dims[m];
  val_t * const restrict avals = model->factors[m] + (f * dim);
  idx_t const offset = 0;
#endif

  val_t const reg = ws->regularization[m];
  val_t const * const restrict numer = ws->numerator + offset;
  val_t const * const restrict denom = ws->denominator + offset;

  #pragma omp for schedule(static)
  for(idx_t i=0; i < dim; ++i) {
    avals[i] = numer[i] / (reg + denom[i]);
  }

  #pragma omp master
  timer_stop(&ws->newcol_time);
}



/**
* @brief Transpose a model's factor matrices.
*
* @param model The model to transpose.
*/
static void p_transpose_model(
    tc_model * model)
{
  idx_t const maxdim = model->dims[argmax_elem(model->dims, model->nmodes)];
  printf("maxdim: %lu\n", maxdim);

  val_t * restrict buf = splatt_malloc(maxdim * model->rank * sizeof(*buf));

  #pragma omp parallel
  for(idx_t m=0; m < model->nmodes; ++m) {
    idx_t const nrows = model->dims[m];
    idx_t const ncols = model->rank;
    val_t * const restrict factor = model->factors[m];

    for(idx_t j=0; j < model->rank; ++j) {
      #pragma omp for schedule(static)
      for(idx_t i=0; i < nrows; ++i) {
        buf[i + (j*nrows)] = factor[j + (i*ncols)];
      }
    }

    par_memcpy(factor, buf, nrows * ncols * sizeof(*factor));
  }

  splatt_free(buf);
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void splatt_tc_ccd(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nmodes = train->nmodes;
  idx_t const nfactors = model->rank;

#ifdef SPLATT_USE_MPI
  int const rank = ws->rinfo->rank;
  p_init_mpi(train, model, ws);
#else
  int const rank = 0;
#endif

  if(rank == 0) {
    printf("INNER ITS: %"SPLATT_PF_IDX"\n", ws->num_inner);
    printf("USING 3MODE OPTS: %d\n", USE_3MODE_OPT);
  }

  /* setup dense modes */
  thd_info * thd_densefactors = NULL;
  if(ws->num_dense > 0) {
    thd_densefactors = thd_init(ws->nthreads, 2,
        ws->maxdense_dim * sizeof(val_t),  /* numerator */
        ws->maxdense_dim * sizeof(val_t)); /* denominator */


    if(rank == 0) {
      printf("REPLICATING MODES:");
      for(idx_t m=0; m < nmodes; ++m) {
        if(ws->isdense[m]) {
          printf(" %"SPLATT_PF_IDX, m+1);
        }
      }
      printf("\n\n");
    }
  }

  p_transpose_model(model);

  /* convert training data to CSF-ONEMODE with full tiling */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS] = ws->nthreads;
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE] = SPLATT_CCPTILE;
  opts[SPLATT_OPTION_TILEDEPTH] = SS_MAX(ws->num_dense, 1);


  splatt_csf * csf = csf_alloc(train, opts);

  /* initialize residual */
  #pragma omp parallel
  {
    for(idx_t f=0; f < nfactors; ++f) {
      p_update_residual(csf, f, model, ws, -1);
    }
  }

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  timer_start(&ws->tc_time);

  timer_reset(&ws->resid_time);
  timer_reset(&ws->dense_time);
  timer_reset(&ws->sparse_time);
  timer_reset(&ws->newcol_time);
#ifdef SPLATT_USE_MPI
  timer_reset(&timers[TIMER_MPI_COMM]);
#endif

  /* foreach epoch */
  for(idx_t e=1; e < ws->max_its+1; ++e) {
    loss = 0;
    #pragma omp parallel reduction(+:loss)
    {
      int const tid = omp_get_thread_num();

      for(idx_t f=0; f < nfactors; ++f) {
        /* add current component to residual */
        p_update_residual(csf, f, model, ws, 1);

        for(idx_t inner=0; inner < ws->num_inner; ++inner) {

          /* compute new column 'f' for each factor */
          for(idx_t m=0; m < nmodes; ++m) {
            if(ws->isdense[m]) {
              p_densemode_ccd_update(csf, m, f, model, ws, thd_densefactors,
                  tid);
            } else {
              p_sparsemode_ccd_update(csf, m, f, model, ws, tid);
            }

#ifdef SPLATT_USE_MPI
            /* exchange partial products */
            p_reduce_partials_all2all(model, ws, m, f);
#endif

            /* numerator/denominator are now computed; update factor column */
            p_compute_newcol(model, ws, m, f);

#ifdef SPLATT_USE_MPI
            /* exchange update columns */
            p_update_col_all2all(model, ws, m, f);
#endif
          } /* foreach mode */
        } /* foreach inner iteration */

        /* subtract new rank-1 factor from residual */
        loss = p_update_residual(csf, f, model, ws, -1);

      } /* foreach factor */
    } /* omp parallel */


#if MEASURE_DRIFT == 1
    val_t const gold = tc_loss_sq(train, model, ws);
    if(rank == 0) {
      printf("  residual: %e actual: %e diff: %e\n", loss, gold, loss - gold);
    }
#endif

    /* compute RMSE and adjust learning rate */
    frobsq = tc_frob_sq(model, ws);

#ifdef SPLATT_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &loss, 1, SPLATT_MPI_VAL, MPI_SUM,
        ws->rinfo->comm_3d);
#endif

    if(tc_converge(train, validate, model, loss, frobsq, e, ws)) {
      break;
    }

  } /* foreach epoch */

  /* print times */
#ifdef SPLATT_USE_MPI
  fflush(stdout);
  MPI_Barrier(ws->rinfo->comm_3d);
  printf("  rank: %d residual %0.3fs dense: %0.3fs sparse: %0.3fs newcol: %0.3fs\n",
    ws->rinfo->rank, ws->resid_time.seconds, ws->dense_time.seconds, ws->sparse_time.seconds, ws->newcol_time.seconds);
#else
  printf("  residual %0.3fs dense: %0.3fs sparse: %0.3fs newcol: %0.3fs\n",
    ws->resid_time.seconds, ws->dense_time.seconds, ws->sparse_time.seconds, ws->newcol_time.seconds);
#endif

  p_transpose_model(model);

  /* cleanup */
  csf_free(csf, opts);
  if(ws->num_dense > 0) {
    thd_free(thd_densefactors, ws->nthreads);
  }
}


