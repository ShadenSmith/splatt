
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "mttkrp.h"
#include "thd_info.h"
#include "tile.h"
#include "util.h"

#include "mutex_pool.h"

/******************************************************************************
 * MUTEX FUNCTIONS
 *****************************************************************************/

static mutex_pool * pool = NULL;


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
int splatt_mttkrp(
    splatt_idx_t const mode,
    splatt_idx_t const ncolumns,
    splatt_csf const * const tensors,
    splatt_val_t ** matrices,
    splatt_val_t * const matout,
    double const * const options)
{
  idx_t const nmodes = tensors->nmodes;

  /* fill matrix pointers  */
  matrix_t * mats[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (matrix_t *) splatt_malloc(sizeof(matrix_t));
    mats[m]->I = tensors->dims[m];
    mats[m]->J = ncolumns,
    mats[m]->rowmajor = 1;
    mats[m]->vals = matrices[m];
  }
  mats[MAX_NMODES] = (matrix_t *) splatt_malloc(sizeof(matrix_t));
  mats[MAX_NMODES]->I = tensors->dims[mode];
  mats[MAX_NMODES]->J = ncolumns;
  mats[MAX_NMODES]->rowmajor = 1;
  mats[MAX_NMODES]->vals = matout;

  /* Setup thread structures. + 64 bytes is to avoid false sharing. */
  idx_t const nthreads = (idx_t) options[SPLATT_OPTION_NTHREADS];
  splatt_omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 3,
    (ncolumns * ncolumns * sizeof(val_t)) + 64,
    0,
    (nmodes * ncolumns * sizeof(val_t)) + 64);

  /* do the MTTKRP */
  mttkrp_csf(tensors, mats, mode, thds, options);

  /* cleanup */
  thd_free(thds, nthreads);
  for(idx_t m=0; m < nmodes; ++m) {
    free(mats[m]);
  }
  free(mats[MAX_NMODES]);

  return SPLATT_SUCCESS;
}



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static inline void p_add_hada(
  val_t * const restrict out,
  val_t const * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] += a[f] * b[f];
  }
}


static inline void p_add_hada_clear(
  val_t * const restrict out,
  val_t * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] += a[f] * b[f];
    a[f] = 0;
  }
}


static inline void p_assign_hada(
  val_t * const restrict out,
  val_t const * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] = a[f] * b[f];
  }
}


static inline void p_csf_process_fiber_lock(
  val_t * const leafmat,
  val_t const * const restrict accumbuf,
  idx_t const nfactors,
  idx_t const start,
  idx_t const end,
  idx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = leafmat + (inds[jj] * nfactors);
    val_t const v = vals[jj];
    mutex_set_lock(pool, inds[jj]);
    for(idx_t f=0; f < nfactors; ++f) {
      leafrow[f] += v * accumbuf[f];
    }
    mutex_unset_lock(pool, inds[jj]);
  }
}

static inline void p_csf_process_fiber_nolock(
  val_t * const leafmat,
  val_t const * const restrict accumbuf,
  idx_t const nfactors,
  idx_t const start,
  idx_t const end,
  idx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = leafmat + (inds[jj] * nfactors);
    val_t const v = vals[jj];
    for(idx_t f=0; f < nfactors; ++f) {
      leafrow[f] += v * accumbuf[f];
    }
  }
}


static inline void p_csf_process_fiber(
  val_t * const restrict accumbuf,
  idx_t const nfactors,
  val_t const * const leafmat,
  idx_t const start,
  idx_t const end,
  idx_t const * const inds,
  val_t const * const vals)
{
  /* foreach nnz in fiber */
  for(idx_t j=start; j < end; ++j) {
    val_t const v = vals[j] ;
    val_t const * const restrict row = leafmat + (nfactors * inds[j]);
    for(idx_t f=0; f < nfactors; ++f) {
      accumbuf[f] += v * row[f];
    }
  }
}


static inline void p_propagate_up(
  val_t * const out,
  val_t * const * const buf,
  idx_t * const restrict idxstack,
  idx_t const init_depth,
  idx_t const init_idx,
  idx_t const * const * const fp,
  idx_t const * const * const fids,
  val_t const * const restrict vals,
  val_t ** mvals,
  idx_t const nmodes,
  idx_t const nfactors)
{
  /* push initial idx initialize idxstack */
  idxstack[init_depth] = init_idx;
  for(idx_t m=init_depth+1; m < nmodes; ++m) {
    idxstack[m] = fp[m-1][idxstack[m-1]];
  }

  assert(init_depth < nmodes-1);

  /* clear out accumulation buffer */
  for(idx_t f=0; f < nfactors; ++f) {
    buf[init_depth+1][f] = 0;
  }

  while(idxstack[init_depth+1] < fp[init_depth][init_idx+1]) {
    /* skip to last internal mode */
    idx_t depth = nmodes - 2;

    /* process all nonzeros [start, end) into buf[depth]*/
    idx_t const start = fp[depth][idxstack[depth]];
    idx_t const end   = fp[depth][idxstack[depth]+1];
    p_csf_process_fiber(buf[depth+1], nfactors, mvals[depth+1],
        start, end, fids[depth+1], vals);

    idxstack[depth+1] = end;

    /* exit early if there is no propagation to do... */
    if(init_depth == nmodes-2) {
      for(idx_t f=0; f < nfactors; ++f) {
        out[f] = buf[depth+1][f];
      }
      return;
    }

    /* Propagate up until we reach a node with more children to process */
    do {
      /* propagate result up and clear buffer for next sibling */
      val_t const * const restrict fibrow
          = mvals[depth] + (fids[depth][idxstack[depth]] * nfactors);
      p_add_hada_clear(buf[depth], buf[depth+1], fibrow, nfactors);

      ++idxstack[depth];
      --depth;
    } while(depth > init_depth &&
        idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
  } /* end DFS */

  /* copy to out */
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] = buf[init_depth+1][f];
  }
}


static void p_csf_mttkrp_root_tiled3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[1]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[splatt_omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    val_t * const restrict mv = ovals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] += v * bv[r];
        }
      }

      /* scale inner products by row of A and update to M */
      val_t const * const restrict av = avals  + (fids[f] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        mv[r] += accumF[r] * av[r];
      }
    }
  }
}


static void p_csf_mttkrp_root3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[1]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[splatt_omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    val_t * const restrict mv = ovals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] += v * bv[r];
        }
      }

      /* scale inner products by row of A and update to M */
      val_t const * const restrict av = avals  + (fids[f] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        mv[r] += accumF[r] * av[r];
      }
    }
  }
}


static void p_csf_mttkrp_internal3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[splatt_omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict rv = avals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] += v * bv[r];
        }
      }

      /* write to fiber row */
      val_t * const restrict ov = ovals  + (fids[f] * nfactors);
      mutex_set_lock(pool, fids[f]);
      for(idx_t r=0; r < nfactors; ++r) {
        ov[r] += rv[r] * accumF[r];
      }
      mutex_unset_lock(pool, fids[f]);
    }
  }
}


static void p_csf_mttkrp_leaf3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[1]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[splatt_omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict rv = avals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* fill fiber with hada */
      val_t const * const restrict av = bvals  + (fids[f] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = rv[r] * av[r];
      }

      /* foreach nnz in fiber, scale with hada and write to ovals */
      for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t * const restrict ov = ovals + (inds[jj] * nfactors);
        mutex_set_lock(pool, inds[jj]);
        for(idx_t r=0; r < nfactors; ++r) {
          ov[r] += v * accumF[r];
        }
        mutex_unset_lock(pool, inds[jj]);
      }
    }
  }
}


static void p_csf_mttkrp_root_tiled(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;

  /* empty tile, just return */
  if(vals == NULL) {
    return;
  }

  if(nmodes == 3) {
    p_csf_mttkrp_root_tiled3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = splatt_omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }

  val_t * const ovals = mats[MAX_NMODES]->vals;

  idx_t const nfibs = ct->pt[tile_id].nfibs[0];
  assert(nfibs <= mats[MAX_NMODES]->I);

  for(idx_t s=0; s < nfibs; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    assert(fid < mats[MAX_NMODES]->I);

    p_propagate_up(buf[0], buf, idxstack, 0, s, fp, fids,
        vals, mvals, nmodes, nfactors);

    val_t * const restrict orow = ovals + (fid * nfactors);
    val_t const * const restrict obuf = buf[0];
    for(idx_t f=0; f < nfactors; ++f) {
      orow[f] += obuf[f];
    }
  } /* end foreach outer slice */
}



static void p_csf_mttkrp_root(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;

  /* empty tile, just return */
  if(vals == NULL) {
    return;
  }

  if(nmodes == 3) {
    p_csf_mttkrp_root3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = splatt_omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }

  val_t * const ovals = mats[MAX_NMODES]->vals;

  idx_t const nfibs = ct->pt[tile_id].nfibs[0];
  assert(nfibs <= mats[MAX_NMODES]->I);

  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nfibs; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    assert(fid < mats[MAX_NMODES]->I);

    p_propagate_up(buf[0], buf, idxstack, 0, s, fp, fids,
        vals, mvals, nmodes, nfactors);

    val_t * const restrict orow = ovals + (fid * nfactors);
    val_t const * const restrict obuf = buf[0];
    for(idx_t f=0; f < nfactors; ++f) {
      orow[f] += obuf[f];
    }
  } /* end foreach outer slice */
}


static void p_csf_mttkrp_leaf_tiled3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[1]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[splatt_omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict rv = avals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* fill fiber with hada */
      val_t const * const restrict av = bvals  + (fids[f] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = rv[r] * av[r];
      }

      /* foreach nnz in fiber, scale with hada and write to ovals */
      for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t * const restrict ov = ovals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          ov[r] += v * accumF[r];
        }
      }
    }
  }
}




static void p_csf_mttkrp_leaf_tiled(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  val_t const * const vals = ct->pt[tile_id].vals;
  idx_t const nmodes = ct->nmodes;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_leaf_tiled3(ct, tile_id, mats, thds);
    return;
  }

  /* extract tensor structures */
  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;

  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = splatt_omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
  }

  /* foreach outer slice */
  idx_t const nouter = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nouter; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
    idxstack[0] = s;

    /* clear out stale data */
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* first buf will always just be a matrix row */
    val_t const * const rootrow = mvals[0] + (fid*nfactors);
    val_t * const rootbuf = buf[0];
    for(idx_t f=0; f < nfactors; ++f) {
      rootbuf[f] = rootrow[f];
    }

    idx_t depth = 0;

    idx_t const outer_end = fp[0][s+1];
    while(idxstack[1] < outer_end) {
      /* move down to an nnz node */
      for(; depth < nmodes-2; ++depth) {
        /* propogate buf down */
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      /* process all nonzeros [start, end) */
      idx_t const start = fp[depth][idxstack[depth]];
      idx_t const end   = fp[depth][idxstack[depth]+1];
      p_csf_process_fiber_nolock(mats[MAX_NMODES]->vals, buf[depth],
          nfactors, start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end outer slice loop */
}


static void p_csf_mttkrp_leaf(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  /* extract tensor structures */
  val_t const * const vals = ct->pt[tile_id].vals;
  idx_t const nmodes = ct->nmodes;

  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_leaf3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;

  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = splatt_omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
  }

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
    idxstack[0] = s;

    /* clear out stale data */
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* first buf will always just be a matrix row */
    val_t const * const restrict rootrow = mvals[0] + (fid*nfactors);
    val_t * const rootbuf = buf[0];
    for(idx_t f=0; f < nfactors; ++f) {
      rootbuf[f] = rootrow[f];
    }

    idx_t depth = 0;

    idx_t const outer_end = fp[0][s+1];
    while(idxstack[1] < outer_end) {
      /* move down to an nnz node */
      for(; depth < nmodes-2; ++depth) {
        /* propogate buf down */
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      /* process all nonzeros [start, end) */
      idx_t const start = fp[depth][idxstack[depth]];
      idx_t const end   = fp[depth][idxstack[depth]+1];
      p_csf_process_fiber_lock(mats[MAX_NMODES]->vals, buf[depth],
          nfactors, start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end outer slice loop */
}


static void p_csf_mttkrp_internal_tiled3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[splatt_omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict rv = avals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] += v * bv[r];
        }
      }

      /* write to fiber row */
      val_t * const restrict ov = ovals  + (fids[f] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        ov[r] += rv[r] * accumF[r];
      }
    }
  }
}


static void p_csf_mttkrp_internal_tiled(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_internal_tiled3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;

  idx_t const nfactors = mats[0]->J;

  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes);

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = splatt_omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }
  val_t * const ovals = mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    /* push outer slice and fill stack */
    idxstack[0] = s;
    for(idx_t m=1; m <= outdepth; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* fill first buf */
    val_t const * const restrict rootrow = mvals[0] + (fid*nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buf[0][f] = rootrow[f];
    }

    /* process entire subtree */
    idx_t depth = 0;
    while(idxstack[1] < fp[0][s+1]) {
      /* propagate values down to outdepth-1 */
      for(; depth < outdepth; ++depth) {
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      /* write to output and clear buf[outdepth] for next subtree */
      idx_t const noderow = fids[outdepth][idxstack[outdepth]];

      /* propagate value up to buf[outdepth] */
      p_propagate_up(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
          fp, fids, vals, mvals, nmodes, nfactors);

      val_t * const restrict outbuf = ovals + (noderow * nfactors);
      p_add_hada_clear(outbuf, buf[outdepth], buf[outdepth-1], nfactors);

      /* backtrack to next unfinished node */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end foreach outer slice */
}


static void p_csf_mttkrp_internal(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_internal3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes);

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = splatt_omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }
  val_t * const ovals = mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    /* push outer slice and fill stack */
    idxstack[0] = s;
    for(idx_t m=1; m <= outdepth; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* fill first buf */
    val_t const * const restrict rootrow = mvals[0] + (fid*nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buf[0][f] = rootrow[f];
    }

    /* process entire subtree */
    idx_t depth = 0;
    while(idxstack[1] < fp[0][s+1]) {
      /* propagate values down to outdepth-1 */
      for(; depth < outdepth; ++depth) {
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      /* write to output and clear buf[outdepth] for next subtree */
      idx_t const noderow = fids[outdepth][idxstack[outdepth]];

      /* propagate value up to buf[outdepth] */
      p_propagate_up(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
          fp, fids, vals, mvals, nmodes, nfactors);

      val_t * const restrict outbuf = ovals + (noderow * nfactors);
      mutex_set_lock(pool, noderow);
      p_add_hada_clear(outbuf, buf[outdepth], buf[outdepth-1], nfactors);
      mutex_unset_lock(pool, noderow);

      /* backtrack to next unfinished node */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end foreach outer slice */
}


/* determine which function to call */
static void p_root_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  #pragma omp parallel
  {
    timer_start(&thds[splatt_omp_get_thread_num()].ttime);
    /* tile id */
    idx_t tid = 0;
    switch(tensor->which_tile) {
    case SPLATT_NOTILE:
      p_csf_mttkrp_root(tensor, 0, mats, thds);
      break;
    case SPLATT_DENSETILE:
      /* this mode may not be tiled due to minimum tiling depth */
      if(opts[SPLATT_OPTION_TILEDEPTH] > 0) {
        for(idx_t t=0; t < tensor->ntiles; ++t) {
          p_csf_mttkrp_root(tensor, t, mats, thds);
          #pragma omp barrier
        }
      } else {
        /* distribute tiles to threads */
        #pragma omp for schedule(dynamic, 1) nowait
        for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
          tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
              mode, t);
          while(tid != TILE_END) {
            p_csf_mttkrp_root_tiled(tensor, tid, mats, thds);
            tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
          }
        }
      }
      break;

    /* XXX */
    case SPLATT_SYNCTILE:
      break;
    case SPLATT_COOPTILE:
      break;
    }
    timer_stop(&thds[splatt_omp_get_thread_num()].ttime);
  } /* end omp parallel */
}


static void p_leaf_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  idx_t const depth = nmodes - 1;

  #pragma omp parallel
  {
    timer_start(&thds[splatt_omp_get_thread_num()].ttime);

    /* tile id */
    idx_t tid = 0;
    switch(tensor->which_tile) {
    case SPLATT_NOTILE:
      p_csf_mttkrp_leaf(tensor, 0, mats, thds);
      break;
    case SPLATT_DENSETILE:
      /* this mode may not be tiled due to minimum tiling depth */
      if(opts[SPLATT_OPTION_TILEDEPTH] > depth) {
        for(idx_t t=0; t < tensor->ntiles; ++t) {
          p_csf_mttkrp_leaf(tensor, 0, mats, thds);
        }
      } else {
        #pragma omp for schedule(dynamic, 1) nowait
        for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
          tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
              mode, t);
          while(tid != TILE_END) {
            p_csf_mttkrp_leaf_tiled(tensor, tid, mats, thds);
            tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
          }
        }
      }
      break;

    /* XXX */
    case SPLATT_SYNCTILE:
      break;
    case SPLATT_COOPTILE:
      break;
    }
    timer_stop(&thds[splatt_omp_get_thread_num()].ttime);
  } /* end omp parallel */
}


static void p_intl_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  idx_t const depth = csf_mode_depth(mode, tensor->dim_perm, nmodes);

  #pragma omp parallel
  {
    timer_start(&thds[splatt_omp_get_thread_num()].ttime);
    /* tile id */
    idx_t tid = 0;
    switch(tensor->which_tile) {
    case SPLATT_NOTILE:
      p_csf_mttkrp_internal(tensor, 0, mats, mode, thds);
      break;
    case SPLATT_DENSETILE:
      /* this mode may not be tiled due to minimum tiling depth */
      if(opts[SPLATT_OPTION_TILEDEPTH] > depth) {
        for(idx_t t=0; t < tensor->ntiles; ++t) {
          p_csf_mttkrp_internal(tensor, t, mats, mode, thds);
        }
      } else {
        #pragma omp for schedule(dynamic, 1) nowait
        for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
          tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
              mode, t);
          while(tid != TILE_END) {
            p_csf_mttkrp_internal_tiled(tensor, tid, mats, mode, thds);
            tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
          }
        }
      }
      break;

    /* XXX */
    case SPLATT_SYNCTILE:
      break;
    case SPLATT_COOPTILE:
      break;
    }

    timer_stop(&thds[splatt_omp_get_thread_num()].ttime);
  } /* end omp parallel */
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mttkrp_csf(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  double const * const opts)
{
  if(pool == NULL) {
    pool = mutex_alloc();
  }

  /* clear output matrix */
  matrix_t * const M = mats[MAX_NMODES];
  M->I = tensors[0].dims[mode];
  memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  splatt_omp_set_num_threads(opts[SPLATT_OPTION_NTHREADS]);

  idx_t nmodes = tensors[0].nmodes;
  /* find out which level in the tree this is */
  idx_t outdepth = MAX_NMODES;

  /* choose which MTTKRP function to use */
  splatt_csf_type which = opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which) {
  case SPLATT_CSF_ONEMODE:
    outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
    if(outdepth == 0) {
      p_root_decide(tensors+0, mats, mode, thds, opts);
    } else if(outdepth == nmodes - 1) {
      p_leaf_decide(tensors+0, mats, mode, thds, opts);
    } else {
      p_intl_decide(tensors+0, mats, mode, thds, opts);
    }
    break;

  case SPLATT_CSF_TWOMODE:
    /* longest mode handled via second tensor's root */
    if(mode == tensors[0].dim_perm[nmodes-1]) {
      p_root_decide(tensors+1, mats, mode, thds, opts);
    /* root and internal modes are handled via first tensor */
    } else {
      outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
      if(outdepth == 0) {
        p_root_decide(tensors+0, mats, mode, thds, opts);
      } else {
        p_intl_decide(tensors+0, mats, mode, thds, opts);
      }
    }
    break;

  case SPLATT_CSF_ALLMODE:
    p_root_decide(tensors+mode, mats, mode, thds, opts);
    break;
  }
}









/******************************************************************************
 * DEPRECATED FUNCTIONS
 *****************************************************************************/








/******************************************************************************
 * SPLATT MTTKRP
 *****************************************************************************/

void mttkrp_splatt(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  if(ft->tiled == SPLATT_SYNCTILE) {
    mttkrp_splatt_sync_tiled(ft, mats, mode, thds, nthreads);
    return;
  }
  if(ft->tiled == SPLATT_COOPTILE) {
    mttkrp_splatt_coop_tiled(ft, mats, mode, thds, nthreads);
    return;
  }

  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];
  idx_t const nslices = ft->dims[mode];
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict sptr = ft->sptr;
  idx_t const * const restrict fptr = ft->fptr;
  idx_t const * const restrict fids = ft->fids;
  idx_t const * const restrict inds = ft->inds;
  val_t const * const restrict vals = ft->vals;

  #pragma omp parallel
  {
    int const tid = splatt_omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
    timer_start(&thds[tid].ttime);

    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
      val_t * const restrict mv = mvals + (s * rank);

      /* foreach fiber in slice */
      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t const * const restrict av = avals  + (fids[f] * rank);
        for(idx_t r=0; r < rank; ++r) {
          mv[r] += accumF[r] * av[r];
        }
      }
    }

    timer_stop(&thds[tid].ttime);
  } /* end parallel region */
}


void mttkrp_splatt_sync_tiled(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];

  idx_t const nslabs = ft->nslabs;
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict slabptr = ft->slabptr;
  idx_t const * const restrict sids = ft->sids;
  idx_t const * const restrict fptr = ft->fptr;
  idx_t const * const restrict fids = ft->fids;
  idx_t const * const restrict inds = ft->inds;
  val_t const * const restrict vals = ft->vals;

  #pragma omp parallel
  {
    int const tid = splatt_omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
    timer_start(&thds[tid].ttime);

    #pragma omp for schedule(dynamic, 1) nowait
    for(idx_t s=0; s < nslabs; ++s) {
      /* foreach fiber in slice */
      for(idx_t f=slabptr[s]; f < slabptr[s+1]; ++f) {
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t       * const restrict mv = mvals + (sids[f] * rank);
        val_t const * const restrict av = avals + (fids[f] * rank);
        for(idx_t r=0; r < rank; ++r) {
          mv[r] += accumF[r] * av[r];
        }
      }
    }

    timer_stop(&thds[tid].ttime);
  } /* end parallel region */
}


void mttkrp_splatt_coop_tiled(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];

  idx_t const nslabs = ft->nslabs;
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict slabptr = ft->slabptr;
  idx_t const * const restrict sptr = ft->sptr;
  idx_t const * const restrict sids = ft->sids;
  idx_t const * const restrict fptr = ft->fptr;
  idx_t const * const restrict fids = ft->fids;
  idx_t const * const restrict inds = ft->inds;
  val_t const * const restrict vals = ft->vals;

  #pragma omp parallel
  {
    int const tid = splatt_omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
    val_t * const localm = (val_t *) thds[tid].scratch[1];
    timer_start(&thds[tid].ttime);

    /* foreach slab */
    for(idx_t s=0; s < nslabs; ++s) {
      /* foreach fiber in slab */
      #pragma omp for schedule(dynamic, 8)
      for(idx_t sl=slabptr[s]; sl < slabptr[s+1]; ++sl) {
        idx_t const slice = sids[sl];
        for(idx_t f=sptr[sl]; f < sptr[sl+1]; ++f) {
          /* first entry of the fiber is used to initialize accumF */
          idx_t const jjfirst  = fptr[f];
          val_t const vfirst   = vals[jjfirst];
          val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] = vfirst * bv[r];
          }

          /* foreach nnz in fiber */
          for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
            val_t const v = vals[jj];
            val_t const * const restrict bv = bvals + (inds[jj] * rank);
            for(idx_t r=0; r < rank; ++r) {
              accumF[r] += v * bv[r];
            }
          }

          /* scale inner products by row of A and update thread-local M */
          val_t       * const restrict mv = localm + ((slice % TILE_SIZES[0]) * rank);
          val_t const * const restrict av = avals + (fids[f] * rank);
          for(idx_t r=0; r < rank; ++r) {
            mv[r] += accumF[r] * av[r];
          }
        }
      }

      idx_t const start = s * TILE_SIZES[0];
      idx_t const stop  = SS_MIN((s+1) * TILE_SIZES[0], ft->dims[mode]);

      #pragma omp for schedule(static)
      for(idx_t i=start; i < stop; ++i) {
        /* map i back to global slice id */
        idx_t const localrow = i % TILE_SIZES[0];
        for(idx_t t=0; t < nthreads; ++t) {
          val_t * const threadm = (val_t *) thds[t].scratch[1];
          for(idx_t r=0; r < rank; ++r) {
            mvals[r + (i*rank)] += threadm[r + (localrow*rank)];
            threadm[r + (localrow*rank)] = 0.;
          }
        }
      }

    } /* end foreach slab */
    timer_stop(&thds[tid].ttime);
  } /* end omp parallel */
}



/******************************************************************************
 * GIGA MTTKRP
 *****************************************************************************/
void mttkrp_giga(
  spmatrix_t const * const spmat,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];

  idx_t const I = spmat->I;
  idx_t const rank = M->J;

  idx_t const * const restrict rowptr = spmat->rowptr;
  idx_t const * const restrict colind = spmat->colind;
  val_t const * const restrict vals   = spmat->vals;

  #pragma omp parallel
  {
    for(idx_t r=0; r < rank; ++r) {
      val_t       * const restrict mv =  M->vals + (r * I);
      val_t const * const restrict av =  A->vals + (r * A->I);
      val_t const * const restrict bv =  B->vals + (r * B->I);

      /* Joined Hadamard products of X, C, and B */
      #pragma omp for schedule(dynamic, 16)
      for(idx_t i=0; i < I; ++i) {
        for(idx_t y=rowptr[i]; y < rowptr[i+1]; ++y) {
          idx_t const a = colind[y] / B->I;
          idx_t const b = colind[y] % B->I;
          scratch[y] = vals[y] * av[a] * bv[b];
        }
      }

      /* now accumulate rows into column of M1 */
      #pragma omp for schedule(dynamic, 16)
      for(idx_t i=0; i < I; ++i) {
        val_t sum = 0;
        for(idx_t y=rowptr[i]; y < rowptr[i+1]; ++y) {
          sum += scratch[y];
        }
        mv[i] = sum;
      }
    }
  }
}


/******************************************************************************
 * TTBOX MTTKRP
 *****************************************************************************/
void mttkrp_ttbox(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];

  idx_t const I = tt->dims[mode];
  idx_t const rank = M->J;

  memset(M->vals, 0, I * rank * sizeof(val_t));

  idx_t const nnz = tt->nnz;
  idx_t const * const restrict indM = tt->ind[mode];
  idx_t const * const restrict indA =
    mode == 0 ? tt->ind[1] : tt->ind[0];
  idx_t const * const restrict indB =
    mode == 2 ? tt->ind[1] : tt->ind[2];

  val_t const * const restrict vals = tt->vals;

  for(idx_t r=0; r < rank; ++r) {
    val_t       * const restrict mv =  M->vals + (r * I);
    val_t const * const restrict av =  A->vals + (r * A->I);
    val_t const * const restrict bv =  B->vals + (r * B->I);

    /* stretch out columns of A and B */
    #pragma omp parallel for
    for(idx_t x=0; x < nnz; ++x) {
      scratch[x] = vals[x] * av[indA[x]] * bv[indB[x]];
    }

    /* now accumulate into m1 */
    for(idx_t x=0; x < nnz; ++x) {
      mv[indM[x]] += scratch[x];
    }
  }
}

void mttkrp_stream(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode)
{
  matrix_t * const M = mats[MAX_NMODES];
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = M->J;

  val_t * const outmat = M->vals;
  memset(outmat, 0, I * nfactors * sizeof(val_t));

  idx_t const nmodes = tt->nmodes;

  val_t * accum = (val_t *) splatt_malloc(nfactors * sizeof(val_t));

  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[m]->vals;
  }

  val_t const * const restrict vals = tt->vals;

  /* stream through nnz */
  for(idx_t n=0; n < tt->nnz; ++n) {
    /* initialize with value */
    for(idx_t f=0; f < nfactors; ++f) {
      accum[f] = vals[n];
    }

    for(idx_t m=0; m < nmodes; ++m) {
      if(m == mode) {
        continue;
      }
      val_t const * const restrict inrow = mvals[m] + (tt->ind[m][n] * nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        accum[f] *= inrow[f];
      }
    }

    /* write to output */
    val_t * const restrict outrow = outmat + (tt->ind[mode][n] * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      outrow[f] += accum[f];
    }
  }

  free(accum);
}


