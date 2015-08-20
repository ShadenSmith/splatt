
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "mttkrp.h"
#include "thd_info.h"
#include "tile.h"
#include <omp.h>

#include "io.h"

//#define VERB
#define CSF_DOWN 0
#define CSF_UP   1

#define NLOCKS 1024
static omp_lock_t locks[NLOCKS];

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static inline void __add_hada(
  val_t * const restrict out,
  val_t const * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] += a[f] * b[f];
  }
}


static inline void __add_hada_clear(
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


static inline void __assign_hada(
  val_t * const restrict out,
  val_t const * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] = a[f] * b[f];
  }
}


static inline void __csf_process_fiber_lock(
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
    omp_set_lock(locks + (inds[jj] % NLOCKS));
    for(idx_t f=0; f < nfactors; ++f) {
      leafrow[f] += v * accumbuf[f];
    }
    omp_unset_lock(locks + (inds[jj] % NLOCKS));
  }
}


static inline void __csf_process_fiber(
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
    val_t const v = vals[j];
    val_t const * const restrict row = leafmat + (nfactors * inds[j]);
    for(idx_t f=0; f < nfactors; ++f) {
      accumbuf[f] += v * row[f];
    }
  }
}


static inline void __propagate_up(
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
    __csf_process_fiber(buf[depth+1], nfactors, mvals[depth+1],
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
      __add_hada_clear(buf[depth], buf[depth+1], fibrow, nfactors);

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

static void __ctensor_mttkrp_root(
  ctensor_t const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  printf("ROOT2");
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const nfactors = mats[0]->J;

  #pragma omp parallel default(shared)
  {
    int const tid = omp_get_thread_num();
    timer_start(&thds[tid].ttime);

    val_t * mvals[MAX_NMODES];
    val_t * buf[MAX_NMODES];
    idx_t idxstack[MAX_NMODES];

    for(idx_t m=0; m < nmodes; ++m) {
      mvals[m] = mats[ct->dim_perm[m]]->vals;
      /* grab the next row of buf from thds */
      buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
      memset(buf[m], 0, nfactors * sizeof(idx_t));
    }

    val_t * const ovals = mats[MAX_NMODES]->vals;

    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < ct->dims[ct->dim_perm[0]]; ++s) {
      __propagate_up(ovals + (s * nfactors), buf, idxstack, 0, s, fp, fids,
          vals, mvals, nmodes, nfactors);
    } /* end foreach outer slice */

    timer_start(&thds[tid].ttime);
  } /* end omp parallel */
}


#if 0
static void __csf_mttkrp_internal2(
  csf_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  printf("INTL2");
  /* extract tensor structures */
  idx_t const nmodes = ft->nmodes;
  idx_t const * const * const restrict fp = (idx_t const * const *) ft->fptr;
  idx_t const * const * const restrict fids = (idx_t const * const *) ft->fids;
  val_t const * const vals = ft->vals;

  idx_t const nfactors = mats[0]->J;
  /* find out which level in the tree this is */
  idx_t outdepth = 0;
  for(idx_t m=0; m < ft->nmodes; ++m) {
    if(ft->dim_perm[m] == mode) {
      outdepth = m;
      break;
    }
  }

  #pragma omp parallel default(shared)
  {
    int const tid = omp_get_thread_num();
    timer_start(&thds[tid].ttime);

    val_t * mvals[MAX_NMODES];
    val_t * buf[MAX_NMODES];
    idx_t idxstack[MAX_NMODES];

    for(idx_t m=0; m < nmodes; ++m) {
      mvals[m] = mats[ft->dim_perm[m]]->vals;
      /* grab the next row of buf from thds */
      buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
      memset(buf[m], 0, nfactors * sizeof(idx_t));
    }
    val_t * const ovals = mats[MAX_NMODES]->vals;

    #pragma omp for schedule(dynamic, 16)
    for(idx_t s=0; s < ft->dims[ft->dim_perm[0]]; ++s) {
      /* push outer slice and fill stack */
      idxstack[0] = s;
      for(idx_t m=1; m <= outdepth; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* fill first buf */
      val_t const * const restrict rootrow = mvals[0] + (s*nfactors);
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
          __assign_hada(buf[depth+1], buf[depth], drow, nfactors);
        }

        /* write to output and clear buf[outdepth] for next subtree */
        idx_t const noderow = fids[outdepth][idxstack[outdepth]];

        /* propagate value up to buf[outdepth] */
        __propagate_up(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
            fp, fids, vals, mvals, nmodes, nfactors);

        val_t * const restrict outbuf = ovals + (noderow * nfactors);
        omp_set_lock(locks + (noderow % NLOCKS));
        __add_hada_clear(outbuf, buf[outdepth], buf[outdepth-1], nfactors);
        omp_unset_lock(locks + (noderow % NLOCKS));

        /* backtrack to next unfinished node */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */
    } /* end foreach outer slice */

    timer_stop(&thds[tid].ttime);
  } /* end omp parallel */
}


static void __csf_mttkrp_internal(
  csf_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  printf("INTL");
  /* extract tensor structures */
  idx_t const nmodes = ft->nmodes;
  idx_t const * const * const restrict fp = (idx_t const * const *) ft->fptr;
  idx_t const * const * const restrict fids = (idx_t const * const *) ft->fids;

  idx_t const nfactors = mats[0]->J;

  idx_t const outdepth = ft->dim_perm[mode];
  idx_t const nnzdepth = nmodes - 2; /* the last internal nodes */

  val_t * const outvals = mats[MAX_NMODES]->vals;

  #pragma omp parallel default(shared)
  {
    int const tid = omp_get_thread_num();
    timer_start(&thds[tid].ttime);

    val_t * mvals[MAX_NMODES];
    val_t * buf[MAX_NMODES];
    idx_t idxstack[MAX_NMODES];
    for(idx_t m=0; m < nmodes; ++m) {
      mvals[m] = mats[ft->dim_perm[m]]->vals;
      /* grab the next row of buf from thds */
      buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    }

    /* foreach outer slice */
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < ft->dims[ft->dim_perm[0]]; ++s) {
      idx_t depth = 0;
      /* push current outer slice */
      idxstack[depth++] = s;

      /* clear out stale data */
      for(idx_t m=1; m < ft->nmodes-1; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* fill first buf */
      val_t * const rootbuf = buf[0];
      val_t * const nextbuf = buf[1];
      val_t const * const restrict rootrow = mvals[0] + (s*nfactors);
      val_t const * const restrict nextrow
          = mvals[1] + (fids[1][idxstack[1]]*nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        rootbuf[f] = rootrow[f];
        /* also initialize next level if it's not our output */
        if(outdepth > 1) {
          nextbuf[f] = rootbuf[f] * nextrow[f];
        }
      }

      /* loop until entire tree has been processed */
      while(idxstack[1] < fp[0][s+1]) {
        int direction = CSF_DOWN;

        /* index of current row */
        idx_t noderow = depth > 0 ? fids[depth][idxstack[depth]] : 0;

        val_t * hadabuf;
        val_t * clearbuf;
        val_t * multbuf;

        /* Are we at the bottom (nnz) or all children processed? */
        if(depth == nnzdepth ||
              idxstack[depth+1] == fp[depth][idxstack[depth]+1]) {

          /* process all nonzeros [start, end) */
          if(depth == nnzdepth) {
            idx_t const start = fp[depth][idxstack[depth]];
            idx_t const end   = fp[depth][idxstack[depth]+1];

            /* regular fiber processing; write to buf[depth] */
            __csf_process_fiber(buf[depth], nfactors, mvals[depth+1],
                start, end, fids[depth+1], ft->vals);
          }

          clearbuf = buf[depth];
          /* all children have now been processed and multiplied into buf[depth],
           * we can now add to output */
          if(depth == outdepth) {
            /* write to output and use a mutex */
            hadabuf = outvals + (noderow * nfactors);
            multbuf = buf[depth-1];
            omp_set_lock(locks + (noderow % NLOCKS));
            __add_hada_clear(hadabuf, clearbuf, multbuf, nfactors);
            omp_unset_lock(locks + (noderow % NLOCKS));
          } else if(depth > outdepth) {
            /* just propagate results up tree */
            hadabuf = buf[depth-1];
            multbuf = mvals[depth] + (noderow * nfactors);
            __add_hada_clear(hadabuf, clearbuf, multbuf, nfactors);
          } else {
            /* don't accumulate products up the tree if we are above outdepth */
          }

          /* children are done, move up the tree */
          direction = CSF_UP;
        }

        switch(direction) {
        case CSF_DOWN:
          ++depth;
          /* propagate down buf[depth-1] down to new child */
          if(depth < outdepth) {
            noderow = fids[depth][idxstack[depth]];
            __assign_hada(buf[depth], buf[depth-1], mvals[depth]+(noderow*nfactors), nfactors);
          }
          break;
        case CSF_UP:

          /* move up */
          ++idxstack[depth];
          --depth;
          break;
        }
      } /* end DFS */
    } /* end outer slice loop */

    timer_stop(&thds[tid].ttime);
  } /* end omp parallel */
}


static void __csf_mttkrp_leaf(
  csf_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  printf("LEAF");
  /* extract tensor structures */
  idx_t const nmodes = ft->nmodes;
  idx_t const * const * const restrict fp = (idx_t const * const *) ft->fptr;
  idx_t const * const * const restrict fids = (idx_t const * const *) ft->fids;

  idx_t const nfactors = mats[0]->J;

  #pragma omp parallel default(shared)
  {
    int const tid = omp_get_thread_num();
    timer_start(&thds[tid].ttime);

    val_t * mvals[MAX_NMODES];
    val_t * buf[MAX_NMODES];
    idx_t idxstack[MAX_NMODES];

    for(idx_t m=0; m < nmodes; ++m) {
      mvals[m] = mats[ft->dim_perm[m]]->vals;
      /* grab the next row of buf from thds */
      buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    }

    /* foreach outer slice */
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < ft->dims[ft->dim_perm[0]]; ++s) {
      idx_t depth = 0;
      /* push current outer slice */
      idxstack[depth++] = s;

      /* clear out stale data */
      for(idx_t m=1; m < ft->nmodes-1; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* fill first bufs */
      val_t * const rootbuf = buf[0];
      val_t * const nextbuf = buf[1];
      val_t const * const restrict rootrow = mvals[0] + (s*nfactors);
      val_t const * const restrict nextrow
          = mvals[1] + (fids[1][idxstack[1]]*nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        rootbuf[f] = rootrow[f];
        nextbuf[f] = rootbuf[f] * nextrow[f];
      }

      idx_t const outer_end = fp[0][s+1];
      while(idxstack[1] < outer_end) {
        /* last node before nonzeros, handle those quickly */
        if(depth == nmodes - 2) {
          /* process all nonzeros [start, end) */
          idx_t const start = fp[depth][idxstack[depth]];
          idx_t const end   = fp[depth][idxstack[depth]+1];

          __csf_process_fiber_lock(mats[MAX_NMODES]->vals, buf[depth],
              nfactors, start, end, fids[depth+1], ft->vals);

          ++idxstack[depth];
          --depth;
          continue;
        }

        /* Node is internal. */
        /* Are all children processed? */
        if(idxstack[depth+1] == fp[depth][idxstack[depth]+1]) {
          ++idxstack[depth];
          --depth;
        } else {
          /* No, move to next child */
          ++depth;
          /* propogate buf down */
          val_t const * const restrict drow
              = mvals[depth] + (fids[depth][idxstack[depth]] * nfactors);
          __assign_hada(buf[depth], buf[depth-1], drow, nfactors);
        }
      } /* end DFS */
    } /* end outer slice loop */

    timer_stop(&thds[tid].ttime);
  } /* end omp parallel */
}


static void __csf_mttkrp_root(
  csf_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  printf("ROOT");
  /* extract tensor structures */
  idx_t const nmodes = ft->nmodes;
  idx_t const * const * const restrict fp = (idx_t const * const *) ft->fptr;
  idx_t const * const * const restrict fids = (idx_t const * const *) ft->fids;
  val_t const * const vals = ft->vals;

  idx_t const nfactors = mats[0]->J;

  #pragma omp parallel default(shared)
  {
    int const tid = omp_get_thread_num();
    timer_start(&thds[tid].ttime);

    val_t * mvals[MAX_NMODES];
    val_t * buf[MAX_NMODES];
    idx_t idxstack[MAX_NMODES];

    for(idx_t m=0; m < nmodes; ++m) {
      mvals[m] = mats[ft->dim_perm[m]]->vals;
      /* grab the next row of buf from thds */
      buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
      memset(buf[m], 0, nfactors * sizeof(idx_t));
    }
    val_t * const ovals = mats[MAX_NMODES]->vals;

    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < ft->dims[ft->dim_perm[0]]; ++s) {
      idx_t depth = 0;
      /* push current outer slice */
      idxstack[depth++] = s;

      /* clear out stale data */
      for(idx_t m=1; m < ft->nmodes-1; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      idx_t const outer_end = fp[0][s+1];
      while(idxstack[1] < outer_end) {
        /* last node before nonzeros, handle those quickly */
        if(depth == nmodes - 2) {
          /* process all nonzeros [start, end) */
          idx_t const start = fp[depth][idxstack[depth]];
          idx_t const end   = fp[depth][idxstack[depth]+1];

          __csf_process_fiber(buf[depth], nfactors, mvals[depth+1],
              start, end, fids[depth+1], vals);

          /* propagate results up the tree */
          val_t const * const restrict fibrow
              = mvals[depth] + (fids[depth][idxstack[depth]] * nfactors);
          __add_hada_clear(buf[depth-1], buf[depth], fibrow, nfactors);

          ++idxstack[depth];
          --depth;
          continue;
        }

        /* Node is internal. */
        /* Are all children processed? */
        if(idxstack[depth+1] == fp[depth][idxstack[depth]+1]) {
          /* propagate up and clear buffer for next sibling */
          val_t const * const restrict myrow =
              mvals[depth] + (nfactors * fids[depth][idxstack[depth]]);
          __add_hada_clear(buf[depth-1], buf[depth], myrow, nfactors);

          ++idxstack[depth];
          --depth;
        } else {
          /* No, move to next child */
          ++depth;
        }
      } /* end DFS */

      /* flush buffer to matrix row */
      val_t * const restrict orow = ovals + (nfactors * s);
      val_t * const restrict brow = buf[0];
      for(idx_t f=0; f < nfactors; ++f) {
        orow[f] = brow[f];
        brow[f] = 0;
      }
    } /* end outer slice loop */

    timer_stop(&thds[tid].ttime);
  } /* end omp parallel */
}

static void __csf_mttkrp_leaf2(
  csf_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  printf("LEAF2");
  /* extract tensor structures */
  idx_t const nmodes = ft->nmodes;
  idx_t const * const * const restrict fp = (idx_t const * const *) ft->fptr;
  idx_t const * const * const restrict fids = (idx_t const * const *) ft->fids;

  idx_t const nfactors = mats[0]->J;

  #pragma omp parallel default(shared)
  {
    int const tid = omp_get_thread_num();
    timer_start(&thds[tid].ttime);

    val_t * mvals[MAX_NMODES];
    val_t * buf[MAX_NMODES];
    idx_t idxstack[MAX_NMODES];

    for(idx_t m=0; m < nmodes; ++m) {
      mvals[m] = mats[ft->dim_perm[m]]->vals;
      /* grab the next row of buf from thds */
      buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    }

    /* foreach outer slice */
    #pragma omp for schedule(dynamic, 16)
    for(idx_t s=0; s < ft->dims[ft->dim_perm[0]]; ++s) {
      /* push current outer slice */
      idxstack[0] = s;

      /* clear out stale data */
      for(idx_t m=1; m < ft->nmodes-1; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* first buf will always just be a matrix row */
      val_t const * const restrict rootrow = mvals[0] + (s*nfactors);
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
          __assign_hada(buf[depth+1], buf[depth], drow, nfactors);
        }

        /* process all nonzeros [start, end) */
        idx_t const start = fp[depth][idxstack[depth]];
        idx_t const end   = fp[depth][idxstack[depth]+1];
        __csf_process_fiber_lock(mats[MAX_NMODES]->vals, buf[depth],
            nfactors, start, end, fids[depth+1], ft->vals);

        /* now move back up to the next unprocessed child */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */
    } /* end outer slice loop */
  } /* end omp parallel */
}


static void __csf_mttkrp_root2(
  csf_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  printf("ROOT2");
  /* extract tensor structures */
  idx_t const nmodes = ft->nmodes;
  idx_t const * const * const restrict fp = (idx_t const * const *) ft->fptr;
  idx_t const * const * const restrict fids = (idx_t const * const *) ft->fids;
  val_t const * const vals = ft->vals;

  idx_t const nfactors = mats[0]->J;

  #pragma omp parallel default(shared)
  {
    int const tid = omp_get_thread_num();
    timer_start(&thds[tid].ttime);

    val_t * mvals[MAX_NMODES];
    val_t * buf[MAX_NMODES];
    idx_t idxstack[MAX_NMODES];

    for(idx_t m=0; m < nmodes; ++m) {
      mvals[m] = mats[ft->dim_perm[m]]->vals;
      /* grab the next row of buf from thds */
      buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
      memset(buf[m], 0, nfactors * sizeof(idx_t));
    }

    val_t * const ovals = mats[MAX_NMODES]->vals;

    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < ft->dims[ft->dim_perm[0]]; ++s) {
      __propagate_up(ovals + (s * nfactors), buf, idxstack, 0, s, fp, fids,
          vals, mvals, nmodes, nfactors);
    } /* end foreach outer slice */

    timer_start(&thds[tid].ttime);
  } /* end omp parallel */
}
#endif


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
int splatt_mttkrp(
    splatt_idx_t const mode,
    splatt_idx_t const ncolumns,
    splatt_csf_t const * const tensor,
    splatt_val_t ** matrices,
    splatt_val_t * const matout,
    double const * const options)
{
  idx_t const nmodes = tensor->nmodes;

  /* fill matrix pointers  */
  matrix_t * mats[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (matrix_t *) malloc(sizeof(matrix_t));
    mats[m]->I = tensor->dims[m];
    mats[m]->J = ncolumns,
    mats[m]->rowmajor = 1;
    mats[m]->vals = matrices[m];
  }
  mats[MAX_NMODES] = (matrix_t *) malloc(sizeof(matrix_t));
  mats[MAX_NMODES]->I = tensor->dims[mode];
  mats[MAX_NMODES]->J = ncolumns;
  mats[MAX_NMODES]->rowmajor = 1;
  mats[MAX_NMODES]->vals = matout;

  /* Setup thread structures. + 64 bytes is to avoid false sharing. */
  idx_t const nthreads = (idx_t) options[SPLATT_OPTION_NTHREADS];
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 2,
    (ncolumns * ncolumns * sizeof(val_t)) + 64,
    TILE_SIZES[0] * ncolumns * sizeof(val_t) + 64);

  /* do the MTTKRP */
  mttkrp_splatt(tensor, mats, mode, thds, nthreads);

  /* cleanup */
  thd_free(thds, nthreads);
  for(idx_t m=0; m < nmodes; ++m) {
    free(mats[m]);
  }
  free(mats[MAX_NMODES]);

  return SPLATT_SUCCESS;
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mttkrp_ctensor(
  ctensor_t const * const ct,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  /* clear output matrix */
  matrix_t * const M = mats[MAX_NMODES];
  M->I = ct->dims[mode];
  memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  /* find out which level in the tree this is */
  idx_t outdepth = 0;
  for(idx_t m=0; m < ct->nmodes; ++m) {
    if(ct->dim_perm[m] == mode) {
      outdepth = m;
      break;
    }
  }

  omp_set_num_threads(nthreads);
  if(outdepth == 0) {
    __ctensor_mttkrp_root(ct, 0, mats, mode, thds);
  }
}


/******************************************************************************
 * SPLATT MTTKRP
 *****************************************************************************/
#if 0
void mttkrp_csf(
  csf_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  /* clear output matrix */
  matrix_t * const M = mats[MAX_NMODES];
  M->I = ft->dims[mode];
  memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  /* find out which level in the tree this is */
  idx_t outdepth = 0;
  for(idx_t m=0; m < ft->nmodes; ++m) {
    if(ft->dim_perm[m] == mode) {
      outdepth = m;
      break;
    }
  }

  omp_set_num_threads(nthreads);
  if(outdepth == 0) {
    __csf_mttkrp_root2(ft, mats, mode, thds);
  } else if(outdepth == ft->nmodes-1) {
    __csf_mttkrp_leaf2(ft, mats, mode, thds);
  } else {
    __csf_mttkrp_internal2(ft, mats, mode, thds);
  }
}
#endif




void mttkrp_splatt(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  if(ft->tiled == SPLATT_COOPTILE) {
    mttkrp_splatt_coop_tiled(ft, mats, mode, thds, nthreads);
    return;
  }

  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];
  idx_t const nslices = ft->dims[mode];
  idx_t const rank = M->J;

  M->I = nslices;
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
    int const tid = omp_get_thread_num();
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


void mttkrp_splatt_tiled(
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
    int const tid = omp_get_thread_num();
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
    int const tid = omp_get_thread_num();
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
      //scratch[x] = vals[x] * A->vals[r + (rank*indA[x])] * B->vals[r + (rank*indB[x])];
    }

    /* now accumulate into m1 */
    for(idx_t x=0; x < nnz; ++x) {
      mv[indM[x]] += scratch[x];
      //M->vals[r + (rank * indM[x])] += scratch[x];
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

  val_t * accum = (val_t *) malloc(nfactors * sizeof(val_t));

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


