
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
#define MEASURE_DRIFT 0


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
    idx_t const * const part,
    val_t const mult)
{
  idx_t const nfactors = model->rank;

  idx_t const I = model->dims[csf->dim_perm[0]];
  idx_t const J = model->dims[csf->dim_perm[1]];
  idx_t const K = model->dims[csf->dim_perm[2]];
  val_t const * const restrict avals = model->factors[csf->dim_perm[0]]+(f*I);
  val_t const * const restrict bvals = model->factors[csf->dim_perm[1]]+(f*J);
  val_t const * const restrict cvals = model->factors[csf->dim_perm[2]]+(f*K);

  val_t myloss = 0;

  int const tid = omp_get_thread_num();

  /* update residual in parallel */
  for(idx_t tile=part[tid]; tile < part[tid+1]; ++tile) {
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

  #pragma omp barrier

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
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;

  GRAB_SPARSITY(tile)
  GRAB_CONST_FACTORS

  val_t * const restrict numer = ws->numerator;
  val_t * const restrict denom = ws->denominator;

  for(idx_t i=0; i < pt->nfibs[0]; ++i) {
    idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];

    /* grab the top-level row to update */
    val_t const aval = avals[a_id];

    /* process each fiber */
    for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
      val_t const bval = bvals[fids[fib]];

      /* push Hadmard product down tree */
      val_t const predict = aval * bval;

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj) {
        val_t const cval = cvals[inds[jj]];

        val_t const sgrad = bval * cval;
        //numer[a_id] += (residual[jj] + (predict * cval)) * sgrad;
        //#pragma omp atomic
        numer[a_id] += residual[jj] * bval * cval;
        //#pragma omp atomic
        denom[a_id] += sgrad * sgrad;
      }
    } /* foreach fiber */
  } /* foreach slice */
}


static void p_process_intl3(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const f,
    tc_model const * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;

  GRAB_SPARSITY(tile)
  GRAB_CONST_FACTORS

  val_t * const restrict numer = ws->numerator;
  val_t * const restrict denom = ws->denominator;

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
        //numer[b_id] += (residual[jj] + (predict * cval)) * sgrad;
        //#pragma omp atomic
        numer[b_id] += residual[jj] * sgrad;
        //#pragma omp atomic
        denom[b_id] += sgrad * sgrad;
      }
    } /* foreach fiber */
  } /* foreach slice */
}


static void p_process_leaf3(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const f,
    tc_model const * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;

  GRAB_SPARSITY(tile)
  GRAB_CONST_FACTORS

  val_t * const restrict numer = ws->numerator;
  val_t * const restrict denom = ws->denominator;

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
        //numer[c_id] += (residual[jj] + (predict * cval)) * sgrad;
        //#pragma omp atomic
        numer[c_id] += residual[jj] * predict;
        //#pragma omp atomic
        denom[c_id] += sgrad * sgrad;
      }
    } /* foreach fiber */
  } /* foreach slice */
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
  /* convert training data to CSF-ONEMODE with full tiling */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS] = ws->nthreads;
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  opts[SPLATT_OPTION_TILEDEPTH] = 1;

  splatt_csf * csf = csf_alloc(train, opts);

  idx_t * residual_part = csf_partition_tiles_1d(csf, ws->nthreads);

  printf("ntiles: %lu\n", csf->ntiles);

  idx_t const nfactors = model->rank;

  /* initialize residual */
  #pragma omp parallel
  {
    for(idx_t f=0; f < nfactors; ++f) {
      p_update_residual3(csf, f, model, ws, residual_part, -1);
    }
  }

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  timer_start(&ws->tc_time);

  idx_t const nmodes = csf->nmodes;

  val_t * const restrict numer = ws->numerator;
  val_t * const restrict denom = ws->denominator;

  /* foreach epoch */
  for(idx_t e=1; e < ws->max_its+1; ++e) {
    /* update model from all training observations */

    loss = 0;
    #pragma omp parallel reduction(+:loss)
    {
      int const tid = omp_get_thread_num();

      for(idx_t f=0; f < nfactors; ++f) {
        /* add current component to residual */
        p_update_residual3(csf, f, model, ws, residual_part, 1);

        for(idx_t inner=0; inner < ws->num_inner; ++inner) {

          /* compute new column 'f' for each factor */
          for(idx_t m=0; m < nmodes; ++m) {

            idx_t const dim = model->dims[m];

            /* initialize numerator/denominator */
            #pragma omp for schedule(static)
            for(idx_t i=0; i < dim; ++i) {
              numer[i] = 0;
              denom[i] = ws->regularization[m];
            }

            /* which routine to call? */
            node_type const which = which_depth(csf, m);

            /* Compute numerator/denominator. Distribute tile layer to threads
             *  to avoid locks. */
#if 0
            #pragma omp for schedule(static, 1)
            for(idx_t t=0; t < csf->tile_dims[m]; ++t) {
              idx_t tile = get_next_tileid(TILE_BEGIN, csf->tile_dims, nmodes,
                  m, t);
              while(tile != TILE_END) {
            #pragma omp for
            for(idx_t tile=0; tile < csf->ntiles; ++tile) {
#endif
            for(idx_t tile=residual_part[tid]; tile < residual_part[tid+1]; ++tile) {
                /* process tile */
                switch(which) {
                case NODE_ROOT:
                  p_process_root3(csf, tile, f, model, ws);
                  break;
                case NODE_INTL:
                  p_process_intl3(csf, tile, f, model, ws);
                  break;
                case NODE_LEAF:
                  p_process_leaf3(csf, tile, f, model, ws);
                  break;
                }

                /* move on to text tile in my layer */
#if 0
                tile = get_next_tileid(tile, csf->tile_dims, nmodes, m, t);
              }
#endif
            } /* foreach tile */
            #pragma omp barrier

            /* numerator/denominator are now computed; update factor column */
            val_t * const restrict avals = model->factors[m] + (f * dim);
            #pragma omp for schedule(static)
            for(idx_t i=0; i < dim; ++i) {
              avals[i] = numer[i] / denom[i];
            }

          } /* foreach mode */
        } /* foreach inner iteration */

        /* subtract new rank-1 factor from residual */
        loss = p_update_residual3(csf, f, model, ws, residual_part, -1);

      } /* foreach factor */
    } /* omp parallel */


#if MEASURE_DRIFT == 1
    val_t const gold = tc_loss_sq(train, model, ws);
    printf("  residual: %e actual: %e diff: %e\n", loss, gold, loss - gold);
#endif

    /* compute RMSE and adjust learning rate */
    frobsq = tc_frob_sq(model, ws);
    if(tc_converge(train, validate, model, loss, frobsq, e, ws)) {
      break;
    }

  } /* foreach epoch */

  /* cleanup */
  splatt_free(residual_part);
  csf_free(csf, opts);
}


