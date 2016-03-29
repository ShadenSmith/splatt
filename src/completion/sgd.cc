

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "completion.h"
#include "../csf.h"
#include "../reorder.h"
#include "../util.h"
#include "../io.h"
#include "../sort.h"

#include <cmath>
#include <omp.h>
#include <algorithm>
#include <set>
#include <map>
#include <unordered_map>
#include <vector>
#include <sstream>

using std::set;
using std::map;
using std::unordered_map;
using std::vector;
using std::stringstream;

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Update a three-mode model based on a given observation.
*
* @param train The training data.
* @param nnz_index The index of the observation to update from.
* @param model The model to update.
* @param ws Workspace to use.
*/
static inline void p_update_model3(
    sptensor_t const * const train,
    idx_t const nnz_index,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  idx_t const x = nnz_index;

  assert(train->nmodes == 3);

  idx_t * * const ind = train->ind;
  assert(ind[0][x] < model->dims[0]);
  assert(ind[1][x] < model->dims[1]);
  assert(ind[2][x] < model->dims[2]);
  val_t * const restrict arow = model->factors[0] + (ind[0][x] * nfactors);
  val_t * const restrict brow = model->factors[1] + (ind[1][x] * nfactors);
  val_t * const restrict crow = model->factors[2] + (ind[2][x] * nfactors);

  /* predict value */
  val_t predicted = 0;
  for(idx_t f=0; f < nfactors; ++f) {
    assert(std::isfinite(arow[f]));
    assert(std::isfinite(brow[f]));
    assert(std::isfinite(crow[f]));
    predicted += arow[f] * brow[f] * crow[f];
    assert(std::isfinite(predicted));
  }
  val_t const loss = train->vals[x] - predicted;

  val_t const rate = ws->learn_rate;
  val_t const * const restrict reg = ws->regularization;

  /* update rows */
  for(idx_t f=0; f < nfactors; ++f) {
    val_t a = loss*arow[f];
    val_t const moda = (loss * brow[f] * crow[f]) - (reg[0] * arow[f]);
    val_t const modb = (a * crow[f]) - (reg[1] * brow[f]);
    val_t const modc = (a * brow[f]) - (reg[2] * crow[f]);
    arow[f] += rate * moda;
    assert(std::isfinite(arow[f]));
    brow[f] += rate * modb;
    assert(std::isfinite(brow[f]));
    crow[f] += rate * modc;
    assert(std::isfinite(crow[f]));
  }
}

static inline void p_update_model4(
    sptensor_t const * const train,
    idx_t const nnz_index,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  idx_t const x = nnz_index;

  assert(train->nmodes == 4);

  idx_t * * const ind = train->ind;
  assert(ind[0][x] < model->dims[0]);
  assert(ind[1][x] < model->dims[1]);
  assert(ind[2][x] < model->dims[2]);
  assert(ind[3][x] < model->dims[3]);
  val_t * const restrict arow = model->factors[0] + (ind[0][x] * nfactors);
  val_t * const restrict brow = model->factors[1] + (ind[1][x] * nfactors);
  val_t * const restrict crow = model->factors[2] + (ind[2][x] * nfactors);
  val_t * const restrict drow = model->factors[3] + (ind[3][x] * nfactors);

  /* predict value */
  val_t predicted = 0;
  for(idx_t f=0; f < nfactors; ++f) {
    assert(std::isfinite(arow[f]));
    assert(std::isfinite(brow[f]));
    assert(std::isfinite(crow[f]));
    assert(std::isfinite(drow[f]));
    predicted += arow[f] * brow[f] * crow[f] * drow[f];
    assert(std::isfinite(predicted));
  }
  val_t const loss = train->vals[x] - predicted;

  val_t const rate = ws->learn_rate;
  val_t const * const restrict reg = ws->regularization;

  /* update rows */
  for(idx_t f=0; f < nfactors; ++f) {
    /* 12 multiplications -> 8 */
    double ab = loss*arow[f]*brow[f];
    double cd = loss*crow[f]*drow[f];
    val_t const moda = (brow[f] * cd) - (reg[0] * arow[f]);
    val_t const modb = (arow[f] * cd) - (reg[1] * brow[f]);
    val_t const modc = (ab * drow[f]) - (reg[2] * crow[f]);
    val_t const modd = (ab * crow[f]) - (reg[3] * drow[f]);
    arow[f] += rate * moda;
    assert(std::isfinite(arow[f]));
    brow[f] += rate * modb;
    assert(std::isfinite(brow[f]));
    crow[f] += rate * modc;
    assert(std::isfinite(crow[f]));
    drow[f] += rate * modd;
    assert(std::isfinite(drow[f]));
  }
}

static inline void p_update_model5(
    sptensor_t const * const train,
    idx_t const nnz_index,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  idx_t const x = nnz_index;

  assert(train->nmodes == 5);

  idx_t * * const ind = train->ind;
  assert(ind[0][x] < model->dims[0]);
  assert(ind[1][x] < model->dims[1]);
  assert(ind[2][x] < model->dims[2]);
  assert(ind[3][x] < model->dims[3]);
  assert(ind[4][x] < model->dims[4]);
  val_t * const restrict arow = model->factors[0] + (ind[0][x] * nfactors);
  val_t * const restrict brow = model->factors[1] + (ind[1][x] * nfactors);
  val_t * const restrict crow = model->factors[2] + (ind[2][x] * nfactors);
  val_t * const restrict drow = model->factors[3] + (ind[3][x] * nfactors);
  val_t * const restrict erow = model->factors[4] + (ind[4][x] * nfactors);

  /* predict value */
  val_t predicted = 0;
  for(idx_t f=0; f < nfactors; ++f) {
    assert(std::isfinite(arow[f]));
    assert(std::isfinite(brow[f]));
    assert(std::isfinite(crow[f]));
    assert(std::isfinite(drow[f]));
    assert(std::isfinite(erow[f]));
    predicted += arow[f] * brow[f] * crow[f] * drow[f] * erow[f];
    assert(std::isfinite(predicted));
  }
  val_t const loss = train->vals[x] - predicted;

  val_t const rate = ws->learn_rate;
  val_t const * const restrict reg = ws->regularization;

  /* update rows */
  for(idx_t f=0; f < nfactors; ++f) {
    /* 20 multiplications -> 11 */
    double ab = loss*arow[f]*brow[f];
    double abc = ab*crow[f];
    double de_temp = drow[f]*erow[f];
    double cde = loss*crow[f]*de_temp;
    val_t const moda = (brow[f] * cde) - (reg[0] * arow[f]);
    val_t const modb = (arow[f] * cde) - (reg[1] * brow[f]);
    val_t const modc = (ab * de_temp) - (reg[2] * crow[f]);
    val_t const modd = (abc * erow[f]) - (reg[3] * drow[f]);
    val_t const mode = (abc * drow[f]) - (reg[4] * erow[f]);
    arow[f] += rate * moda;
    assert(std::isfinite(arow[f]));
    brow[f] += rate * modb;
    assert(std::isfinite(brow[f]));
    crow[f] += rate * modc;
    assert(std::isfinite(crow[f]));
    drow[f] += rate * modd;
    assert(std::isfinite(drow[f]));
    erow[f] += rate * mode;
    assert(std::isfinite(erow[f]));
  }
}

static inline void p_update_model6(
    sptensor_t const * const train,
    idx_t const nnz_index,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  idx_t const x = nnz_index;

  assert(train->nmodes == 6);

  idx_t * * const ind = train->ind;
  assert(ind[0][x] < model->dims[0]);
  assert(ind[1][x] < model->dims[1]);
  assert(ind[2][x] < model->dims[2]);
  assert(ind[3][x] < model->dims[3]);
  assert(ind[4][x] < model->dims[4]);
  assert(ind[5][x] < model->dims[5]);
  val_t * const restrict arow = model->factors[0] + (ind[0][x] * nfactors);
  val_t * const restrict brow = model->factors[1] + (ind[1][x] * nfactors);
  val_t * const restrict crow = model->factors[2] + (ind[2][x] * nfactors);
  val_t * const restrict drow = model->factors[3] + (ind[3][x] * nfactors);
  val_t * const restrict erow = model->factors[4] + (ind[4][x] * nfactors);
  val_t * const restrict frow = model->factors[5] + (ind[5][x] * nfactors);

  /* predict value */
  val_t predicted = 0;
  for(idx_t f=0; f < nfactors; ++f) {
    assert(std::isfinite(arow[f]));
    assert(std::isfinite(brow[f]));
    assert(std::isfinite(crow[f]));
    assert(std::isfinite(drow[f]));
    assert(std::isfinite(erow[f]));
    assert(std::isfinite(frow[f]));
    predicted += arow[f] * brow[f] * crow[f] * drow[f] * erow[f] * frow[f];
    assert(std::isfinite(predicted));
  }
  val_t const loss = train->vals[x] - predicted;

  val_t const rate = ws->learn_rate;
  val_t const * const restrict reg = ws->regularization;

  /* update rows */
  for(idx_t f=0; f < nfactors; ++f) {
    /* in general, we need m*(m-1) multiplications */
    /* m*(m-1) multiplications can be reduced to */
    /* (m/2)*(m/2-1) multiplication followed by m multiplications */
    /* This reduces O(m^2) FLOPS to O(mlogm) */
    /* For a large enough m, we should switch to O(m) method */
    /* with division */
    /* 30 multiplications -> 14 */
    double ab = loss*arow[f]*brow[f];
    double cd = crow[f]*drow[f];
    double ef = erow[f]*frow[f];

    double abcd = ab*cd;
    double abef = ab*ef;
    double cdef = loss*cd*ef;

    val_t const moda = (brow[f] * cdef) - (reg[0] * arow[f]);
    val_t const modb = (arow[f] * cdef) - (reg[1] * brow[f]);
    val_t const modc = (abef * drow[f]) - (reg[2] * crow[f]);
    val_t const modd = (abef * crow[f]) - (reg[3] * drow[f]);
    val_t const mode = (abcd * frow[f]) - (reg[4] * erow[f]);
    val_t const modf = (abcd * erow[f]) - (reg[5] * frow[f]);
    arow[f] += rate * moda;
    assert(std::isfinite(arow[f]));
    brow[f] += rate * modb;
    assert(std::isfinite(brow[f]));
    crow[f] += rate * modc;
    assert(std::isfinite(crow[f]));
    drow[f] += rate * modd;
    assert(std::isfinite(drow[f]));
    erow[f] += rate * mode;
    assert(std::isfinite(erow[f]));
    frow[f] += rate * modf;
    assert(std::isfinite(frow[f]));
  }
}

template<int NFACTORS>
static inline void p_update_model_csf3_(
    splatt_csf const * const train,
    idx_t const i,
    tc_model * const model,
    tc_ws * const ws)
{
  csf_sparsity const * const pt = train->pt;
  assert(model->nmodes == 3);
  assert(train->ntiles == 1);

  /* sparsity structure */
  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr = pt->fptr[1];
  idx_t const * const restrict fids = pt->fids[1];
  idx_t const * const restrict inds = pt->fids[2];

  /* current model */
  val_t const * const restrict vals = pt->vals;
  val_t * const restrict avals = model->factors[train->dim_perm[0]];
  val_t * const restrict bvals = model->factors[train->dim_perm[1]];
  val_t * const restrict cvals = model->factors[train->dim_perm[2]];


  val_t const rate = ws->learn_rate;
  val_t const areg = ws->regularization[train->dim_perm[0]];
  val_t const breg = ws->regularization[train->dim_perm[1]];
  val_t const creg = ws->regularization[train->dim_perm[2]];

  /* grab the top-level row */
  idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
  val_t * const restrict arow = avals + (a_id * NFACTORS);

  /* process each fiber */
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
    val_t * const restrict brow = bvals + (fids[fib] * NFACTORS);

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj) {
      val_t * const restrict crow = cvals + (inds[jj] * NFACTORS);

      /* compute the loss */
      val_t loss = vals[jj];
      for(idx_t f=0; f < NFACTORS; ++f) {
        loss -= arow[f] * brow[f] * crow[f];
      }

      /* update model */
      for(idx_t f=0; f < NFACTORS; ++f) {
        /* compute all modifications FIRST since we are updating all rows */
        double a = loss*arow[f];
        val_t const moda = (loss * brow[f] * crow[f]) - (areg * arow[f]);
        val_t const modb = (a * crow[f]) - (breg * brow[f]);
        val_t const modc = (a * brow[f]) - (creg * crow[f]);
        arow[f] += rate * moda;
        brow[f] += rate * modb;
        crow[f] += rate * modc;
      }
    }
  } /* foreach fiber */
}

/**
* @brief Update a three-mode model based on the i-th node of a CSF tensor.
*
* @param train The training data (in CSf format).
* @param i Which node to process.
* @param model The model to update.
* @param ws Workspace to use.
*/
static inline void p_update_model_csf3(
    splatt_csf const * const train,
    idx_t const i,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  csf_sparsity const * const pt = train->pt;
  assert(model->nmodes == 3);
  assert(train->ntiles == 1);

  /* sparsity structure */
  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr = pt->fptr[1];
  idx_t const * const restrict fids = pt->fids[1];
  idx_t const * const restrict inds = pt->fids[2];

  /* current model */
  val_t const * const restrict vals = pt->vals;
  val_t * const restrict avals = model->factors[train->dim_perm[0]];
  val_t * const restrict bvals = model->factors[train->dim_perm[1]];
  val_t * const restrict cvals = model->factors[train->dim_perm[2]];


  val_t const rate = ws->learn_rate;
  val_t const areg = ws->regularization[train->dim_perm[0]];
  val_t const breg = ws->regularization[train->dim_perm[1]];
  val_t const creg = ws->regularization[train->dim_perm[2]];

  /* grab the top-level row */
  idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
  val_t * const restrict arow = avals + (a_id * nfactors);

  /* process each fiber */
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
    val_t * const restrict brow = bvals + (fids[fib] * nfactors);

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj) {
      val_t * const restrict crow = cvals + (inds[jj] * nfactors);

      /* compute the loss */
      val_t loss = vals[jj];
      for(idx_t f=0; f < nfactors; ++f) {
        loss -= arow[f] * brow[f] * crow[f];
      }

      /* update model */
      for(idx_t f=0; f < nfactors; ++f) {
        /* compute all modifications FIRST since we are updating all rows */
        double a = loss*arow[f];
        val_t const moda = (loss * brow[f] * crow[f]) - (areg * arow[f]);
        val_t const modb = (a * crow[f]) - (breg * brow[f]);
        val_t const modc = (a * brow[f]) - (creg * crow[f]);
        arow[f] += rate * moda;
        brow[f] += rate * modb;
        crow[f] += rate * modc;
      }
    }
  } /* foreach fiber */
}

static inline void p_update_model_csf4(
    splatt_csf const * const train,
    idx_t const i,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  csf_sparsity const * const pt = train->pt;
  assert(model->nmodes == 4);
  assert(train->ntiles == 1);

  /* sparsity structure */
  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr1 = pt->fptr[1];
  idx_t const * const restrict fptr2 = pt->fptr[2];
  idx_t const * const restrict fids1 = pt->fids[1];
  idx_t const * const restrict fids2 = pt->fids[2];
  idx_t const * const restrict inds = pt->fids[3];

  /* current model */
  val_t const * const restrict vals = pt->vals;
  val_t * const restrict avals = model->factors[train->dim_perm[0]];
  val_t * const restrict bvals = model->factors[train->dim_perm[1]];
  val_t * const restrict cvals = model->factors[train->dim_perm[2]];
  val_t * const restrict dvals = model->factors[train->dim_perm[3]];


  val_t const rate = ws->learn_rate;
  val_t const areg = ws->regularization[train->dim_perm[0]];
  val_t const breg = ws->regularization[train->dim_perm[1]];
  val_t const creg = ws->regularization[train->dim_perm[2]];
  val_t const dreg = ws->regularization[train->dim_perm[3]];

  /* grab the top-level row */
  idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
  val_t * const restrict arow = avals + (a_id * nfactors);

  /* process each fiber */
  for(idx_t fib1=sptr[i]; fib1 < sptr[i+1]; ++fib1) {
    val_t * const restrict brow = bvals + (fids1[fib1] * nfactors);

    for(idx_t fib2=fptr1[fib1]; fib2 < fptr1[fib1+1]; ++fib2) {
      val_t * const restrict crow = cvals + (fids2[fib2] * nfactors);

      /* foreach nnz in fiber */
      for(idx_t jj=fptr2[fib2]; jj < fptr2[fib2+1]; ++jj) {
        val_t * const restrict drow = dvals + (inds[jj] * nfactors);

        /* compute the loss */
        val_t loss = vals[jj];
        for(idx_t f=0; f < nfactors; ++f) {
          loss -= arow[f] * brow[f] * crow[f] * drow[f];
        }

        /* update model */
        for(idx_t f=0; f < nfactors; ++f) {
          /* compute all modifications FIRST since we are updating all rows */
          double ab = loss*arow[f]*brow[f];
          double cd = loss*crow[f]*drow[f];
          val_t const moda = (brow[f] * cd) - (areg * arow[f]);
          val_t const modb = (arow[f] * cd) - (breg * brow[f]);
          val_t const modc = (ab * drow[f]) - (creg * crow[f]);
          val_t const modd = (ab * crow[f]) - (dreg * drow[f]);
          arow[f] += rate * moda;
          brow[f] += rate * modb;
          crow[f] += rate * modc;
          drow[f] += rate * modd;
        }
      }
    }
  } /* foreach fiber */
}

static inline void p_update_model_csf5(
    splatt_csf const * const train,
    idx_t const i,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  csf_sparsity const * const pt = train->pt;
  assert(model->nmodes == 5);
  assert(train->ntiles == 1);

  /* sparsity structure */
  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr1 = pt->fptr[1];
  idx_t const * const restrict fptr2 = pt->fptr[2];
  idx_t const * const restrict fptr3 = pt->fptr[3];
  idx_t const * const restrict fids1 = pt->fids[1];
  idx_t const * const restrict fids2 = pt->fids[2];
  idx_t const * const restrict fids3 = pt->fids[3];
  idx_t const * const restrict inds = pt->fids[4];

  /* current model */
  val_t const * const restrict vals = pt->vals;
  val_t * const restrict avals = model->factors[train->dim_perm[0]];
  val_t * const restrict bvals = model->factors[train->dim_perm[1]];
  val_t * const restrict cvals = model->factors[train->dim_perm[2]];
  val_t * const restrict dvals = model->factors[train->dim_perm[3]];
  val_t * const restrict evals = model->factors[train->dim_perm[4]];


  val_t const rate = ws->learn_rate;
  val_t const areg = ws->regularization[train->dim_perm[0]];
  val_t const breg = ws->regularization[train->dim_perm[1]];
  val_t const creg = ws->regularization[train->dim_perm[2]];
  val_t const dreg = ws->regularization[train->dim_perm[3]];
  val_t const ereg = ws->regularization[train->dim_perm[4]];

  /* grab the top-level row */
  idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
  val_t * const restrict arow = avals + (a_id * nfactors);

  /* process each fiber */
  for(idx_t fib1=sptr[i]; fib1 < sptr[i+1]; ++fib1) {
    val_t * const restrict brow = bvals + (fids1[fib1] * nfactors);

    for(idx_t fib2=fptr1[fib1]; fib2 < fptr1[fib1+1]; ++fib2) {
      val_t * const restrict crow = cvals + (fids2[fib2] * nfactors);

      for(idx_t fib3=fptr2[fib2]; fib3 < fptr2[fib2+1]; ++fib3) {
        val_t * const restrict drow = dvals + (fids3[fib3] * nfactors);

        /* foreach nnz in fiber */
        for(idx_t jj=fptr3[fib3]; jj < fptr3[fib3+1]; ++jj) {
          val_t * const restrict erow = evals + (inds[jj] * nfactors);

          /* compute the loss */
          val_t loss = vals[jj];
          for(idx_t f=0; f < nfactors; ++f) {
            loss -= arow[f] * brow[f] * crow[f] * drow[f] * erow[f];
          }

          /* update model */
          for(idx_t f=0; f < nfactors; ++f) {
            /* compute all modifications FIRST since we are updating all rows */
            double ab = loss*arow[f]*brow[f];
            double abc = ab*crow[f];
            double de_temp = drow[f]*erow[f];
            double cde = loss*crow[f]*de_temp;
            val_t const moda = (brow[f] * cde) - (areg * arow[f]);
            val_t const modb = (arow[f] * cde) - (breg * brow[f]);
            val_t const modc = (ab * de_temp) - (creg * crow[f]);
            val_t const modd = (abc * erow[f]) - (dreg * drow[f]);
            val_t const mode = (abc * drow[f]) - (dreg * erow[f]);
            arow[f] += rate * moda;
            brow[f] += rate * modb;
            crow[f] += rate * modc;
            drow[f] += rate * modd;
            erow[f] += rate * mode;
          }
        }
      }
    }
  } /* foreach fiber */
}

static inline void p_update_model_csf6(
    splatt_csf const * const train,
    idx_t const i,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  csf_sparsity const * const pt = train->pt;
  assert(model->nmodes == 5);
  assert(train->ntiles == 1);

  /* sparsity structure */
  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr1 = pt->fptr[1];
  idx_t const * const restrict fptr2 = pt->fptr[2];
  idx_t const * const restrict fptr3 = pt->fptr[3];
  idx_t const * const restrict fptr4 = pt->fptr[4];
  idx_t const * const restrict fids1 = pt->fids[1];
  idx_t const * const restrict fids2 = pt->fids[2];
  idx_t const * const restrict fids3 = pt->fids[3];
  idx_t const * const restrict fids4 = pt->fids[4];
  idx_t const * const restrict inds = pt->fids[5];

  /* current model */
  val_t const * const restrict vals = pt->vals;
  val_t * const restrict avals = model->factors[train->dim_perm[0]];
  val_t * const restrict bvals = model->factors[train->dim_perm[1]];
  val_t * const restrict cvals = model->factors[train->dim_perm[2]];
  val_t * const restrict dvals = model->factors[train->dim_perm[3]];
  val_t * const restrict evals = model->factors[train->dim_perm[4]];
  val_t * const restrict fvals = model->factors[train->dim_perm[5]];


  val_t const rate = ws->learn_rate;
  val_t const areg = ws->regularization[train->dim_perm[0]];
  val_t const breg = ws->regularization[train->dim_perm[1]];
  val_t const creg = ws->regularization[train->dim_perm[2]];
  val_t const dreg = ws->regularization[train->dim_perm[3]];
  val_t const ereg = ws->regularization[train->dim_perm[4]];
  val_t const freg = ws->regularization[train->dim_perm[5]];

  /* grab the top-level row */
  idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
  val_t * const restrict arow = avals + (a_id * nfactors);

  /* process each fiber */
  for(idx_t fib1=sptr[i]; fib1 < sptr[i+1]; ++fib1) {
    val_t * const restrict brow = bvals + (fids1[fib1] * nfactors);

    for(idx_t fib2=fptr1[fib1]; fib2 < fptr1[fib1+1]; ++fib2) {
      val_t * const restrict crow = cvals + (fids2[fib2] * nfactors);

      for(idx_t fib3=fptr2[fib2]; fib3 < fptr2[fib2+1]; ++fib3) {
        val_t * const restrict drow = dvals + (fids3[fib3] * nfactors);

        for(idx_t fib4=fptr3[fib3]; fib4 < fptr3[fib3+1]; ++fib4) {
          val_t * const restrict erow = evals + (fids4[fib4] * nfactors);

          /* foreach nnz in fiber */
          for(idx_t jj=fptr4[fib4]; jj < fptr4[fib4+1]; ++jj) {
            val_t * const restrict frow = evals + (inds[jj] * nfactors);

            /* compute the loss */
            val_t loss = vals[jj];
            for(idx_t f=0; f < nfactors; ++f) {
              loss -= arow[f] * brow[f] * crow[f] * drow[f] * erow[f] * frow[f];
            }

            /* update model */
            for(idx_t f=0; f < nfactors; ++f) {
              /* compute all modifications FIRST since we are updating all rows */

              double ab = loss*arow[f]*brow[f];
              double cd = crow[f]*drow[f];
              double ef = erow[f]*frow[f];

              double abcd = ab*cd;
              double abef = ab*ef;
              double cdef = loss*cd*ef;

              val_t const moda = (brow[f] * cdef) - (areg * arow[f]);
              val_t const modb = (arow[f] * cdef) - (breg * brow[f]);
              val_t const modc = (abef * drow[f]) - (creg * crow[f]);
              val_t const modd = (abef * crow[f]) - (dreg * drow[f]);
              val_t const mode = (abcd * frow[f]) - (dreg * erow[f]);
              val_t const modf = (abcd * erow[f]) - (dreg * frow[f]);
              arow[f] += rate * moda;
              brow[f] += rate * modb;
              crow[f] += rate * modc;
              drow[f] += rate * modd;
              erow[f] += rate * mode;
              frow[f] += rate * modf;
            }
          }
        }
      }
    }
  } /* foreach fiber */
}

/**
* @brief Update a model based on a given observation.
*
* @param train The training data.
* @param nnz_index The index of the observation to update from.
* @param model The model to update.
* @param ws Workspace to use.
*/
static void p_update_model(
    sptensor_t const * const train,
    idx_t const nnz_index,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nmodes = train->nmodes;
  if(nmodes == 3) {
    p_update_model3(train, nnz_index, model, ws);
    return;
  }

  idx_t const nfactors = model->rank;
  idx_t const x = nnz_index;

  val_t * const restrict buffer = (val_t *)ws->thds[omp_get_thread_num()].scratch[0];

  /* compute the error */
  val_t const err = train->vals[x] - tc_predict_val(model, train, x, buffer);

  idx_t * * const ind = train->ind;

  /* update each of the factor (row-wise) */
  for(idx_t m=0; m < nmodes; ++m) {

    /* first fill buffer with the Hadamard product of all rows but current */
    idx_t moff = (m + 1) % nmodes;
    val_t const * const restrict init_row = model->factors[moff] +
        (ind[moff][x] * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buffer[f] = init_row[f];
    }
    for(moff = 2; moff < nmodes; ++moff) {
      idx_t const madj = (m + moff) % nmodes;
      val_t const * const restrict row = model->factors[madj] +
          (ind[madj][x] * nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        buffer[f] *= row[f];
      }
    }

    /* now actually update the row */
    val_t * const restrict update_row = model->factors[m] +
        (ind[m][x] * nfactors);
    val_t const reg = ws->regularization[m];
    val_t const rate = ws->learn_rate;
    for(idx_t f=0; f < nfactors; ++f) {
      update_row[f] += rate * ((err * buffer[f]) - (reg * update_row[f]));
    }
  }
}





/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#ifdef SPLATT_USE_MPI

extern "C" {

// defined in mpi/mpi_io.c
void p_find_layer_boundaries(
  idx_t ** const ssizes,
  idx_t const mode,
  rank_info * const rinfo);

// defined in mpi/mpi_io.c
void p_fill_ssizes(
  sptensor_t const * const tt,
  idx_t ** const ssizes,
  rank_info const * const rinfo);

} /* extern "C" */

/**
 * In distributed-SGD, we first partition tensor into a P^M grid of tiles,
 * in a way that nnz per tile is load balanced, where P is # of ranks
 * and M is # of modes.
 *
 * We distribute the tensor following the partition of the first mode
 * of the P^M grid so that all updates to the first mode happen locally.
 * We sort the modes in a descending order, making the first mode
 * the longest to minimize communication volume of exchanging factor
 * matrices.
 *
 * We also partition the factor matrices for the remaining mode,
 * following the P^M grid.
 *
 * As a result, each rank will own P^(M-1) grid.
 * Suppose (p, r_2, r_3, ..., r_M) is the coordinate of a tile
 * owned by pth rank.
 * When we update this tile, we need to communicate with rank r_2 for
 * 2nd mode, rank r_3 for 3rd mode, and so on.
 * When communicating, we only exchange the portion of factor matrices
 * that correspond to non-zeros present in the given tile.
 * For example, suppose we have a 8x8x8 tensor, uniformly decomposed
 * for 2 MPI ranks.
 * Let's assume that the tile at coordinate (0, 0, 1) has non-zeros at
 * (1, 3, 5), (3, 0, 5), (3, 2, 7).
 * Instead of exchanging rows 4-7 with rank 1, rank 0 just need to exchange
 * rows 5 and 7.
 *
 * As you can see, due to sparsity, this saves the communication volume
 * significantly.
 * For example, if we don't consider the sparsity when exchanging factor
 * matrices, communication volume per epoch would be
 * O((I_2 + I_3 + ... + I_M)*F*P^(M-1)), where I_m is the dimension of mth
 * mode as analyzed in Shin and Kang's paper.
 * By exploiting sparsity, communication volume reduces to O(nnz*F*M).
 *
 * For efficient communication, we compact the indices that are not owned
 * by the given rank.
 * Going back to the example above, indices of non-zeros at (1, 3, 5),
 * (3, 0, 5), and (3, 2, 7) can be compacted into (1, 3, 4), (3, 0, 4),
 * and (3, 2, 5).
 */

/**
 * mode specific information of stratum
 */
typedef struct
{
  int owner;
  bool is_local;
  vector<int> leasers;

  vector<idx_t> nnzs_of_non_empty_slice[2];
    /* global index to nnz map (empty slice omitted) */

  // only valid for non-local
  vector<vector<idx_t> > local_rows_to_send;
    /* local rows to send/recv to/from each leaser */
  vector<vector<idx_t> > remote_hists;
    /* nnzs in local rows of each leaser */

  vector<idx_t> external_to_global;
    /* map from external indices to global indices */
  unordered_map<idx_t, idx_t> global_to_external;
    /* inverse map of external_to_globals */

  // only valid when 1) non-local and > 1 leasers
  // 2) local and >= 1 leasers
  // negative weight indicate first number in reduction
  vector<val_t> leaser_weights;

  // only valid when local and >= 1 leasers
  val_t local_weight;
} stratum_mode_t;

typedef struct
{
  /* tensor tile of this stratum.
   * If stratum is local for mth mode, indices for mth mode is local
   * (shifted from global indices so that they start from 0).
   * If stratum is remote for mth mode, indices for mth mode is external
   * (compacted so that they can directly index received data). */
  splatt_csf *tile_csf;
  sptensor_t *tile;

  idx_t *perm; /* permutation of non-zeros in tile */

  stratum_mode_t mode_infos[MAX_NMODES];

  vector<MPI_Request>
    recv_from_owner_requests,
    send_to_owner_requests,
    recv_from_leasers_requests,
    send_to_leasers_requests;

} stratum_t;

/**
 * per-mode communication information for compact_train and compact_validate
 */
typedef struct
{
  vector<vector<idx_t> > local_rows_to_send;

  vector<vector<idx_t> > external_to_global;
    /* external indices separated for each remote rank */
  unordered_map<idx_t, idx_t> global_to_external;

  val_t *send_buf, *recv_buf;

  vector<vector<idx_t> > non_empty_slices;
} sgd_comm_mode_t;

typedef struct
{
  tc_model *model;
  rank_info *rinfo;

  int nstratum;
  vector<stratum_t> stratums;
  idx_t *stratum_perm;

  /* In these two tensors, indices owned by this rank are local.
   * Other indices are external */
  sptensor_t *compact_train;
  sptensor_t *compact_validate;

  sgd_comm_mode_t mode_infos[MAX_NMODES];

  vector<MPI_Request> requests;
    /* length 2*(npes-1)*(nmodes-1) array */
} sgd_comm_t;

sgd_comm_t sgd_comm;

idx_t dims_temp[MAX_NMODES];

static int p_get_nstratum(int nmodes, int npes)
{
  int nstratum = 1;
  for(int m=1; m < nmodes; ++m) {
    nstratum *= SS_MIN(npes, dims_temp[m]);
  }
  return nstratum;
}

static int p_find_stratum_of(const sptensor_t *tensor, idx_t idx, sgd_comm_t *sgd_comm)
{
  rank_info *rinfo = sgd_comm->rinfo;
  int rank = rinfo->rank;
  int npes = rinfo->npes;
  int nmodes = tensor->nmodes;

  int stratum = 0;
  for(int m=nmodes - 1; m >= 1; --m) {
    int p = SS_MIN(npes, dims_temp[m]);
    stratum *= p;

    int owner = std::upper_bound(rinfo->layer_ptrs[m], rinfo->layer_ptrs[m] + npes + 1, tensor->ind[m][idx]) - rinfo->layer_ptrs[m] - 1;
    stratum += (owner + p*npes - rank)%p;
    //printf("[%d] m=%d owner=%d offset=%d\n", rank, m, owner, (owner + p - rank)%p);
  }

  assert(stratum >= 0 && stratum < sgd_comm->nstratum);

  return stratum;
}

static void p_count_nnz_of_tiles(
  idx_t **nnzs, sgd_comm_t *sgd_comm, sptensor_t *train)
{
  rank_info *rinfo = sgd_comm->rinfo;
  int npes = rinfo->npes;
  int rank = rinfo->rank;
  int nstratum = sgd_comm->nstratum;
  int nmodes = train->nmodes;

  *nnzs = (idx_t *)splatt_malloc(sizeof(idx_t) * nstratum);
  for(idx_t s=0; s < nstratum; ++s) {
    (*nnzs)[s] = 0;
  }

  vector<idx_t> *indices_per_stratum[nmodes];
  for(int m=1; m < nmodes; ++m) {
    indices_per_stratum[m] = new vector<idx_t>[nstratum];
  }

  /* count nnz of each tile and each slice */
  double t = omp_get_wtime();
  for(idx_t n=0; n < train->nnz; ++n) {
    int s = p_find_stratum_of(train, n, sgd_comm);
    for(int m=1; m < nmodes; ++m) {
      indices_per_stratum[m][s].push_back(train->ind[m][n]);
      /* collect non-local indices of this stratum */
      /*map<idx_t, idx_t> *nnzs_of_slice =
        &sgd_comm->stratums[s].mode_infos[m].nnzs_of_non_empty_slice;
      idx_t idx = train->ind[m][n];
      if(nnzs_of_slice->find(idx) == nnzs_of_slice->end()) {
        (*nnzs_of_slice)[idx] = 1;
      }
      else {
        ++(*nnzs_of_slice)[idx];
      }*/
    }

    ++(*nnzs)[s];
  }
  if (0 == rank) printf("%s:%d %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  t = omp_get_wtime();

#pragma omp parallel for collapse(2)
  for(int s=0; s < nstratum; ++s) {
    for(int m=1; m < nmodes; ++m) {
      map<idx_t, idx_t> nnzs_of_slice;
      assert(indices_per_stratum[m][s].size() == (*nnzs)[s]);
      for(int i=0; i < indices_per_stratum[m][s].size(); ++i) {
        idx_t idx = indices_per_stratum[m][s][i];
        if(nnzs_of_slice.find(idx) == nnzs_of_slice.end()) {
          nnzs_of_slice[idx] = 1;
        }
        else {
          ++nnzs_of_slice[idx];
        }
      }

      map<idx_t, idx_t>::iterator itr;
      sgd_comm->stratums[s].mode_infos[m].nnzs_of_non_empty_slice[0].resize(nnzs_of_slice.size());
      sgd_comm->stratums[s].mode_infos[m].nnzs_of_non_empty_slice[1].resize(nnzs_of_slice.size());
      idx_t i = 0;
      for(itr = nnzs_of_slice.begin(); itr != nnzs_of_slice.end(); ++itr, ++i) {
        sgd_comm->stratums[s].mode_infos[m].nnzs_of_non_empty_slice[0][i] = itr->first;
        sgd_comm->stratums[s].mode_infos[m].nnzs_of_non_empty_slice[1][i] = itr->second;
      }
    }
  }
  for(int m=1; m < nmodes; ++m) {
    delete[] indices_per_stratum[m];
  }

  if (0 == rank) printf("%s:%d %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
}

static void p_find_non_empty_slices(
  sgd_comm_t *sgd_comm, const sptensor_t *validate)
{
  rank_info *rinfo = sgd_comm->rinfo;
  int rank = rinfo->rank;
  int npes = rinfo->npes;
  int nstratum = sgd_comm->nstratum;
  int nmodes = sgd_comm->model->nmodes;

  vector<idx_t> indices_per_mode[nmodes][npes];
  for(int m=1; m < nmodes; ++m) {
    sgd_comm->mode_infos[m].non_empty_slices.resize(npes);
  }

  /* collect non-local indices separating them by their owners */
  for(int s=0; s < nstratum; ++s) {
    for(idx_t m=1; m < nmodes; ++m) {
      stratum_mode_t *mode_info = &sgd_comm->stratums[s].mode_infos[m];
      int owner = mode_info->owner;
      if(owner == rank) continue;

      map<idx_t, idx_t>::iterator itr;
      for(idx_t i = 0; i < mode_info->nnzs_of_non_empty_slice[0].size(); ++i) {
        indices_per_mode[m][owner].push_back(mode_info->nnzs_of_non_empty_slice[0][i]);
      }
    }
  }

  /* also collect non-local indices in validate matrix */
  for(idx_t n=0; n < validate->nnz; ++n) {
    int s = p_find_stratum_of(validate, n, sgd_comm);
    for(int m=1; m < nmodes; ++m) {
      int owner = sgd_comm->stratums[s].mode_infos[m].owner;
      if(owner != rank) {
        indices_per_mode[m][owner].push_back(validate->ind[m][n]);
      }
    }
  }

  for(int m=0; m < nmodes; ++m) {
    for(int p=0; p < npes; ++p) {
      set<idx_t> temp_set;
      for(int i=0; i < indices_per_mode[m][p].size(); ++i) {
        temp_set.insert(indices_per_mode[m][p][i]);
      }
      set<idx_t>::iterator itr;
      for(itr = temp_set.begin(); itr != temp_set.end(); ++itr) {
        sgd_comm->mode_infos[m].non_empty_slices[p].push_back(*itr);
      }
      indices_per_mode[m][p].clear();
    }
  }
}

static void p_map_global_to_local(sgd_comm_t *sgd_comm)
{
  int nmodes = sgd_comm->model->nmodes;
  rank_info *rinfo = sgd_comm->rinfo;
  int npes = rinfo->npes;
  int rank = rinfo->rank;
  int nstratum = sgd_comm->nstratum;

  for(int s=0; s < nstratum; ++s) {
    for(int m=1; m < nmodes; ++m) {
      stratum_mode_t *mode_info = &sgd_comm->stratums[s].mode_infos[m];
      if(mode_info->is_local) continue;

      mode_info->external_to_global.resize(
        mode_info->nnzs_of_non_empty_slice[0].size());

      idx_t i = 0;
      for(idx_t i=0; i < mode_info->nnzs_of_non_empty_slice[0].size(); ++i) {
        idx_t idx = mode_info->nnzs_of_non_empty_slice[0][i];
        mode_info->external_to_global[i] = idx;
        mode_info->global_to_external[idx] = i;
      }
    }
  }

  printf("[%d] %s:%d\n", rank, __FILE__, __LINE__);

  MPI_Request send_request, recv_request;

  for(int s=0; s < nstratum; ++s) {
    for(int m=1; m < nmodes; ++m) {
      stratum_mode_t *mode_info = &sgd_comm->stratums[s].mode_infos[m];

      int owner = mode_info->owner;
      idx_t send_len = 0;
      if(owner != rank) {
        send_len = mode_info->nnzs_of_non_empty_slice[0].size();
        MPI_Isend(
          &send_len, 1, SPLATT_MPI_IDX, owner, m, MPI_COMM_WORLD, &send_request);
      }

      int nleasers = mode_info->leasers.size();
      if(nleasers) {
        mode_info->local_rows_to_send.resize(nleasers);
        mode_info->remote_hists.resize(nleasers);

        for(int i=0; i < mode_info->leasers.size(); ++i) {
          idx_t recv_len = 0;
          MPI_Irecv(
            &recv_len, 1, SPLATT_MPI_IDX,
            mode_info->leasers[i], m, MPI_COMM_WORLD, &recv_request);
          MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

          mode_info->local_rows_to_send[i].resize(recv_len);
          mode_info->remote_hists[i].resize(recv_len);
        }
      }

      if(owner != rank) {
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);

        MPI_Isend(
          &mode_info->external_to_global[0],
          send_len,
          SPLATT_MPI_IDX,
          owner,
          m,
          MPI_COMM_WORLD,
          &send_request);
      }

      if(nleasers) {
        for(int i=0; i < mode_info->leasers.size(); ++i) {
          MPI_Irecv(
            &mode_info->local_rows_to_send[i][0],
            mode_info->local_rows_to_send[i].size(),
            SPLATT_MPI_IDX,
            mode_info->leasers[i],
            m,
            MPI_COMM_WORLD,
            &recv_request);
          MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

          /* convert received global indices to local indices */
          for(idx_t j=0; j < mode_info->local_rows_to_send[i].size(); ++j) {
            mode_info->local_rows_to_send[i][j] -= rinfo->layer_ptrs[m][rank];
          }
        }
      }

      if(owner != rank) {
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);

        vector<idx_t> external_hist(send_len);
        map<idx_t, idx_t>::iterator itr;
        idx_t i = 0;
        for(idx_t i=0; i < mode_info->nnzs_of_non_empty_slice[1].size(); ++i) {
          external_hist[i] = mode_info->nnzs_of_non_empty_slice[1][i];
        }

        MPI_Isend(
          &external_hist[0],
          send_len,
          SPLATT_MPI_IDX,
          owner,
          m,
          MPI_COMM_WORLD,
          &send_request);
      }

      if(nleasers) {
        for(int i=0; i < mode_info->leasers.size(); ++i) {
          MPI_Irecv(
            &mode_info->remote_hists[i][0],
            mode_info->remote_hists[i].size(),
            SPLATT_MPI_IDX,
            mode_info->leasers[i],
            m,
            MPI_COMM_WORLD,
            &recv_request);
          MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        }
        if(!mode_info->is_local && nleasers <= 1) {
          mode_info->remote_hists.clear();
        }
      }

      if(owner != rank) MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    }
  } /* for each stratum */

  printf("[%d] %s:%d\n", rank, __FILE__, __LINE__);

  /* compute weights for reducing multiple updates */
  for(int s=0; s < nstratum; ++s) {
    for(int m=1; m < nmodes; ++m) {
      stratum_mode_t *mode_info = &sgd_comm->stratums[s].mode_infos[m];
      int nleasers = mode_info->leasers.size();
      if(nleasers == 0 || !mode_info->is_local && nleasers <= 1) {
        // if we don't need to reduce data received from multiple leasers,
        // this is the last place we'd need nnzs_of_non_empty_slice, so free the memory.
        mode_info->nnzs_of_non_empty_slice[0].clear();
        mode_info->nnzs_of_non_empty_slice[1].clear();
        continue;
      }

      idx_t baserow = rinfo->layer_ptrs[m][rank];
      idx_t nlocalrow = rinfo->layer_ptrs[m][rank + 1] - baserow;

      idx_t total = 0;

      if(mode_info->is_local && !mode_info->nnzs_of_non_empty_slice[1].empty()) {
        //printf("%s:%d\n", __FILE__, __LINE__);
        assert(mode_info->nnzs_of_non_empty_slice[1].size() <= 1);
        total += mode_info->nnzs_of_non_empty_slice[1][0];
      }

      mode_info->leaser_weights.resize(nleasers);

      for(int leaser=0; leaser < nleasers; ++leaser) {
        assert(mode_info->local_rows_to_send[leaser].size() <= 1);
        if(!mode_info->local_rows_to_send[leaser].empty()) {
          //printf("%s:%d\n", __FILE__, __LINE__);
          if(total == 0) {
            /* first reduction */
            mode_info->leaser_weights[leaser] = -1;
          }
          else {
            mode_info->leaser_weights[leaser] = 1;
          }
          total += mode_info->remote_hists[leaser][0];
        }
      }

      if(mode_info->is_local && !mode_info->nnzs_of_non_empty_slice[1].empty()) {
        mode_info->local_weight =
          (val_t)mode_info->nnzs_of_non_empty_slice[1][0]/total;
      }

      for(int leaser=0; leaser < nleasers; ++leaser) {
        if(!mode_info->local_rows_to_send[leaser].empty()) {
          mode_info->leaser_weights[leaser] *=
            (val_t)(mode_info->remote_hists[leaser][0])/total;
        }
      }
    }
  }

  printf("[%d] %s:%d\n", rank, __FILE__, __LINE__);

  for(int m=1; m < nmodes; ++m) {
    sgd_comm_mode_t *mode_info = &sgd_comm->mode_infos[m];

    mode_info->local_rows_to_send.resize(npes);
    mode_info->external_to_global.resize(npes);

    idx_t i_base = 0;
    for(int p=0; p < npes; ++p) {
      if(p == rank) assert(mode_info->non_empty_slices[p].empty());
      mode_info->external_to_global[p].resize(mode_info->non_empty_slices[p].size());

      for(idx_t i=0; i < mode_info->non_empty_slices[p].size(); ++i) {
        mode_info->external_to_global[p][i] = mode_info->non_empty_slices[p][i];
        mode_info->global_to_external[mode_info->non_empty_slices[p][i]] = i + i_base;
      }
      i_base += mode_info->non_empty_slices[p].size();
    }
  }

  printf("[%d] %s:%d\n", rank, __FILE__, __LINE__);

  for(int p=0; p < npes; ++p) {
    for(int m=1; m < nmodes; ++m) {
      sgd_comm_mode_t *mode_info = &sgd_comm->mode_infos[m];

      // FIXME: can do this with all-to-all
      idx_t send_len = mode_info->external_to_global[p].size();
      MPI_Isend(
        &send_len, 1, SPLATT_MPI_IDX, p, m, MPI_COMM_WORLD, &send_request);

      idx_t recv_len = 0;
      MPI_Irecv(
        &recv_len, 1, SPLATT_MPI_IDX, p, m, MPI_COMM_WORLD, &recv_request);

      MPI_Wait(&send_request, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

      if(p == rank) assert(recv_len == 0);
      mode_info->local_rows_to_send[p].resize(recv_len);

      MPI_Isend(
        &mode_info->external_to_global[p][0],
        send_len, SPLATT_MPI_IDX, p, m, MPI_COMM_WORLD, &send_request);

      MPI_Irecv(
        &mode_info->local_rows_to_send[p][0],
        recv_len, SPLATT_MPI_IDX, p, m, MPI_COMM_WORLD, &recv_request);

      MPI_Wait(&send_request, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

      /* convert received global indices to local indices */
      for(idx_t i=0; i < mode_info->local_rows_to_send[p].size(); ++i) {
        mode_info->local_rows_to_send[p][i] -= rinfo->layer_ptrs[m][rank];
      }
    }
  }
}

static void p_populate_tiles(tc_ws *ws, sptensor_t *train, sgd_comm_t *sgd_comm, idx_t *nnzs)
{
  rank_info *rinfo = sgd_comm->rinfo;
  int npes = rinfo->npes;
  int rank = rinfo->rank;
  int nstratum = sgd_comm->nstratum;
  int nmodes = train->nmodes;

#define SPLATT_MEASURE_LOAD_IMBALANCE
#ifdef SPLATT_MEASURE_LOAD_IMBALANCE
  idx_t global_nnzs[npes][nstratum];
  for(int p=0; p < npes; ++p) {
    if(p == rank) {
      for(int s=0; s < nstratum; ++s) {
        global_nnzs[p][s] = nnzs[s];
      }
    }
    MPI_Bcast(global_nnzs[p], nstratum, SPLATT_MPI_IDX, p, MPI_COMM_WORLD);
  }
  if(0 == rank) {
    double avg_total = 0;
    idx_t maximum_total = 0;
    for(int s=0; s < nstratum; ++s) {
      idx_t sum = 0;
      idx_t maximum = 0;
      for(int p=0; p < npes; ++p) {
        sum += global_nnzs[p][s];
        maximum = SS_MAX(maximum, global_nnzs[p][s]);
      }
      double avg = (double)sum/npes;
      avg_total += avg;
      maximum_total += maximum;
      //printf("stratum %d avg_nnz %g max_nnz %ld load_imbalance %g\n", s, avg, maximum, maximum/avg);
    }
    printf("total nnz_load_imbalance %g\n", maximum_total/avg_total);
  }
#endif

  /* allocate a tile for each stratum */
  for(int s=0; s < nstratum; ++s) {
    sgd_comm->stratums[s].tile = tt_alloc(nnzs[s], nmodes);
  }
  sgd_comm->compact_train = tt_alloc(train->nnz, nmodes);
  splatt_free(sgd_comm->compact_train->vals);
  sgd_comm->compact_train->vals = train->vals;
  splatt_free(sgd_comm->compact_train->ind[0]);
  sgd_comm->compact_train->ind[0] = train->ind[0];

  for(int s=0; s < nstratum; ++s) {
    nnzs[s] = 0;
  }

  idx_t nlocalrows[nmodes];
  for(idx_t m=0; m < nmodes; ++m) {
    nlocalrows[m] = rinfo->layer_ptrs[m][rank + 1] - rinfo->layer_ptrs[m][rank];
  }

  idx_t base[nmodes][npes + 1];
  for(int m=1; m < nmodes; ++m) {
    for(int p=1; p <=npes; ++p) {
      base[m][p] = 0;
    }
    base[m][0] = nlocalrows[m];
  }
  for(int s=0; s < nstratum; ++s) {
    for(int m=1; m < nmodes; ++m) {
      stratum_mode_t *mode_info = &sgd_comm->stratums[s].mode_infos[m];
      int owner = mode_info->owner;

      if (owner != rank) {
        base[m][owner + 1] = SS_MAX(base[m][owner + 1], mode_info->external_to_global.size());
      }
    }
  }
  for(int m=1; m < nmodes; ++m) {
    for(int p=0; p < npes; ++p) {
      base[m][p + 1] += base[m][p];
    }
  }

  for(idx_t n=0; n < train->nnz; ++n) {
    /* find tile id */
    int s = p_find_stratum_of(train, n, sgd_comm);
    sptensor_t *tile = sgd_comm->stratums[s].tile;

    tile->vals[nnzs[s]] = train->vals[n];
    tile->ind[0][nnzs[s]] = train->ind[0][n];

    for(int m=1; m < nmodes; ++m) {
      stratum_mode_t *stratum_mode_info = &sgd_comm->stratums[s].mode_infos[m];
      sgd_comm_mode_t *mode_info = &sgd_comm->mode_infos[m];

      int owner = stratum_mode_info->owner;

      idx_t idx = train->ind[m][n];
      if(owner != rank) {
        assert(stratum_mode_info->global_to_external.find(idx) != stratum_mode_info->global_to_external.end());
        tile->ind[m][nnzs[s]] = base[m][owner] + stratum_mode_info->global_to_external[idx];

        assert(mode_info->global_to_external.find(idx) != mode_info->global_to_external.end());
        sgd_comm->compact_train->ind[m][n] = nlocalrows[m] + mode_info->global_to_external[idx];
      }
      else {
        /* convert global index to local index */
        tile->ind[m][nnzs[s]] = idx - rinfo->layer_ptrs[m][rank];

        assert(idx >= rinfo->layer_ptrs[m][rank]);
        assert(idx < rinfo->layer_ptrs[m][rank + 1]);
        sgd_comm->compact_train->ind[m][n] = idx - rinfo->layer_ptrs[m][rank];
      }
    }

    ++nnzs[s];
  }
  for(int s=0; s < nstratum; ++s) {
    for(int m=1; m < nmodes; ++m) {
      sgd_comm->stratums[s].mode_infos[m].global_to_external.clear();
    }
  }

  for(idx_t s=0; s < nstratum; ++s) {
    sptensor_t *tile = sgd_comm->stratums[s].tile;
    assert(nnzs[s] == tile->nnz);
    for(int m=0; m < nmodes; ++m) {
      tile->dims[m] = 0;
    }
    for(idx_t n=0; n < tile->nnz; ++n) {
      for(int m=0; m < nmodes; ++m) {
        tile->dims[m] = SS_MAX(tile->dims[m], 1 + tile->ind[m][n]);
      }
    }

    /*std::stringstream stream;
    stream << "[" << rank << "]" << s;
    for(idx_t n=0; n < tiles[s]->nnz; ++n) {
      stream << " " << (tiles[s]->ind[0][n] + rinfo->layer_ptrs[0][rank]);
      for(int m=1; m < nmodes; ++m) {
        stratum_mode_t *stratum_mode_info = &sgd_comm->stratums[s].mode_infos[m];
        int owner = stratum_mode_info->owner;
        if(owner != rank) {
          stream << "::" << stratum_mode_info->external_to_global[tiles[s]->ind[m][n] - base[m][owner]];
        }
        else {
          stream << ":" << tiles[s]->ind[m][n] + rinfo->layer_ptrs[m][rank];
        }
      }
    }
    printf("%s\n", stream.str().c_str());*/
  }

  if(0 == rank) printf("%s:%d\n", __FILE__, __LINE__);

  double t = omp_get_wtime();

  if(ws->csf) {
    double * opts = splatt_default_opts();
    opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

    for(idx_t s=0; s < nstratum; ++s) {
      stratum_t *stratum = &sgd_comm->stratums[s];

      stratum->tile_csf = (splatt_csf *)splatt_malloc(sizeof(splatt_csf));
      csf_alloc_mode(stratum->tile, CSF_SORTED_BIGFIRST, 0, stratum->tile_csf, opts);
      assert(stratum->tile_csf->ntiles == 1);
      tt_free(stratum->tile);
      stratum->tile = NULL;

      idx_t nslices = stratum->tile_csf->pt->nfibs[0];
      stratum->perm = (idx_t *)splatt_malloc(sizeof(idx_t) * nslices);
#pragma omp parallel for
      for(idx_t n=0; n < nslices; ++n) {
        stratum->perm[n] = n;
      }
    }
    splatt_free_opts(opts);
  }
  else {
    for(idx_t s=0; s < nstratum; ++s) {
      stratum_t *stratum = &sgd_comm->stratums[s];

      stratum->perm = (idx_t *)splatt_malloc(sizeof(idx_t) * stratum->tile->nnz);
#pragma omp parallel for
      for(idx_t n=0; n < stratum->tile->nnz; ++n) {
        stratum->perm[n] = n;
      }

      stratum->tile_csf = NULL;
    }
  }

  if (0 == rank) {
    printf("%s:%d %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  }
}

static void p_setup_sgd_persistent_comm(sgd_comm_t *sgd_comm)
{
  tc_model *model = sgd_comm->model;
  int nmodes = model->nmodes;
  int nfactors = model->rank;

  int nstratum = sgd_comm->nstratum;
  rank_info *rinfo = sgd_comm->rinfo;
  int npes = rinfo->npes;
  int rank = rinfo->rank;

  for(int m=1; m < nmodes; ++m) {
    idx_t maximum = 0;
    for(int p=0; p < npes; ++p) {
      if(p != rank) {
        maximum += sgd_comm->mode_infos[m].local_rows_to_send[p].size();
      }
    }
    for(int s=0; s < nstratum; ++s) {
      stratum_mode_t *mode_info = &sgd_comm->stratums[s].mode_infos[m];

      maximum = SS_MAX(maximum, mode_info->external_to_global.size());

      idx_t len = 0;
      for(int i=0; i < mode_info->leasers.size(); ++i) {
        len += mode_info->local_rows_to_send[i].size();
      }
      assert(maximum >= len);
    }
    sgd_comm->mode_infos[m].send_buf = (val_t *)splatt_malloc(sizeof(val_t)*maximum*nfactors);
    sgd_comm->mode_infos[m].recv_buf = (val_t *)splatt_malloc(sizeof(val_t)*maximum*nfactors);
    /* TODO: tighter memory allocation */
  }

  idx_t nlocalrows[nmodes];
  for(idx_t m=0; m < nmodes; ++m) {
    nlocalrows[m] = rinfo->layer_ptrs[m][rank + 1] - rinfo->layer_ptrs[m][rank];
  }

  idx_t base[nmodes][npes + 1];
  for(int m=1; m < nmodes; ++m) {
    for(int p=1; p <=npes; ++p) {
      base[m][p] = 0;
    }
    base[m][0] = nlocalrows[m];
  }
  for(int s=0; s < nstratum; ++s) {
    for(int m=1; m < nmodes; ++m) {
      stratum_mode_t *mode_info = &sgd_comm->stratums[s].mode_infos[m];
      int owner = mode_info->owner;

      if (owner != rank) {
        base[m][owner + 1] = SS_MAX(base[m][owner + 1], mode_info->external_to_global.size());
      }
    }
  }
  for(int m=1; m < nmodes; ++m) {
    for(int p=0; p < npes; ++p) {
      base[m][p + 1] += base[m][p];
    }
  }

  for(int s=0; s < nstratum; ++s) {
    stratum_t *stratum = &sgd_comm->stratums[s];

    int owner_request_cnt = 0, leaser_request_cnt = 0;
    for(int m=1; m < nmodes; ++m) {
      stratum_mode_t *mode_info = &stratum->mode_infos[m];
      if(!mode_info->is_local) {
        ++owner_request_cnt;
      }
      leaser_request_cnt += mode_info->leasers.size();
    }

    stratum->recv_from_owner_requests.resize(owner_request_cnt);
    stratum->send_to_owner_requests.resize(owner_request_cnt);
    stratum->recv_from_leasers_requests.resize(leaser_request_cnt);
    stratum->send_to_leasers_requests.resize(leaser_request_cnt);

    owner_request_cnt = 0;
    leaser_request_cnt = 0;
    for(int m=1; m < nmodes; ++m) {
      stratum_mode_t *mode_info = &stratum->mode_infos[m];
      int owner = mode_info->owner;
      if(owner != rank) {
        /* receive model from owner */
        MPI_Recv_init(
          model->factors[m] + base[m][owner]*nfactors, /* begin of external rows */
          mode_info->external_to_global.size()*nfactors,
          SPLATT_MPI_VAL,
          owner,
          m,
          MPI_COMM_WORLD,
          &stratum->recv_from_owner_requests[owner_request_cnt]);

        /* send model back to owner */
        MPI_Send_init(
          model->factors[m] + base[m][owner]*nfactors, /* begin of external rows */
          mode_info->external_to_global.size()*nfactors,
          SPLATT_MPI_VAL,
          owner,
          m + nmodes,
          MPI_COMM_WORLD,
          &stratum->send_to_owner_requests[owner_request_cnt]);
        ++owner_request_cnt;
      }

      idx_t buf_offset = 0;
      for(int i=0; i < mode_info->leasers.size(); ++i) {
        int leaser = mode_info->leasers[i];

        /* lease model */
        MPI_Send_init(
          sgd_comm->mode_infos[m].send_buf, /* don't need to add buf_offset because multiple leasers mean only one row owned per rank */
          mode_info->local_rows_to_send[i].size()*nfactors,
          SPLATT_MPI_VAL,
          leaser,
          m,
          MPI_COMM_WORLD,
          &stratum->send_to_leasers_requests[leaser_request_cnt]);

        /* get model back */
        MPI_Recv_init(
          sgd_comm->mode_infos[m].recv_buf + buf_offset*nfactors,
          mode_info->local_rows_to_send[i].size()*nfactors,
          SPLATT_MPI_VAL,
          leaser,
          m + nmodes,
          MPI_COMM_WORLD,
          &stratum->recv_from_leasers_requests[leaser_request_cnt]);

        buf_offset += mode_info->local_rows_to_send[i].size();
        ++leaser_request_cnt;
      }
    }
  }

  sgd_comm->requests.resize(2*(npes - 1)*(nmodes - 1));
  int request_cnt = 0;
  for(int m=1; m < nmodes; ++m) {
    sgd_comm_mode_t *mode_info = &sgd_comm->mode_infos[m];
    idx_t local_offset = 0, remote_offset = 0;
    for(int p=0; p < npes; ++p) {
      if(p == rank) continue;

      MPI_Recv_init(
        model->factors[m] + (nlocalrows[m] + local_offset)*nfactors,
        mode_info->external_to_global[p].size()*nfactors,
        SPLATT_MPI_VAL,
        p,
        m,
        MPI_COMM_WORLD,
        &sgd_comm->requests[request_cnt]);
      local_offset += mode_info->external_to_global[p].size();

      MPI_Send_init(
        sgd_comm->mode_infos[m].send_buf + remote_offset*nfactors,
        mode_info->local_rows_to_send[p].size()*nfactors,
        SPLATT_MPI_VAL,
        p,
        m,
        MPI_COMM_WORLD,
        &sgd_comm->requests[(npes - 1)*(nmodes - 1) + request_cnt]);
      remote_offset += mode_info->local_rows_to_send[p].size();

      ++request_cnt;
    }
  }
}

static void p_free_stratum_mode(stratum_mode_t *mode_info)
{
  mode_info->local_rows_to_send.clear();
  mode_info->remote_hists.clear();
  mode_info->external_to_global.clear();
  mode_info->global_to_external.clear();
  mode_info->nnzs_of_non_empty_slice[0].clear();
  mode_info->nnzs_of_non_empty_slice[1].clear();
  mode_info->leaser_weights.clear();
}

static void p_free_sgd_comm_mode(sgd_comm_mode_t *mode_info)
{
  mode_info->local_rows_to_send.clear();
  mode_info->external_to_global.clear();
  mode_info->global_to_external.clear();
  mode_info->non_empty_slices.clear();

  splatt_free(mode_info->send_buf);
  splatt_free(mode_info->recv_buf);
}

static void p_free_stratum(stratum_t *stratum)
{
  if(stratum->tile_csf) {
    for(int m=1; m < stratum->tile_csf->nmodes; ++m) {
      p_free_stratum_mode(&stratum->mode_infos[m]);
    }
    csf_free_mode(stratum->tile_csf);
    splatt_free(stratum->tile_csf);
  }
  else {
    assert(stratum->tile);
    for(int m=1; m < stratum->tile->nmodes; ++m) {
      p_free_stratum_mode(&stratum->mode_infos[m]);
    }
    tt_free(stratum->tile);
  }
  splatt_free(stratum->perm);

  for(int i=0; i < stratum->recv_from_owner_requests.size(); ++i) {
    MPI_Request_free(&stratum->recv_from_owner_requests[i]);
  }
  for(int i=0; i < stratum->send_to_owner_requests.size(); ++i) {
    MPI_Request_free(&stratum->send_to_owner_requests[i]);
  }
  for(int i=0; i < stratum->recv_from_leasers_requests.size(); ++i) {
    MPI_Request_free(&stratum->recv_from_leasers_requests[i]);
  }
  for(int i=0; i < stratum->send_to_leasers_requests.size(); ++i) {
    MPI_Request_free(&stratum->send_to_leasers_requests[i]);
  }

  stratum->recv_from_owner_requests.clear();
  stratum->send_to_owner_requests.clear();
  stratum->recv_from_leasers_requests.clear();
  stratum->send_to_leasers_requests.clear();
}

static void p_init_sgd_comm(
  sgd_comm_t *sgd_comm,
  rank_info *rinfo, tc_model *model)
{
  int nmodes = model->nmodes;
  int npes = rinfo->npes;
  int rank = rinfo->rank;

  sgd_comm->nstratum = p_get_nstratum(nmodes, npes);
  sgd_comm->rinfo = rinfo;
  sgd_comm->stratums.resize(sgd_comm->nstratum);
  sgd_comm->model = model;
  sgd_comm->stratum_perm = (idx_t *)splatt_malloc(sizeof(idx_t) * sgd_comm->nstratum);

  for(int s=0; s < sgd_comm->nstratum; ++s) {
    stratum_t *stratum = &sgd_comm->stratums[s];

    int s_temp = s;
    for(int m=1; m < nmodes; ++m) {
      stratum_mode_t *mode_info = &stratum->mode_infos[m];
      int p = SS_MIN(npes, dims_temp[m]);
      int offset = s_temp%p;
      mode_info->owner = (offset + rank)%p;
      mode_info->is_local = mode_info->owner == rank;
      /*std::stringstream stream;
      stream << "[" << rank << "] " << s << ":" << m;*/
      for(int q=0; q < npes; ++q) {
        if((offset + q)%p == rank && q != rank) {
          mode_info->leasers.push_back(q);
          //stream << " " << q;
        }
      }
      //if(m == 2) printf("%s\n", stream.str().c_str());
      s_temp /= p;
    }

    sgd_comm->stratum_perm[s] = s;
  }
}

static void p_free_sgd_comm(sgd_comm_t *sgd_comm)
{
  int nstratum = sgd_comm->nstratum;
  int nmodes = sgd_comm->model->nmodes;
  int npes = sgd_comm->rinfo->npes;

  for(idx_t s=0; s < nstratum; ++s) {
    p_free_stratum(&sgd_comm->stratums[s]);
  }
  splatt_free(sgd_comm->stratum_perm);

  sgd_comm->compact_train->vals = NULL;
  sgd_comm->compact_train->ind[0] = NULL;
  sgd_comm->compact_validate->vals = NULL;
  sgd_comm->compact_validate->ind[0] = NULL;

  tt_free(sgd_comm->compact_train);
  tt_free(sgd_comm->compact_validate);

  for(int m=1; m < nmodes; ++m) {
    p_free_sgd_comm_mode(&sgd_comm->mode_infos[m]);
  }
  for(int i=0; i < sgd_comm->requests.size(); ++i) {
    MPI_Request_free(&sgd_comm->requests[i]);
  }
  sgd_comm->requests.clear();
}
#endif /* SPLATT_USE_MPI */

val_t p_frob_sq(
    sgd_comm_t * sgd_comm,
    tc_model const * const model,
    tc_ws const * const ws)
{
  idx_t const nfactors = model->rank;
  int nmodes = model->nmodes;
  rank_info *rinfo = sgd_comm->rinfo;
  int rank = rinfo->rank;

  val_t reg_obj = 0.;

  #pragma omp parallel reduction(+:reg_obj)
  {
    for(idx_t m=0; m < nmodes; ++m) {
      idx_t nlocalrow = rinfo->layer_ptrs[m][rank + 1] - rinfo->layer_ptrs[m][rank];

      val_t accum = 0;
      val_t const * const restrict mat = model->factors[m];

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < nlocalrow * nfactors; ++x) {
        assert(!isnan(mat[x]));
        accum += mat[x] * mat[x];
      }
      reg_obj += ws->regularization[m] * accum;
    }
  } /* end omp parallel */

#ifdef SPLATT_USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &reg_obj, 1, SPLATT_MPI_VAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  assert(reg_obj > 0);
  return reg_obj;
}

void sgd_save_best_model(tc_ws *ws)
{
  int npes = ws->rinfo->npes;
  int rank = ws->rinfo->rank;
  int nmodes = ws->nmodes;
  int nfactors = ws->best_model->rank;

  for(idx_t m=0; m < nmodes; ++m) {
    sgd_comm_mode_t *mode_info = &sgd_comm.mode_infos[m];
    idx_t base = sgd_comm.rinfo->layer_ptrs[m][rank];
    idx_t nlocalrow = sgd_comm.rinfo->layer_ptrs[m][rank + 1] - base;

    if(m == 0) {
      par_memcpy(ws->best_model->factors[m], sgd_comm.model->factors[m], sizeof(val_t)*nlocalrow*nfactors);
    }
    else {
      par_memcpy(
        ws->best_model->factors[m] + base*nfactors,
        sgd_comm.model->factors[m],
        sizeof(val_t)*nlocalrow*nfactors);
      idx_t offset = 0;
      for(int p=0; p < npes; ++p) {
        if(p == rank) continue;
#pragma omp parallel for
        for(int i=0; i < mode_info->external_to_global[p].size(); ++i) {
          idx_t external_idx = nlocalrow + offset + i;
          idx_t global_idx = mode_info->external_to_global[p][i];

          for(int f=0; f < nfactors; ++f) {

            ws->best_model->factors[m][global_idx*nfactors + f] =
              sgd_comm.model->factors[m][external_idx*nfactors + f];
          }
        }
        offset += mode_info->external_to_global[p].size();
      }
    }
  }
}

void splatt_tc_sgd(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;

#ifdef SPLATT_USE_MPI
  rank_info *rinfo = ws->rinfo;

  double t = omp_get_wtime();
  /* count # nonzeros found in each index */
  idx_t * ssizes[MAX_NMODES];
  for(idx_t m=0; m < train->nmodes; ++m) {
    ssizes[m] = (idx_t *) calloc(rinfo->global_dims[m], sizeof(idx_t));
  }
  p_fill_ssizes(train, ssizes, rinfo);

  /* Find p*p*p decomposition */
  rank_info sgd_rinfo;
  memcpy(&sgd_rinfo, rinfo, sizeof(rank_info));
  for(idx_t m=0; m < model->nmodes; ++m) {
    dims_temp[m] = SS_MIN(sgd_rinfo.npes, model->dims[m]);
    sgd_rinfo.dims_3d[m] = sgd_rinfo.npes;
    p_find_layer_boundaries(ssizes, m, &sgd_rinfo);
  }
  for(idx_t m=0; m < train->nmodes; ++m) {
    free(ssizes[m]);
  }

  int npes = rinfo->npes;
  int rank = rinfo->rank;
  int nmodes = train->nmodes;

  if(rank == 0) printf("%s:%d %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  t = omp_get_wtime();

  p_init_sgd_comm(&sgd_comm, &sgd_rinfo, model);

  /* count nnz of each tile */
  idx_t *nnzs;
  p_count_nnz_of_tiles(&nnzs, &sgd_comm, train);
  if(rank == 0) printf("%s:%d p_count_nnz_of_tiles %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  t = omp_get_wtime();

  p_find_non_empty_slices(&sgd_comm, validate);

  if(rank == 0) printf("%s:%d p_find_non_empty_slices %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  t = omp_get_wtime();

  /* mapping from global to local compacted indices within each tile */
  p_map_global_to_local(&sgd_comm);

  if(rank == 0) printf("%s:%d p_map_global_to_local %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  t = omp_get_wtime();

  /* populate each tile */
  p_populate_tiles(ws, train, &sgd_comm, nnzs);
  splatt_free(nnzs);

  if(rank == 0) printf("%s:%d p_populate_tiles %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  t = omp_get_wtime();

  sgd_comm.compact_validate = tt_alloc(validate->nnz, nmodes); 
  splatt_free(sgd_comm.compact_validate->vals);
  sgd_comm.compact_validate->vals = validate->vals;
  splatt_free(sgd_comm.compact_validate->ind[0]);
  sgd_comm.compact_validate->ind[0] = validate->ind[0];

  idx_t nlocalrows[nmodes];
  for(idx_t m=0; m < nmodes; ++m) {
    nlocalrows[m] = sgd_rinfo.layer_ptrs[m][rank + 1] - sgd_rinfo.layer_ptrs[m][rank];
  }

  for(idx_t n=0; n < validate->nnz; ++n) {
    int s = p_find_stratum_of(validate, n, &sgd_comm);

    for(idx_t m=1; m < nmodes; ++m) {
      sgd_comm_mode_t *mode_info = &sgd_comm.mode_infos[m];

      idx_t idx = validate->ind[m][n];
      if(sgd_comm.stratums[s].mode_infos[m].is_local) {
        sgd_comm.compact_validate->ind[m][n] = idx - sgd_rinfo.layer_ptrs[m][rank];
      }
      else {
        assert(mode_info->global_to_external.find(idx) != mode_info->global_to_external.end());
        sgd_comm.compact_validate->ind[m][n] = nlocalrows[m] + mode_info->global_to_external[idx];
      }
    }
  }
  for(int m=0; m < nmodes; ++m) {
    sgd_comm.mode_infos[m].global_to_external.clear();
  }

  if(rank == 0) printf("%s:%d %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  t = omp_get_wtime();

  /* set up persistent communication */
  tc_model *model_compacted = (tc_model *)splatt_malloc(sizeof(*model_compacted));
  sgd_comm.model = model_compacted;

  model_compacted->which = model->which;
  model_compacted->rank = nfactors;
  model_compacted->nmodes = train->nmodes;

  for(int m=0; m < train->nmodes; ++m) {
    idx_t bytes = train->dims[m] * nfactors * sizeof(**(model_compacted->factors));
    model_compacted->dims[m] = train->dims[m];
    if(m > 0) {
      bytes = 0;
      for(int p=0; p < npes; ++p) {
        if(p != rank) {
          bytes += sgd_comm.mode_infos[m].external_to_global[p].size();
        }
      }
      bytes += nlocalrows[m];
      model_compacted->dims[m] = bytes;
      bytes *= nfactors*sizeof(val_t);
    }
    model_compacted->factors[m] = (val_t *)splatt_malloc(bytes);
  }
  if(rank == 0) printf("%s:%d %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  t = omp_get_wtime();

  p_setup_sgd_persistent_comm(&sgd_comm);

  if(rank == 0) printf("%s:%d %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  t = omp_get_wtime();

  timer_reset(&ws->shuffle_time);
  timer_reset(&ws->train_time);
  timer_reset(&ws->test_time);

  sp_timer_t comm_with_owner_time, bcast_time, compute_time;
  sp_timer_t recv_from_owner_start_time, send_to_owner_start_time;
  sp_timer_t recv_from_owner_wait_time, send_to_owner_wait_time;
  sp_timer_t gather_time, scatter_time;
  sp_timer_t testall1_time, testall2_time, testall3_time;
  timer_reset(&comm_with_owner_time);
  timer_reset(&bcast_time);
  timer_reset(&compute_time);
  timer_reset(&recv_from_owner_start_time);
  timer_reset(&send_to_owner_start_time);
  timer_reset(&recv_from_owner_wait_time);
  timer_reset(&send_to_owner_wait_time);
  timer_reset(&gather_time);
  timer_reset(&scatter_time);
  timer_reset(&testall1_time);
  timer_reset(&testall2_time);
  timer_reset(&testall3_time);

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  par_memcpy(model_compacted->factors[0], model->factors[0], sizeof(val_t)*nlocalrows[0]*nfactors);
  for(idx_t m=1; m < train->nmodes; ++m) {
    par_memcpy(model_compacted->factors[m], model->factors[m] + sgd_rinfo.layer_ptrs[m][rank]*nfactors, sizeof(val_t)*nlocalrows[m]*nfactors);
  }

  idx_t *stratum_perm = sgd_comm.stratum_perm;
  int nstratum = sgd_comm.nstratum;

  /* for bold driver */
  val_t obj = loss + frobsq;
  val_t prev_obj = obj;

#ifdef SPLATT_MEASURE_LOAD_IMBALANCE
  vector<double> compute_times[nstratum];
#endif

  timer_start(&ws->tc_time);
  /* foreach epoch */
  idx_t e;
  for(e=1; e < ws->max_its+1; ++e) {

    /* update model from all training observations */
    timer_start(&ws->train_time);

    if(rank == 0) shuffle_idx(stratum_perm, nstratum);
    /* the same permutation across ranks */
    MPI_Bcast(stratum_perm, nstratum, SPLATT_MPI_IDX, 0, MPI_COMM_WORLD);

    if(!ws->csf && (ws->rand_per_iteration || e == 1)) {
      timer_start(&ws->shuffle_time);
      stratum_t *stratum = &sgd_comm.stratums[stratum_perm[0]];
      idx_t *arr = stratum->perm;
      idx_t N = stratum->tile->nnz;
      for(idx_t n=0; n < N/2; ++n) {
        idx_t j = (rand_idx()%(N - n)) + n;
        assert(j >= n && j < N);

        idx_t const tmp = arr[n];
        arr[n] = arr[j];
        arr[j] = tmp;
      }
      timer_stop(&ws->shuffle_time);
    }

    for(int s=0; s < nstratum; ++s) {
      stratum_t *stratum = &sgd_comm.stratums[stratum_perm[s]];

      timer_start(&comm_with_owner_time);

      /* each owner compacts models before send */
      int request_idx = 0;
      for(int m=1; m < nmodes; ++m) {
        stratum_mode_t *mode_info = &stratum->mode_infos[m];

        int nleasers = mode_info->leasers.size();
        idx_t buf_offset = 0;
        for(int leaser=0; leaser < nleasers; ++leaser) {
          timer_start(&gather_time);
          idx_t len = mode_info->local_rows_to_send[leaser].size();

          //printf("%ld: [%d] leaser=%d len=%ld\n", stratum_perm[s], rank, mode_info->leasers[leaser], len);

#pragma omp parallel for
          for(idx_t i=0; i < len; ++i) {
            idx_t local_row = mode_info->local_rows_to_send[leaser][i];
            assert(local_row < nlocalrows[m]);
            for(idx_t f=0; f < nfactors; ++f) {
              sgd_comm.mode_infos[m].send_buf[(i + buf_offset)*nfactors + f] =
                model_compacted->factors[m][local_row*nfactors + f];
            }
          }
          buf_offset += len;
          timer_stop(&gather_time);

          timer_start(&recv_from_owner_start_time);
          MPI_Start(&stratum->send_to_leasers_requests[request_idx]);
            /* start reading send_buf */
          ++request_idx;
          timer_stop(&recv_from_owner_start_time);
        }
      }

      vector<MPI_Request> *requests = &stratum->recv_from_owner_requests;
      if(0 == s && !requests->empty()) {
        timer_start(&recv_from_owner_start_time);
        MPI_Startall(requests->size(), &(*requests)[0]);
          /* start writing external model */
        timer_stop(&recv_from_owner_start_time);
      }
      //int flags[sgd_comm.request_lens[stratum]*2];
      //timer_start(&testall1_time);
      //MPI_Testall(sgd_comm.request_lens[stratum], sgd_comm.requests[0][stratum], flags, MPI_STATUSES_IGNORE);
      //timer_stop(&testall1_time);
      
      /* can start recv early because we're receiving to a separate buffer */
      requests = &stratum->recv_from_leasers_requests;
      if (!requests->empty()) {
        timer_start(&send_to_owner_start_time);
        MPI_Startall(requests->size(), &(*requests)[0]);
          /* start writing recv_buf */
        timer_stop(&send_to_owner_start_time);
      }

      /* recv models from the owners */
      //MPI_Startall(sgd_comm.request_lens[stratum], sgd_comm.requests[0][stratum] + sgd_comm.request_lens[stratum]);
      //timer_start(&testall2_time);
      //MPI_Testall(sgd_comm.request_lens[stratum], sgd_comm.requests[0][stratum] + sgd_comm.request_lens[stratum], flags, MPI_STATUSES_IGNORE);
      //timer_stop(&testall2_time);

      idx_t nslices;
      if(ws->csf) {
        nslices = stratum->tile_csf->pt->nfibs[0];
      }

      /* overlap shuffle with send/recv */
      if(ws->rand_per_iteration || e == 1) {
        timer_stop(&comm_with_owner_time);
        timer_start(&ws->shuffle_time);
        if(ws->csf) {
          shuffle_idx(stratum->perm, nslices);
        }
        else {
          idx_t *arr = stratum->perm;
          idx_t N = stratum->tile->nnz;
          for(idx_t n=N/2; n < N; ++n) {
            idx_t j = (rand_idx()%(N - n)) + n;

            idx_t const tmp = arr[n];
            arr[n] = arr[j];
            arr[j] = tmp;
          }
          //shuffle_idx(stratum->perm, sgd_comm.stratums[stratum].tile->nnz);
        }
        timer_stop(&ws->shuffle_time);
        timer_start(&comm_with_owner_time);
      }

      requests = &stratum->recv_from_owner_requests;
      if (!requests->empty()) {
        timer_start(&recv_from_owner_wait_time);
        MPI_Waitall(requests->size(), &(*requests)[0], MPI_STATUSES_IGNORE);
          /* wait writing external model */
        timer_stop(&recv_from_owner_wait_time);
      }

      if(s != nstratum - 1) {
        int next_stratum = stratum_perm[s + 1];
        requests = &sgd_comm.stratums[next_stratum].recv_from_owner_requests;
        if(!requests->empty()) {
          timer_start(&recv_from_owner_start_time);
          MPI_Startall(requests->size(), &(*requests)[0]);
            /* start writing external model */
          timer_stop(&recv_from_owner_start_time);
        }
      }

      timer_stop(&comm_with_owner_time);

      {
        /*stringstream stream;
        stream << "[" << rank << "] stratum=" << stratum_perm[s];
        for(int i=0; i < stratum->tile->dims[2]; ++i) {
          stream << " (";
          for(int f=0; f < nfactors; ++f) {
            stream << " " << model_compacted->factors[2][i*nfactors + f];
          }
          stream << ")";
        }
        printf("%s\n", stream.str().c_str());*/
      }

      timer_start(&compute_time);
#ifdef SPLATT_MEASURE_LOAD_IMBALANCE
      double t = omp_get_wtime();
#endif
      /* update model for this stratum */
      if(ws->csf) {
        if(nmodes == 3) {
#pragma omp parallel for
          for(idx_t n=0; n < nslices; ++n) {
            idx_t real_n = stratum->perm[n];
            p_update_model_csf3(stratum->tile_csf, real_n, model_compacted, ws);
          }
        }
        else if(nmodes == 4) {
#pragma omp parallel for
          for(idx_t n=0; n < nslices; ++n) {
            idx_t real_n = stratum->perm[n];
            p_update_model_csf4(stratum->tile_csf, real_n, model_compacted, ws);
          }
        }
        else if(nmodes == 5) {
#pragma omp parallel for
          for(idx_t n=0; n < nslices; ++n) {
            idx_t real_n = stratum->perm[n];
            p_update_model_csf5(stratum->tile_csf, real_n, model_compacted, ws);
          }
        }
        else if(nmodes == 6) {
#pragma omp parallel for
          for(idx_t n=0; n < nslices; ++n) {
            idx_t real_n = stratum->perm[n];
            p_update_model_csf6(stratum->tile_csf, real_n, model_compacted, ws);
          }
        }
        else {
          assert(false);
        }
      }
      else {
        if(nmodes == 3) {
#pragma omp parallel for
          for(idx_t n=0; n < stratum->tile->nnz; ++n) {
            idx_t real_n = stratum->perm[n];
            p_update_model3(stratum->tile, real_n, model_compacted, ws);
          }
        }
        else if (nmodes == 4) {
#pragma omp parallel for
          for(idx_t n=0; n < stratum->tile->nnz; ++n) {
            idx_t real_n = stratum->perm[n];
            p_update_model4(stratum->tile, real_n, model_compacted, ws);
          }
        }
        else if (nmodes == 5) {
#pragma omp parallel for
          for(idx_t n=0; n < stratum->tile->nnz; ++n) {
            idx_t real_n = stratum->perm[n];
            p_update_model5(stratum->tile, real_n, model_compacted, ws);
          }
        }
        else if (nmodes == 6) {
#pragma omp parallel for
          for(idx_t n=0; n < stratum->tile->nnz; ++n) {
            idx_t real_n = stratum->perm[n];
            p_update_model6(stratum->tile, real_n, model_compacted, ws);
          }
        }
        else {
#pragma omp parallel for
          for(idx_t n=0; n < stratum->tile->nnz; ++n) {
            idx_t real_n = stratum->perm[n];
            p_update_model(stratum->tile, real_n, model_compacted, ws);
          }
        }
      }
#ifdef SPLATT_MEASURE_LOAD_IMBALANCE
      compute_times[stratum_perm[s]].push_back(omp_get_wtime() - t);
#endif

      {
        /*stringstream stream;
        stream << "![" << rank << "] stratum=" << stratum_perm[s];
        for(int i=0; i < stratum->tile->dims[2]; ++i) {
          stream << " (";
          for(int f=0; f < nfactors; ++f) {
            stream << " " << model_compacted->factors[2][i*nfactors + f];
          }
          stream << ")";
        }
        printf("%s\n", stream.str().c_str());*/
      }

      timer_stop(&compute_time);

      timer_start(&comm_with_owner_time);

      if(s > 0) {
        int prev_stratum = stratum_perm[s - 1];
        requests = &sgd_comm.stratums[prev_stratum].send_to_owner_requests;
        if(!requests->empty()) {
          timer_start(&send_to_owner_wait_time);
          MPI_Waitall(requests->size(), &(*requests)[0], MPI_STATUSES_IGNORE);
            /* wait reading external model */
          timer_stop(&send_to_owner_wait_time);
        }
      }

      /* send models back to the owners */
      requests = &stratum->send_to_owner_requests;
      if(!requests->empty()) {
        timer_start(&send_to_owner_start_time);
        MPI_Startall(requests->size(), &(*requests)[0]);
          /* start reading external model */
        timer_stop(&send_to_owner_start_time);
      }
      //timer_start(&testall3_time);
      //MPI_Testall(sgd_comm.request_lens[stratum]*2, sgd_comm.requests[1][stratum], flags, MPI_STATUSES_IGNORE);
      //timer_stop(&testall3_time);

      /* overlap shuffle with send/recv */
      if(!ws->csf && (ws->rand_per_iteration || e == 1) && s < nstratum - 1) {
        timer_stop(&comm_with_owner_time);
        timer_start(&ws->shuffle_time);
        stratum_t *next_stratum = &sgd_comm.stratums[stratum_perm[s + 1]];
        idx_t *arr = next_stratum->perm;
        idx_t N = next_stratum->tile->nnz;
        for(idx_t n=0; n < N/2; ++n) {
          idx_t j = (rand_idx()%(N - n)) + n;

          idx_t const tmp = arr[n];
          arr[n] = arr[j];
          arr[j] = tmp;
        }
        timer_stop(&ws->shuffle_time);
        timer_start(&comm_with_owner_time);
      }

      /* can delay wait for send because we're sending from a separate buffer */
      requests = &stratum->send_to_leasers_requests;
      if(!requests->empty()) {
        timer_start(&recv_from_owner_wait_time);
        MPI_Waitall(requests->size(), &(*requests)[0], MPI_STATUSES_IGNORE);
          /* wait reading send_buf */
        timer_stop(&recv_from_owner_wait_time);
      }

      if(s == nstratum - 1) {
        requests = &stratum->send_to_owner_requests;
        if(!requests->empty()) {
          timer_start(&send_to_owner_wait_time);
          MPI_Waitall(requests->size(), &(*requests)[0], MPI_STATUSES_IGNORE);
            /* wait reading external model */
          timer_stop(&send_to_owner_wait_time);
        }
      }

      /* Scale local updates before reduction, if needed */
      for(int m=1; m < nmodes; ++m) {
        stratum_mode_t *mode_info = &stratum->mode_infos[m];
        if(mode_info->is_local && !mode_info->leasers.empty() && !mode_info->nnzs_of_non_empty_slice[0].empty()) {
          //printf("[%d] %s:%d %ld:%d\n", rank, __FILE__, __LINE__, stratum_perm[s], m);
          assert(mode_info->nnzs_of_non_empty_slice[0].size() <= 1);
          //std::stringstream stream;
          //stream << "[" << rank << "] " << itr->first << ":";
          for(int f=0; f < nfactors; ++f) {
            //stream << " " << model_compacted->factors[m][local_idx*nfactors + f] << "*" << mode_info->local_weight;
            model_compacted->factors[m][f] *=
              mode_info->local_weight;
          }
          //printf("%s\n", stream.str().c_str());
        }
      }

      /* each owner scatters received compact data to their original locations */
      /* TODO: a owner may receive multiple updates when we replicate models for short modes */
      request_idx = 0;
      for(int m=1; m < nmodes; ++m) {
        stratum_mode_t *mode_info = &stratum->mode_infos[m];
        int nleasers = mode_info->leasers.size();
        if(nleasers == 0) continue;

        timer_start(&send_to_owner_wait_time);
        MPI_Waitall(
          nleasers,
          &stratum->recv_from_leasers_requests[request_idx],
          MPI_STATUS_IGNORE);
          /* wait writing recv_buf */
        timer_stop(&send_to_owner_wait_time);
        request_idx += nleasers;

        idx_t buf_offset = 0;
        for(int leaser=0; leaser < nleasers; ++leaser) {
          timer_start(&scatter_time);
          idx_t len = mode_info->local_rows_to_send[leaser].size();
          if(len > 0 && (nleasers > 1 || mode_info->is_local && nleasers > 0)) {
            assert(len <= 1);
            //printf("[%d<-%d] %s:%d %ld\n", rank, mode_info->leasers[leaser], __FILE__, __LINE__, stratum_perm[s]);
            val_t weight = mode_info->leaser_weights[leaser];
            //std::stringstream stream;
            //stream << "[" << rank << "] " << (local_idx + rinfo->layer_ptrs[m][rank]) << ":";
            if(weight < 0) {
              /* first reduction */
              for(idx_t f=0; f < nfactors; ++f) {
                //stream << " " << sgd_comm.mode_infos[m].recv_buf[(i + buf_offset)*nfactors + f] << "*" << (-weight);
                model_compacted->factors[m][f] =
                  sgd_comm.mode_infos[m].recv_buf[buf_offset*nfactors + f]*(-weight);
              }
            }
            else {
              for(idx_t f=0; f < nfactors; ++f) {
                //stream << " +" << sgd_comm.mode_infos[m].recv_buf[(i + buf_offset)*nfactors + f] << "*" << weight;
                model_compacted->factors[m][f] +=
                  sgd_comm.mode_infos[m].recv_buf[buf_offset*nfactors + f]*weight;
              }
            }
            //printf("%s\n", stream.str().c_str());
          }
          else {
#pragma omp parallel for
            for(idx_t i=0; i < len; ++i) {
              idx_t local_idx = mode_info->local_rows_to_send[leaser][i];
              assert(local_idx < nlocalrows[m]);
              for(idx_t f=0; f < nfactors; ++f) {
                model_compacted->factors[m][local_idx*nfactors + f] =
                  sgd_comm.mode_infos[m].recv_buf[(i + buf_offset)*nfactors + f];
              }
            }
          }
          buf_offset += len;
          timer_stop(&scatter_time);
        }
      }

      timer_stop(&comm_with_owner_time);
    } /* for each stratum */

    /* collect portions of model required for convergence check from the owners */
    if((npes - 1)*(nmodes - 1) > 0) {
      timer_start(&bcast_time);
      MPI_Startall((npes - 1)*(nmodes - 1), &sgd_comm.requests[0]);
      for(int m=1; m < nmodes; ++m) {
        sgd_comm_mode_t *mode_info = &sgd_comm.mode_infos[m];
        idx_t base = sgd_rinfo.layer_ptrs[m][rank];

        idx_t offset = 0;
        for(int p=0; p < npes; ++p) {
          if(p == rank) continue;
#pragma omp parallel for
          for(idx_t i=0; i < mode_info->local_rows_to_send[p].size(); ++i) {
            idx_t local_idx = mode_info->local_rows_to_send[p][i];
            for(int f=0; f < nfactors; ++f) {
              mode_info->send_buf[(i + offset)*nfactors + f] = model_compacted->factors[m][local_idx*nfactors + f];
            }
            assert(local_idx < nlocalrows[m]);
          }
          offset += mode_info->local_rows_to_send[p].size();
        }
      }

      MPI_Startall((npes - 1)*(nmodes - 1), &sgd_comm.requests[(npes - 1)*(nmodes - 1)]);
      MPI_Waitall(2*(npes - 1)*(nmodes - 1), &sgd_comm.requests[0], MPI_STATUSES_IGNORE);
      timer_stop(&bcast_time);
    }

    timer_stop(&ws->train_time);

    /* compute RMSE and adjust learning rate */
    timer_start(&ws->test_time);
    loss = tc_loss_sq(sgd_comm.compact_train, model_compacted, ws);
    frobsq = p_frob_sq(&sgd_comm, model_compacted, ws);
    obj = loss + frobsq;
    if(tc_converge(sgd_comm.compact_train, sgd_comm.compact_validate, model_compacted, loss, frobsq, e, ws)) {
      timer_stop(&ws->test_time);
      ++e;
      break;
    }
    timer_stop(&ws->test_time);

    /* bold driver */
    if(e > 1) {
      if(obj < prev_obj) {
        ws->learn_rate *= 1.05;
      } else {
        ws->learn_rate *= 0.50;
      }
    }

    prev_obj = obj;
  }

  --e;

  if(rank == 0) {
    printf("   train_time %g (%g)\n", ws->train_time.seconds, ws->train_time.seconds/e);
    printf("     update_time %g (%g)\n", compute_time.seconds, compute_time.seconds/e);
    printf("     shuffle_time %g (%g)\n", ws->shuffle_time.seconds, ws->shuffle_time.seconds/e);
    printf("     comm_with_owner_time %g (%g)\n", comm_with_owner_time.seconds, comm_with_owner_time.seconds/e);
    printf("       recv_from_owner_start_time %g (%g)\n", recv_from_owner_start_time.seconds, recv_from_owner_start_time.seconds/e);
    printf("       recv_from_owner_wait_time %g (%g)\n", recv_from_owner_wait_time.seconds, recv_from_owner_wait_time.seconds/e);
    printf("       send_to_owner_start_time %g (%g)\n", send_to_owner_start_time.seconds, send_to_owner_start_time.seconds/e);
    printf("       send_to_owner_wait_time %g (%g)\n", send_to_owner_wait_time.seconds, send_to_owner_wait_time.seconds/e);
    //printf("       testall1_time %g\n", testall1_time.seconds);
    //printf("       testall2_time %g\n", testall2_time.seconds);
    //printf("       testall3_time %g\n", testall3_time.seconds);
    printf("       gather_time %g (%g)\n", gather_time.seconds, gather_time.seconds/e);
    printf("       scatter_time %g (%g)\n", scatter_time.seconds, scatter_time.seconds/e);
    printf("     bcast_time %g (%g)\n", bcast_time.seconds, bcast_time.seconds/e);
    printf("   test_time %g (%g)\n", ws->test_time.seconds, ws->test_time.seconds/e);
  }

#ifdef SPLATT_MEASURE_LOAD_IMBALANCE
  val_t global_compute_times[npes][nstratum][e];
  for(int p=0; p < npes; ++p) {
    for(int s=0; s < nstratum; ++s) {
      assert(compute_times[s].size() == e);
      if(p == rank) {
        for(int i=0; i < e; ++i) {
          global_compute_times[p][s][i] = compute_times[s][i];
        }
      }
      MPI_Bcast(global_compute_times[p][s], e, SPLATT_MPI_VAL, p, MPI_COMM_WORLD);
    }
  }

  if(rank == 0) {
    double avg_total = 0;
    double maximum_total = 0;
    for(int s=0; s < nstratum; ++s) {
      double avg_stratum_total = 0;
      double maximum_stratum_total = 0;
      for(int i=0; i < e; ++i) {
        double sum = 0;
        double maximum = 0;
        for(int p=0; p < npes; ++p) {
          sum += global_compute_times[p][s][i];
          maximum = SS_MAX(maximum, global_compute_times[p][s][i]);
        }
        double avg = (double)sum/npes;
        avg_stratum_total += avg;
        maximum_stratum_total += maximum;
      }
      avg_stratum_total /= e;
      maximum_stratum_total /= e;
      avg_total += avg_stratum_total;
      maximum_total += maximum_stratum_total;
      printf("stratum %d avg_time %g max_time %g load_imbalance %g\n", s, avg_stratum_total, maximum_stratum_total, maximum_stratum_total/avg_stratum_total);
    }
    printf("total load_imbalance %g\n", maximum_total/avg_total);
  }
#endif

  p_free_sgd_comm(&sgd_comm);

  for(int m=0; m < nmodes; ++m) {
    splatt_free(sgd_rinfo.layer_ptrs[m]);
  }
#else

  splatt_csf *csf;
  idx_t *perm;
  if(ws->csf) {
    /* convert training data to a single CSF */
    double * opts = splatt_default_opts();
    opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
    splatt_csf * csf = splatt_malloc(sizeof(*csf));
    csf_alloc_mode(train, CSF_SORTED_BIGFIRST, 0, csf, opts);

    assert(csf->ntiles == 1);

    idx_t const nslices = csf[0].pt->nfibs[0];
    perm = splatt_malloc(nslices * sizeof(*perm));

    for(idx_t n=0; n < nslices; ++n) {
      perm[n] = n;
    }
  }
  else {
    /* initialize perm */
    perm = splatt_malloc(train->nnz * sizeof(*perm));
    for(idx_t n=0; n < train->nnz; ++n) {
      perm[n] = n;
    }
  }

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  /* for bold driver */
  val_t obj = loss + frobsq;
  val_t prev_obj = obj;

  timer_start(&ws->tc_time);
  /* foreach epoch */
  for(idx_t e=1; e < ws->max_its+1; ++e) {

    timer_start(&ws->train_time);

    /* update model from all training observations */
    timer_start(&ws->shuffle_time);
    shuffle_idx(perm, nslices);
    timer_stop(&ws->shuffle_time);

    if(ws->csf) {
#pragma omp parallel for
      for(idx_t i=0; i < nslices; ++i) {
        p_update_model_csf3(csf, perm_i[i], model, ws);
      }
    }
    else {
#pragma omp parallel for
      for(idx_t n=0; n < train->nnz; ++n) {
        p_update_model(train, perm[n], model, ws);
      }
    }

    timer_stop(&ws->train_time);

    /* compute RMSE and adjust learning rate */
    timer_start(&ws->test_time);
    loss = tc_loss_sq(train, model, ws);
    frobsq = tc_frob_sq(model, ws);
    obj = loss + frobsq;
    if(tc_converge(train, validate, model, loss, frobsq, e, ws)) {
      timer_stop(&ws->test_time);
      break;
    }
    timer_stop(&ws->test_time);

    /* bold driver */
    if(e > 1) {
      if(obj < prev_obj) {
        ws->learn_rate *= 1.05;
      } else {
        ws->learn_rate *= 0.50;
      }
    }

    prev_obj = obj;
  }

  printf("   train_time %g (%g)\n", ws->train_time.seconds, ws->train_time.seconds/e);
  printf("     update_time %g (%g)\n", (ws->train_time.seconds - ws->shuffle_time.seconds).seconds, (ws->train_time.seconds - ws->shuffle_time.seconds)/e);
  printf("     shuffle_time %g (%g)\n", ws->shuffle_time.seconds, ws->shuffle_time.seconds/e);
  printf("   test_time %g (%g)\n", ws->test_time.seconds, ws->test_time.seconds/e);

#if USE_CSF_SGD
  splatt_free(perm_i);
  csf_free_mode(csf);
  splatt_free(csf);
  splatt_free_opts(opts);
#else
  splatt_free(perm);
#endif

#endif /* !SPLATT_USE_MPI */
}
