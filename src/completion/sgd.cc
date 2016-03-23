

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
#include <unordered_map>

#define USE_CSF_SGD 0

using std::set;
using std::unordered_map;

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
    val_t const moda = (loss * brow[f] * crow[f]) - (reg[0] * arow[f]);
    val_t const modb = (loss * arow[f] * crow[f]) - (reg[1] * brow[f]);
    val_t const modc = (loss * arow[f] * brow[f]) - (reg[2] * crow[f]);
    arow[f] += rate * moda;
    assert(std::isfinite(arow[f]));
    brow[f] += rate * modb;
    assert(std::isfinite(brow[f]));
    crow[f] += rate * modc;
    assert(std::isfinite(crow[f]));
  }
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
        val_t const moda = (loss * brow[f] * crow[f]) - (areg * arow[f]);
        val_t const modb = (loss * arow[f] * crow[f]) - (breg * brow[f]);
        val_t const modc = (loss * arow[f] * brow[f]) - (creg * crow[f]);
        arow[f] += rate * moda;
        brow[f] += rate * modb;
        crow[f] += rate * modc;
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

typedef struct
{
  int nstratum;
  tc_model *model;
  rank_info *rinfo;

  idx_t **local_to_global_lens;

  /* mapping from compacted indices within each remote tile to global indices.
   * nstratum*nmodes*l where l is specified by remote_to_global_lens[s][m]
   */
  idx_t ***remote_to_globals;
  idx_t **remote_to_global_lens;

  unordered_map<idx_t, idx_t> **global_to_locals;
    /**< inverse mapping of local_to_globals */

  idx_t ***aggregated_local_to_globals;
  idx_t **aggregated_local_to_global_lens;

  /*
   * npes*nmodes*l array where l is specified by aggregated_remote_to_global_lens
   */
  idx_t ***aggregated_remote_to_globals;
  idx_t **aggregated_remote_to_global_lens;

  unordered_map<idx_t, idx_t> *aggregated_global_to_locals;
    /**< inverse mapping of aggregated_local_to_globals */

  val_t **send_recv_buf; /* nmodes arrays */

  MPI_Request **requests[2];
    /* 2*nstratum*(nmodes*2) array */
    /* requests[0] is for receiving models from owners */
    /* requests[1] is for sending models back to owners */
  int *request_lens;

  MPI_Request *aggregated_requests;
    /* length 2*(npes-1)*(nmodes-1) array */
} sgd_comm_t;

sgd_comm_t sgd_comm;

static int p_get_nstratum(int nmodes, int npes)
{
  int nstratum = 1;
  for(int m=1; m < nmodes; ++m) {
    nstratum *= npes;
  }
  return nstratum;
}

static int p_find_stratum_of(const sptensor_t *tensor, idx_t idx, sgd_comm_t *sgd_comm)
{
  rank_info *rinfo = sgd_comm->rinfo;
  int rank = rinfo->rank;
  int npes = rinfo->npes;
  int nmodes = tensor->nmodes;

  int my_stratum = -1;
  for(int stratum=0; stratum < sgd_comm->nstratum; ++stratum) {
    bool in_tile = true;
    int stratum_temp = stratum;
    for(int m=1; m < nmodes; ++m) {
      int owner = (stratum_temp%npes + rank)%npes;
      stratum_temp /= npes;

      if(tensor->ind[m][idx] < rinfo->layer_ptrs[m][owner] ||
        tensor->ind[m][idx] >= rinfo->layer_ptrs[m][owner + 1]) {
        in_tile = false;
        break;
      }
    }
    if (in_tile) {
      assert(my_stratum == -1);
      my_stratum = stratum;
    }
  }

  assert(my_stratum != -1);
  return my_stratum;
}

static set<idx_t> **p_count_nnz_of_tiles(
  idx_t **nnzs, sgd_comm_t *sgd_comm, sptensor_t *train)
{
  rank_info *rinfo = sgd_comm->rinfo;
  int npes = rinfo->npes;
  int rank = rinfo->rank;
  int nstratum = sgd_comm->nstratum;
  int nmodes = train->nmodes;

  *nnzs = (idx_t *)splatt_malloc(sizeof(idx_t) * nstratum);
  for(idx_t stratum=0; stratum < nstratum; ++stratum) {
    (*nnzs)[stratum] = 0;
  }

  /* count nnz of each tile */
  set<idx_t> **uniq_ids = (set<idx_t> **)splatt_malloc(sizeof(set<idx_t> *)*nstratum);
  for(int stratum=0; stratum < nstratum; ++stratum) {
    uniq_ids[stratum] = new set<idx_t>[nmodes];
  }

  for(idx_t n=0; n < train->nnz; ++n) {
    int stratum = p_find_stratum_of(train, n, sgd_comm);
    int stratum_temp = stratum;
    for(int m=1; m < nmodes; ++m) {
      if (stratum_temp%npes != 0) {
        uniq_ids[stratum][m].insert(train->ind[m][n]);
      }
      stratum_temp /= npes;
    }

    ++(*nnzs)[stratum];
  }

  return uniq_ids;
}

set<idx_t> **aggregate_uniq_ids(set<idx_t> **uniq_ids, sgd_comm_t *sgd_comm, const sptensor_t *validate)
{
  rank_info *rinfo = sgd_comm->rinfo;
  int rank = rinfo->rank;
  int npes = rinfo->npes;
  int nstratum = sgd_comm->nstratum;
  int nmodes = sgd_comm->model->nmodes;

  set<idx_t> **aggregated_uniq_ids = (set<idx_t> **)splatt_malloc(sizeof(set<idx_t> *)*npes);
  for(int p=0; p < npes; ++p) {
    aggregated_uniq_ids[p] = new set<idx_t>[nmodes];
  }

  for(int stratum=0; stratum < nstratum; ++stratum) {
    int stratum_temp = stratum;
    for(idx_t m=1; m < nmodes; ++m) {
      int stratum_offset = stratum_temp%npes;
      stratum_temp /= npes;
      if(stratum_offset == 0) continue;

      int owner = (stratum_offset + rank)%npes;

      for(set<idx_t>::iterator itr=uniq_ids[stratum][m].begin(); itr != uniq_ids[stratum][m].end(); ++itr) {
        aggregated_uniq_ids[owner][m].insert(*itr);
      }
    }
  }

  for(idx_t n=0; n < validate->nnz; ++n) {
    int stratum = p_find_stratum_of(validate, n, sgd_comm);
    for(int m=1; m < nmodes; ++m) {
      int stratum_offset = stratum%npes;
      if (stratum_offset != 0) {
        int owner = (stratum_offset + rank)%npes;
        aggregated_uniq_ids[owner][m].insert(validate->ind[m][n]);
      }
      stratum /= npes;
    }
  }

  return aggregated_uniq_ids;
}

static void p_map_global_to_local(
  sgd_comm_t *sgd_comm,
  set<idx_t> **uniq_ids,
  set<idx_t> **aggregated_uniq_ids)
{
  int nmodes = sgd_comm->model->nmodes;
  int npes = sgd_comm->rinfo->npes;
  int rank = sgd_comm->rinfo->rank;
  int nstratum = p_get_nstratum(nmodes, npes);

  /* mapping from local compacted indices within each tile to global indices.
   * nstratum*nmodes*l where l is specified by local_to_global_lens[s][m]
   */
  idx_t ***local_to_globals = (idx_t ***)splatt_malloc(sizeof(idx_t **)*nstratum);
  sgd_comm->local_to_global_lens = (idx_t **)splatt_malloc(sizeof(idx_t *)*nstratum);

  sgd_comm->remote_to_globals = (idx_t ***)splatt_malloc(sizeof(idx_t **)*nstratum);
  sgd_comm->remote_to_global_lens = (idx_t **)splatt_malloc(sizeof(idx_t *)*nstratum);

  sgd_comm->global_to_locals = (unordered_map<idx_t, idx_t> **)splatt_malloc(sizeof(unordered_map<idx_t, idx_t> *)*nstratum);

  for(idx_t stratum=0; stratum < nstratum; ++stratum) {
    local_to_globals[stratum] = (idx_t **)splatt_malloc(sizeof(idx_t *)*nmodes);
    sgd_comm->local_to_global_lens[stratum] = (idx_t *)splatt_malloc(sizeof(idx_t)*nmodes);

    sgd_comm->remote_to_globals[stratum] = (idx_t **)splatt_malloc(sizeof(idx_t *)*nmodes);
    sgd_comm->remote_to_global_lens[stratum] = (idx_t *)splatt_malloc(sizeof(idx_t)*nmodes);

    sgd_comm->global_to_locals[stratum] = new unordered_map<idx_t, idx_t>[nmodes];

    for(int m=1; m < nmodes; ++m) {
      local_to_globals[stratum][m] = (idx_t *)splatt_malloc(sizeof(idx_t)*uniq_ids[stratum][m].size());
      sgd_comm->local_to_global_lens[stratum][m] = uniq_ids[stratum][m].size();

      idx_t i = 0;
      for(set<idx_t>::iterator itr = uniq_ids[stratum][m].begin(); itr != uniq_ids[stratum][m].end(); ++itr, ++i) {
        local_to_globals[stratum][m][i] = *itr;
        sgd_comm->global_to_locals[stratum][m][*itr] = i;
      }
    }
  }

  MPI_Request send_request, recv_request;

  for(int stratum=0; stratum < nstratum; ++stratum) {
    int stratum_temp = stratum;
    for(int m=1; m < nmodes; ++m) {
      int stratum_offset = stratum_temp%npes;
      stratum_temp /= npes;
      if(stratum_offset == 0) {
        sgd_comm->local_to_global_lens[stratum][m] = 0;
        sgd_comm->remote_to_global_lens[stratum][m] = 0;
        continue;
      }

      int owner = (stratum_offset + rank)%npes;
      idx_t send_len = sgd_comm->local_to_global_lens[stratum][m];
      MPI_Isend(
        &send_len, 1, SPLATT_MPI_IDX, owner, m, MPI_COMM_WORLD, &send_request);

      int leaser = (npes - stratum_offset + rank)%npes;
      idx_t recv_len = 0;
      MPI_Irecv(
        &recv_len, 1, SPLATT_MPI_IDX, leaser, m, MPI_COMM_WORLD, &recv_request);

      MPI_Wait(&send_request, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

      sgd_comm->remote_to_global_lens[stratum][m] = recv_len;

      MPI_Isend(
        local_to_globals[stratum][m],
        send_len,
        SPLATT_MPI_IDX,
        owner,
        m,
        MPI_COMM_WORLD,
        &send_request);

      sgd_comm->remote_to_globals[stratum][m] = (idx_t *)splatt_malloc(sizeof(idx_t)*recv_len);
      MPI_Irecv(
        sgd_comm->remote_to_globals[stratum][m],
        recv_len,
        SPLATT_MPI_IDX,
        leaser,
        m,
        MPI_COMM_WORLD,
        &recv_request);

      MPI_Wait(&send_request, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
    }
  } /* for each stratum */

  for(int stratum=0; stratum < sgd_comm->nstratum; ++stratum) {
    for(int m=1; m < sgd_comm->model->nmodes; ++m) {
      splatt_free(local_to_globals[stratum][m]);
    }
    splatt_free(local_to_globals[stratum]);
  }
  splatt_free(local_to_globals);

  /*
   * npes*nmodes*l array where l is specified by aggregated_local_to_global_lens
   */
  idx_t ***aggregated_local_to_globals = (idx_t ***)splatt_malloc(sizeof(idx_t *)*npes);
  sgd_comm->aggregated_local_to_globals = aggregated_local_to_globals;
  sgd_comm->aggregated_local_to_global_lens = (idx_t **)splatt_malloc(sizeof(idx_t *)*npes);
  
  sgd_comm->aggregated_remote_to_globals = (idx_t ***)splatt_malloc(sizeof(idx_t *)*npes);
  sgd_comm->aggregated_remote_to_global_lens = (idx_t **)splatt_malloc(sizeof(idx_t *)*npes);

  sgd_comm->aggregated_global_to_locals = new unordered_map<idx_t, idx_t>[nmodes];

  for(idx_t p=0; p < npes; ++p) {
    aggregated_local_to_globals[p] = (idx_t **)splatt_malloc(sizeof(idx_t *)*nmodes);
    sgd_comm->aggregated_local_to_global_lens[p] = (idx_t *)splatt_malloc(sizeof(idx_t)*nmodes);

    sgd_comm->aggregated_remote_to_globals[p] = (idx_t **)splatt_malloc(sizeof(idx_t *)*nmodes);
    sgd_comm->aggregated_remote_to_global_lens[p] = (idx_t *)splatt_malloc(sizeof(idx_t)*nmodes);

    for(int m=1; m < nmodes; ++m) {
      aggregated_local_to_globals[p][m] = (idx_t *)splatt_malloc(sizeof(idx_t)*aggregated_uniq_ids[p][m].size());
      sgd_comm->aggregated_local_to_global_lens[p][m] = aggregated_uniq_ids[p][m].size();
    }
  }
  
  for(int m=1; m < nmodes; ++m) {
    idx_t i = 0;
    for(idx_t p=0; p < npes; ++p) {
      if (p == rank) assert(aggregated_uniq_ids[p][m].empty());

      idx_t i_base = i;
      for(set<idx_t>::iterator itr = aggregated_uniq_ids[p][m].begin(); itr != aggregated_uniq_ids[p][m].end(); ++itr, ++i) {
        aggregated_local_to_globals[p][m][i - i_base] = *itr;
        sgd_comm->aggregated_global_to_locals[m][*itr] = i;
      }
    }
  }

  for(int p=0; p < npes; ++p) {
    for(int m=1; m < nmodes; ++m) {
      idx_t send_len = sgd_comm->aggregated_local_to_global_lens[p][m];
      MPI_Isend(
        &send_len, 1, SPLATT_MPI_IDX, p, m, MPI_COMM_WORLD, &send_request);

      idx_t recv_len = 0;
      MPI_Irecv(
        &recv_len, 1, SPLATT_MPI_IDX, p, m, MPI_COMM_WORLD, &recv_request);

      MPI_Wait(&send_request, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

      sgd_comm->aggregated_remote_to_global_lens[p][m] = recv_len;

      MPI_Isend(
        aggregated_local_to_globals[p][m],
        send_len, SPLATT_MPI_IDX, p, m, MPI_COMM_WORLD, &send_request);

      sgd_comm->aggregated_remote_to_globals[p][m] = (idx_t *)splatt_malloc(sizeof(idx_t)*recv_len);
      MPI_Irecv(
        sgd_comm->aggregated_remote_to_globals[p][m],
        recv_len, SPLATT_MPI_IDX, p, m, MPI_COMM_WORLD, &recv_request);

      MPI_Wait(&send_request, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
    }
  }

  /*for(int p=0; p < npes; ++p) {
    for(int m=1; m < nmodes; ++m) {
      splatt_free(aggregated_local_to_globals[p][m]);
    }
    splatt_free(aggregated_local_to_globals[p]);
  }
  splatt_free(aggregated_local_to_globals);*/
}

static void p_populate_tiles(tc_ws *ws, sptensor_t *train, sgd_comm_t *sgd_comm, idx_t *nnzs)
{
  rank_info *rinfo = sgd_comm->rinfo;
  int npes = rinfo->npes;
  int rank = rinfo->rank;
  int nstratum = sgd_comm->nstratum;
  int nmodes = train->nmodes;

  /* allocate a tile for each stratum */
  ws->tiles = (sptensor_t **)splatt_malloc(sizeof(sptensor_t) * nstratum);  
  for(idx_t stratum=0; stratum < nstratum; ++stratum) {
    ws->tiles[stratum] = tt_alloc(nnzs[stratum], nmodes);
  }
  ws->external_tile = tt_alloc(train->nnz, nmodes);
  ws->external_tile->vals = train->vals;
  ws->external_tile->ind[0] = train->ind[0];

  for(idx_t stratum=0; stratum < nstratum; ++stratum) {
    nnzs[stratum] = 0;
  }

  idx_t nrows[nmodes];
  for(idx_t m=0; m < nmodes; ++m) {
    nrows[m] = rinfo->layer_ptrs[m][rank + 1] - rinfo->layer_ptrs[m][rank];
  }

  for(idx_t n=0; n < train->nnz; ++n) {
    /* find tile id */
    int stratum = p_find_stratum_of(train, n, sgd_comm);

    ws->tiles[stratum]->vals[nnzs[stratum]] = train->vals[n];
    ws->tiles[stratum]->ind[0][nnzs[stratum]] = train->ind[0][n];

    int stratum_temp = stratum;
    for(idx_t m=1; m < nmodes; ++m) {
      int stratum_offset = stratum_temp%npes;
      stratum_temp /= npes;

      if(stratum_offset != 0) {
        assert(sgd_comm->global_to_locals[stratum][m].find(train->ind[m][n]) != sgd_comm->global_to_locals[stratum][m].end());
        ws->tiles[stratum]->ind[m][nnzs[stratum]] = nrows[m] + sgd_comm->global_to_locals[stratum][m][train->ind[m][n]];

        assert(sgd_comm->aggregated_global_to_locals[m].find(train->ind[m][n]) != sgd_comm->aggregated_global_to_locals[m].end());
        ws->external_tile->ind[m][n] = nrows[m] + sgd_comm->aggregated_global_to_locals[m][train->ind[m][n]];
      }
      else {
        ws->tiles[stratum]->ind[m][nnzs[stratum]] = train->ind[m][n] - rinfo->layer_ptrs[m][rank];

        assert(train->ind[m][n] >= rinfo->layer_ptrs[m][rank]);
        assert(train->ind[m][n] < rinfo->layer_ptrs[m][rank + 1]);
        ws->external_tile->ind[m][n] = train->ind[m][n] - rinfo->layer_ptrs[m][rank];
      }
    }
    ++nnzs[stratum];
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

  sgd_comm->send_recv_buf = (val_t **)splatt_malloc(sizeof(val_t *) * nmodes);
  for(int m=1; m < nmodes; ++m) {
    idx_t maximum = 0;
    for(int p=0; p < npes; ++p) {
      if(p != rank) {
        maximum += sgd_comm->aggregated_remote_to_global_lens[p][m];
      }
    }
    for(int stratum=0; stratum < nstratum; ++stratum) {
      maximum = SS_MAX(maximum, sgd_comm->local_to_global_lens[stratum][m]);
      assert(maximum >= sgd_comm->remote_to_global_lens[stratum][m]);
    }
    sgd_comm->send_recv_buf[m] = (val_t *)splatt_malloc(sizeof(val_t)*maximum*nfactors);
  }

  sgd_comm->requests[0] = (MPI_Request **)splatt_malloc(sizeof(MPI_Request *) * nstratum);
  sgd_comm->requests[1] = (MPI_Request **)splatt_malloc(sizeof(MPI_Request *) * nstratum);
  sgd_comm->request_lens = (int *)splatt_malloc(sizeof(int) * nstratum);
  for(int stratum=0; stratum < nstratum; ++stratum) {
    sgd_comm->requests[0][stratum] = (MPI_Request *)splatt_malloc(sizeof(MPI_Request) * (2*nmodes));
    sgd_comm->requests[1][stratum] = (MPI_Request *)splatt_malloc(sizeof(MPI_Request) * (2*nmodes));
  }

  idx_t nrows[nmodes];
  idx_t offsets[nmodes];
  for(idx_t m=0; m < nmodes; ++m) {
    nrows[m] = rinfo->layer_ptrs[m][rank + 1] - rinfo->layer_ptrs[m][rank];
    offsets[m] = 0;
  }

  for(int stratum=0; stratum < nstratum; ++stratum) {
    int request_cnt = 0;
    int stratum_temp = stratum;
    for(int m=1; m < nmodes; ++m) {
      int stratum_offset = stratum_temp%npes;
      stratum_temp /= npes;
      if(stratum_offset == 0) continue;
      ++request_cnt;
    }
    sgd_comm->request_lens[stratum] = request_cnt;

    request_cnt = 0;
    stratum_temp = stratum;
    for(int m=1; m < nmodes; ++m) {
      int stratum_offset = stratum_temp%npes;
      stratum_temp /= npes;
      if(stratum_offset == 0) continue;

      int owner = (stratum_offset + rank)%npes;
      MPI_Recv_init(
        model->factors[m] + nrows[m]*nfactors,
        sgd_comm->local_to_global_lens[stratum][m]*nfactors,
        SPLATT_MPI_VAL,
        owner,
        m,
        MPI_COMM_WORLD,
        sgd_comm->requests[0][stratum] + request_cnt);

      int leaser = (npes - stratum_offset + rank)%npes;
      MPI_Send_init(
        sgd_comm->send_recv_buf[m],
        sgd_comm->remote_to_global_lens[stratum][m]*nfactors,
        SPLATT_MPI_VAL,
        leaser,
        m,
        MPI_COMM_WORLD,
        sgd_comm->requests[0][stratum] + sgd_comm->request_lens[stratum] + request_cnt);

      MPI_Send_init(
        model->factors[m] + nrows[m]*nfactors,
        sgd_comm->local_to_global_lens[stratum][m]*nfactors,
        SPLATT_MPI_VAL,
        owner,
        m,
        MPI_COMM_WORLD,
        sgd_comm->requests[1][stratum] + request_cnt);

      MPI_Recv_init(
        sgd_comm->send_recv_buf[m],
        sgd_comm->remote_to_global_lens[stratum][m]*nfactors,
        SPLATT_MPI_VAL,
        leaser,
        m,
        MPI_COMM_WORLD,
        sgd_comm->requests[1][stratum] + sgd_comm->request_lens[stratum] + request_cnt);

      ++request_cnt;
    }
  }

  sgd_comm->aggregated_requests = (MPI_Request *)splatt_malloc(sizeof(MPI_Request)*2*(npes - 1)*(nmodes - 1));
  int request_cnt = 0;
  for(int m=1; m < nmodes; ++m) {
    idx_t local_offset = 0, remote_offset = 0;
    for(int p=0; p < npes; ++p) {
      if(p == rank) continue;

      MPI_Recv_init(
        model->factors[m] + (nrows[m] + local_offset)*nfactors,
        sgd_comm->aggregated_local_to_global_lens[p][m]*nfactors,
        SPLATT_MPI_VAL,
        p,
        m,
        MPI_COMM_WORLD,
        sgd_comm->aggregated_requests + request_cnt);
      local_offset += sgd_comm->aggregated_local_to_global_lens[p][m];

      MPI_Send_init(
        sgd_comm->send_recv_buf[m] + remote_offset*nfactors,
        sgd_comm->aggregated_remote_to_global_lens[p][m]*nfactors,
        SPLATT_MPI_VAL,
        p,
        m,
        MPI_COMM_WORLD,
        sgd_comm->aggregated_requests + (npes - 1)*(nmodes - 1) + request_cnt);
      remote_offset += sgd_comm->aggregated_remote_to_global_lens[p][m];

      ++request_cnt;
    }
  }
}

void p_free_sgd_comm(sgd_comm_t *sgd_comm)
{
  for(int stratum=0; stratum < sgd_comm->nstratum; ++stratum) {
    splatt_free(sgd_comm->local_to_global_lens[stratum]);
    splatt_free(sgd_comm->remote_to_globals[stratum]);
    splatt_free(sgd_comm->remote_to_global_lens[stratum]);

    delete[] sgd_comm->global_to_locals[stratum];

    splatt_free(sgd_comm->requests[0][stratum]);
    splatt_free(sgd_comm->requests[1][stratum]);
  }
  for(int m=1; m < sgd_comm->model->nmodes; ++m) {
    splatt_free(sgd_comm->send_recv_buf[m]);
  }

  splatt_free(sgd_comm->local_to_global_lens);
  splatt_free(sgd_comm->remote_to_globals);
  splatt_free(sgd_comm->remote_to_global_lens);
  splatt_free(sgd_comm->global_to_locals);

  splatt_free(sgd_comm->requests[0]);
  splatt_free(sgd_comm->requests[1]);
  splatt_free(sgd_comm->request_lens);
  splatt_free(sgd_comm->send_recv_buf);

  for(int p=0; p < sgd_comm->rinfo->npes; ++p) {
    for(int m=1; m < sgd_comm->model->nmodes; ++m) {
      splatt_free(sgd_comm->aggregated_remote_to_globals[p][m]);
    }
    splatt_free(sgd_comm->aggregated_local_to_global_lens[p]);
    splatt_free(sgd_comm->aggregated_remote_to_globals[p]);
    splatt_free(sgd_comm->aggregated_remote_to_global_lens[p]);
  }

  delete[] sgd_comm->aggregated_global_to_locals;
  splatt_free(sgd_comm->aggregated_requests);
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

  idx_t lens[nmodes];
  for(int m=0; m < nmodes; ++m) {
    lens[m] = rinfo->layer_ptrs[m][rank + 1] - rinfo->layer_ptrs[m][rank];
  }

  #pragma omp parallel reduction(+:reg_obj)
  {
    for(idx_t m=0; m < nmodes; ++m) {
      val_t accum = 0;
      val_t const * const restrict mat = model->factors[m];

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < lens[m] * nfactors; ++x) {
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

  idx_t nrows[nmodes];
  for(idx_t m=0; m < nmodes; ++m) {
    nrows[m] = sgd_comm.rinfo->layer_ptrs[m][rank + 1] - sgd_comm.rinfo->layer_ptrs[m][rank];
  }

  for(idx_t m=0; m < nmodes; ++m) {
    if(m == 0) {
      par_memcpy(ws->best_model->factors[m], sgd_comm.model->factors[m], sizeof(val_t)*nrows[m]*nfactors);
    }
    else {
      par_memcpy(ws->best_model->factors[m] + sgd_comm.rinfo->layer_ptrs[m][rank]*nfactors, sgd_comm.model->factors[m], sizeof(val_t)*nrows[m]*nfactors);
      idx_t offset = 0;
      for(int p=0; p < npes; ++p) {
        if(p == rank) continue;
#pragma omp parallel for
        for(int i=0; i < sgd_comm.aggregated_local_to_global_lens[p][m]; ++i) {
          for(int f=0; f < nfactors; ++f) {
            ws->best_model->factors[m][sgd_comm.aggregated_local_to_globals[p][m][i]*nfactors + f] = sgd_comm.model->factors[m][(nrows[m] + offset + i)*nfactors + f];
          }
        }
        offset += sgd_comm.aggregated_local_to_global_lens[p][m];
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
    sgd_rinfo.dims_3d[m] = sgd_rinfo.npes;
    p_find_layer_boundaries(ssizes, m, &sgd_rinfo);
  }
#endif

#if USE_CSF_SGD
  /* convert training data to a single CSF */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  splatt_csf * csf = (splatt_csf *)splatt_malloc(sizeof(*csf));
  csf_alloc_mode(train, CSF_SORTED_BIGFIRST, 0, csf, opts);

  assert(csf->ntiles == 1);

  idx_t const nslices = csf[0].pt->nfibs[0];
  idx_t * perm_i = (idx_t *)splatt_malloc(nslices * sizeof(*perm_i));

  for(idx_t n=0; n < nslices; ++n) {
    perm_i[n] = n;
  }
#else
  int npes = rinfo->npes;
  int rank = rinfo->rank;
  int nmodes = train->nmodes;
  int nstratum = p_get_nstratum(nmodes, npes);

  sgd_comm.nstratum = nstratum;
  sgd_comm.rinfo = &sgd_rinfo;
  sgd_comm.model = model;

  /* count nnz of each tile */
  idx_t *nnzs;
  std::set<idx_t> **uniq_ids = p_count_nnz_of_tiles(&nnzs, &sgd_comm, train);
  std::set<idx_t> **aggregated_uniq_ids = aggregate_uniq_ids(uniq_ids, &sgd_comm, validate);

  /* mapping from global to local compacted indices within each tile */
  p_map_global_to_local(&sgd_comm, uniq_ids, aggregated_uniq_ids);

  for(int stratum=0; stratum < nstratum; ++stratum) {
    delete[] uniq_ids[stratum];
  }
  splatt_free(uniq_ids);

  /* populate each tile */
  p_populate_tiles(ws, train, &sgd_comm, nnzs);
  splatt_free(nnzs);

  ws->external_validate = tt_alloc(validate->nnz, nmodes); 
  ws->external_validate->vals = validate->vals;
  ws->external_validate->ind[0] = validate->ind[0];

  idx_t nrows[nmodes];
  for(idx_t m=0; m < nmodes; ++m) {
    nrows[m] = sgd_rinfo.layer_ptrs[m][rank + 1] - sgd_rinfo.layer_ptrs[m][rank];
  }

  for(idx_t n=0; n < validate->nnz; ++n) {
    /* find tile id */
    int tile_id = -1;
    for(idx_t stratum=0; stratum < nstratum; ++stratum) {
      bool in_tile = true;
      int stratum_temp = stratum;
      for(int m=1; m < nmodes; ++m) {
        int layer = (stratum_temp%npes + rank)%npes;
        stratum_temp /= npes;

        if(validate->ind[m][n] < sgd_rinfo.layer_ptrs[m][layer] ||
          validate->ind[m][n] >= sgd_rinfo.layer_ptrs[m][layer + 1]) {
          in_tile = false;
          break;
        }
      }
      if (in_tile) {
        assert(tile_id == -1);
        tile_id = stratum;
      }
    }
    assert(tile_id != -1);

    int stratum_temp = tile_id;
    for(idx_t m=1; m < nmodes; ++m) {
      int stratum_offset = stratum_temp%npes;
      stratum_temp /= npes;

      if(stratum_offset != 0) {
        assert(sgd_comm.aggregated_global_to_locals[m].find(validate->ind[m][n]) != sgd_comm.aggregated_global_to_locals[m].end());
        ws->external_validate->ind[m][n] = nrows[m] + sgd_comm.aggregated_global_to_locals[m][validate->ind[m][n]];
      }
      else {
        ws->external_validate->ind[m][n] = validate->ind[m][n] - sgd_rinfo.layer_ptrs[m][rank];
      }
    }
  }

  /* set up persistent communication */
  tc_model *model_compacted = (tc_model *)splatt_malloc(sizeof(*model_compacted));
  sgd_comm.model = model_compacted;

  model_compacted->which = model->which;
  model_compacted->rank = nfactors;
  model_compacted->nmodes = train->nmodes;

  for(int m=0; m < train->nmodes; ++m) {
    model_compacted->dims[m] = train->dims[m];

    idx_t bytes = model_compacted->dims[m] * nfactors * sizeof(**(model_compacted->factors));
    if(m > 0) {
      bytes = 0;
      for(int p=0; p < npes; ++p) {
        if(p != rank) {
          bytes += sgd_comm.aggregated_local_to_global_lens[p][m];
        }
      }
      bytes += nrows[m];
      bytes *= nfactors*sizeof(val_t);
    }
    model_compacted->factors[m] = (val_t *)splatt_malloc(bytes);
  }
  p_setup_sgd_persistent_comm(&sgd_comm);

  int stratum_offset[MAX_NMODES];
  stratum_offset[0] = 0;

#endif

  timer_reset(&ws->shuffle_time);
  timer_reset(&ws->train_time);
  timer_reset(&ws->test_time);

  sp_timer_t comm_with_owner_time, bcast_time;
  timer_reset(&comm_with_owner_time);
  timer_reset(&bcast_time);

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  par_memcpy(model_compacted->factors[0], model->factors[0], sizeof(val_t)*nrows[0]*nfactors);
  for(idx_t m=1; m < train->nmodes; ++m) {
    par_memcpy(model_compacted->factors[m], model->factors[m] + sgd_rinfo.layer_ptrs[m][rank]*nfactors, sizeof(val_t)*nrows[m]*nfactors);
  }

  /* initialize permutations */
  idx_t **tile_perms = (idx_t **)splatt_malloc(sizeof(idx_t *) * nstratum);
  for(idx_t i=0; i < nstratum; ++i) {
    tile_perms[i] = (idx_t *)splatt_malloc(sizeof(idx_t) * ws->tiles[i]->nnz);
#pragma omp parallel for
    for(idx_t n=0; n < ws->tiles[i]->nnz; ++n) {
      tile_perms[i][n] = n;
    }
  }

  idx_t *stratum_perm = (idx_t *)splatt_malloc(sizeof(idx_t) * nstratum);
#pragma omp parallel for
  for(idx_t i=0; i < nstratum; ++i) {
    stratum_perm[i] = i;
  }

  /* for bold driver */
  val_t obj = loss + frobsq;
  val_t prev_obj = obj;

  timer_start(&ws->tc_time);
  /* foreach epoch */
  for(idx_t e=1; e < ws->max_its+1; ++e) {


    /* update model from all training observations */
#if USE_CSF_SGD
    if (ws->rand_per_iteration || e == 1) {
      timer_start(&ws->shuffle_time);
      shuffle_idx(perm_i, nslices);
      timer_stop(&ws->shuffle_time);
    }

    timer_start(&ws->train_time);
    if (ws->hogwild) {
#pragma omp parallel for
      for(idx_t i=0; i < nslices; ++i) {
        p_update_model_csf3(csf, perm_i[i], model, ws);
      }
    }
    else {
      for(idx_t i=0; i < nslices; ++i) {
        p_update_model_csf3(csf, perm_i[i], model, ws);
      }
    }
#else
    timer_start(&ws->train_time);

    shuffle_idx(stratum_perm, nstratum);
    /* FIXME: the same permutation across ranks without bcast */
    MPI_Bcast(stratum_perm, nstratum, SPLATT_MPI_IDX, 0, MPI_COMM_WORLD);

    stratum_offset[0] = 0;

    for(int stratum=0; stratum < nstratum; ++stratum) {
      int real_stratum = stratum_perm[stratum];

      int stratum_temp = real_stratum;
      for(int m=1; m < nmodes; ++m) {
        stratum_offset[m] = stratum_temp%npes;
        stratum_temp /= npes;
      }

      timer_start(&comm_with_owner_time);

      MPI_Startall(sgd_comm.request_lens[real_stratum], sgd_comm.requests[0][real_stratum]);

      /* each owner compacts models before send */
      for(int m=1; m < nmodes; ++m) {
        if(stratum_offset[m] == 0) continue;

#pragma omp parallel for
        for(idx_t i=0; i < sgd_comm.remote_to_global_lens[real_stratum][m]; ++i) {
          assert(sgd_comm.remote_to_globals[real_stratum][m][i] >= sgd_rinfo.layer_ptrs[m][rank]);
          assert(sgd_comm.remote_to_globals[real_stratum][m][i] < sgd_rinfo.layer_ptrs[m][rank + 1]);
          for(idx_t f=0; f < nfactors; ++f) {
            sgd_comm.send_recv_buf[m][i*nfactors + f] = model_compacted->factors[m][(sgd_comm.remote_to_globals[real_stratum][m][i] - sgd_rinfo.layer_ptrs[m][rank])*nfactors + f];
          }
        }
      }

      /* recv models from the owners */
      MPI_Startall(sgd_comm.request_lens[real_stratum], sgd_comm.requests[0][real_stratum] + sgd_comm.request_lens[real_stratum]);
      timer_stop(&comm_with_owner_time);

      /* overlap shuffle with send/recv */
      if(ws->rand_per_iteration || e == 1) {
        timer_start(&ws->shuffle_time);
        shuffle_idx(tile_perms[real_stratum], ws->tiles[real_stratum]->nnz);
        timer_stop(&ws->shuffle_time);
      }

      timer_start(&comm_with_owner_time);
      MPI_Waitall(2*sgd_comm.request_lens[real_stratum], sgd_comm.requests[0][real_stratum], MPI_STATUSES_IGNORE);
      timer_stop(&comm_with_owner_time);

      /* update model for this stratum */
      if(ws->hogwild) {
#pragma omp parallel for
        for(idx_t n=0; n < ws->tiles[real_stratum]->nnz; ++n) {
          idx_t real_n = tile_perms[real_stratum][n];
          p_update_model(ws->tiles[real_stratum], real_n, model_compacted, ws);
        }
      }
      else {
        for(idx_t n=0; n < ws->tiles[real_stratum]->nnz; ++n) {
          idx_t real_n = tile_perms[real_stratum][n];
          p_update_model(ws->tiles[real_stratum], real_n, model_compacted, ws);
        }
      }

      timer_start(&comm_with_owner_time);

      /* send models back to the owners */
      MPI_Startall(sgd_comm.request_lens[real_stratum]*2, sgd_comm.requests[1][real_stratum]);
      MPI_Waitall(sgd_comm.request_lens[real_stratum]*2, sgd_comm.requests[1][real_stratum], MPI_STATUSES_IGNORE);

      /* each owner scatters received compact data to their original locations */
      /* TODO: a owner may receive multiple updates when we replicate models for short modes */
      for(int m=1; m < nmodes; ++m) {
        if (stratum_offset[m] == 0) continue;

#pragma omp parallel for
        for(idx_t i=0; i < sgd_comm.remote_to_global_lens[real_stratum][m]; ++i) {
          assert(sgd_comm.remote_to_globals[real_stratum][m][i] >= sgd_rinfo.layer_ptrs[m][rank]);
          assert(sgd_comm.remote_to_globals[real_stratum][m][i] < sgd_rinfo.layer_ptrs[m][rank + 1]);
          for(idx_t f=0; f < nfactors; ++f) {
            model_compacted->factors[m][(sgd_comm.remote_to_globals[real_stratum][m][i] - sgd_rinfo.layer_ptrs[m][rank])*nfactors + f] = sgd_comm.send_recv_buf[m][i*nfactors + f];
          }
        }
      }

      timer_stop(&comm_with_owner_time);
    } /* for each stratum */

    /* collect portions of model required for convergence check from the owners */
    timer_start(&bcast_time);
    MPI_Startall((npes - 1)*(nmodes - 1), sgd_comm.aggregated_requests);
    for(int m=1; m < nmodes; ++m) {
      idx_t offset = 0;
      for(int p=0; p < npes; ++p) {
        if(p == rank) continue;
#pragma omp parallel for
        for(idx_t i=0; i < sgd_comm.aggregated_remote_to_global_lens[p][m]; ++i) {
          for(int f=0; f < nfactors; ++f) {
            sgd_comm.send_recv_buf[m][(i + offset)*nfactors + f] = model_compacted->factors[m][(sgd_comm.aggregated_remote_to_globals[p][m][i] - sgd_rinfo.layer_ptrs[m][rank])*nfactors + f];
          }
          assert(sgd_comm.aggregated_remote_to_globals[p][m][i] >= sgd_rinfo.layer_ptrs[m][rank]);
          assert(sgd_comm.aggregated_remote_to_globals[p][m][i] < sgd_rinfo.layer_ptrs[m][rank + 1]);
        }
        offset += sgd_comm.aggregated_remote_to_global_lens[p][m];
      }
    }

    MPI_Startall((npes - 1)*(nmodes - 1), sgd_comm.aggregated_requests + (npes - 1)*(nmodes - 1));
    MPI_Waitall(2*(npes - 1)*(nmodes - 1), sgd_comm.aggregated_requests, MPI_STATUSES_IGNORE);
    timer_stop(&bcast_time);

    timer_stop(&ws->train_time);
#endif // !USE_CSF_SGD

    /* compute RMSE and adjust learning rate */
    timer_start(&ws->test_time);
    loss = tc_loss_sq(ws->external_tile, model_compacted, ws);
    frobsq = p_frob_sq(&sgd_comm, model_compacted, ws);
    obj = loss + frobsq;
    if(tc_converge(ws->external_tile, ws->external_validate, model_compacted, loss, frobsq, e, ws)) {
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

#ifdef SPLATT_USE_MPI
  if(rank == 0) {
    printf("   train_time %g\n", ws->train_time.seconds);
    printf("     comm_with_owner_time %g\n", comm_with_owner_time.seconds);
    printf("     bcast_time %g\n", bcast_time.seconds);
    printf("   test_time %g\n", ws->test_time.seconds);
    printf("   shuffle_time %g\n", ws->shuffle_time.seconds);
  }

  for(idx_t stratum=0; stratum < nstratum; ++stratum) {
    tt_free(ws->tiles[stratum]);
  }
  splatt_free(ws->tiles);
#endif

#if USE_CSF_SGD
  splatt_free(perm_i);
  csf_free_mode(csf);
  splatt_free(csf);
  splatt_free_opts(opts);
#endif
}


