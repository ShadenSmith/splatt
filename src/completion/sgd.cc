

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "completion.h"
#include "../csf.h"
#include "../reorder.h"
#include "../util.h"
#include "../io.h"
#include "../sort.h"

#include <math.h>
#include <omp.h>
#include <algorithm>
#include <set>
#include <unordered_map>

#define USE_CSF_SGD 0

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

}

typedef struct
{
  int nstratum;
  tc_model *model;
  rank_info *rinfo;

  /* mapping from local compacted indices within each tile to global indices.
   * nstratum*nmodes*l where l is specified by local_to_global_lens[s][m]
   */
  idx_t ***local_to_globals;
  idx_t **local_to_global_lens;

  /* mapping from compacted indices within each remote tile to global indices.
   * nstratum*nmodes*l where l is specified by remote_to_global_lens[s][m]
   */
  idx_t ***remote_to_globals;
  idx_t **remote_to_global_lens;

  val_t **send_recv_buf; /* nmodes arrays */

  unordered_map<idx_t, idx_t> **global_to_locals;
    /**< inverse mapping of local_to_globals */

  MPI_Request **requests[2];
    /* 2*nstratum*(nmodes*2) array */
    /* requests[0] is for receiving models from owners */
    /* requests[1] is for sending models back to owners */
  int *request_lens;
} sgd_comm_t;

static int p_get_nstratum(int nmodes, int npes)
{
  int nstratum = 1;
  for(int m=1; m < nmodes; ++m) {
    nstratum *= npes;
  }
  return nstratum;
}

static std::set<idx_t> **p_count_nnz_of_tiles(
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
  std::set<idx_t> **uniq_ids = (std::set<idx_t> **)splatt_malloc(sizeof(std::set<idx_t> *)*nstratum);
  for(int stratum=0; stratum < nstratum; ++stratum) {
    uniq_ids[stratum] = new std::set<idx_t>[nmodes];
  }

  for(idx_t n=0; n < train->nnz; ++n) {
    int tile_id = -1;
    for(idx_t stratum=0; stratum < nstratum; ++stratum) {
      bool in_tile = true;
      int stratum_temp = stratum;
      for(int m=1; m < nmodes; ++m) {
        int layer = (stratum_temp%npes + rank)%npes;
        stratum_temp /= npes;

        if(train->ind[m][n] < rinfo->layer_ptrs[m][layer] ||
          train->ind[m][n] >= rinfo->layer_ptrs[m][layer + 1]) {
          in_tile = false;
          break;
        }
      }
      if (in_tile) {
        assert(tile_id == -1);
        tile_id = stratum;

        stratum_temp = stratum;
        for(int m=1; m < nmodes; ++m) {
          if (stratum_temp%npes != 0) {
            uniq_ids[stratum][m].insert(train->ind[m][n]);
          }
          stratum_temp /= npes;
        }
      }
    }
    assert(tile_id != -1);

    ++(*nnzs)[tile_id];
  }

  return uniq_ids;
}

static void p_populate_tiles(tc_ws *ws, sptensor_t *train, sgd_comm_t *sgd_comm, idx_t *nnzs)
{
  rank_info *rinfo = sgd_comm->rinfo;
  int npes = rinfo->npes;
  int rank = rinfo->rank;
  int nstratum = sgd_comm->nstratum;
  int nmodes = train->nmodes;

  ws->tiles = (sptensor_t **)splatt_malloc(sizeof(sptensor_t) * nstratum);  
  for(idx_t stratum=0; stratum < nstratum; ++stratum) {
    ws->tiles[stratum] = tt_alloc(nnzs[stratum], nmodes);
  }

  for(idx_t stratum=0; stratum < nstratum; ++stratum) {
    nnzs[stratum] = 0;
  }

  for(idx_t n=0; n < train->nnz; ++n) {
    int tile_id = -1;
    for(idx_t stratum=0; stratum < nstratum; ++stratum) {
      bool in_tile = true;
      int stratum_temp = stratum;
      for(int m=1; m < nmodes; ++m) {
        int layer = (stratum_temp%npes + rank)%npes;
        stratum_temp /= npes;

        if(train->ind[m][n] < rinfo->layer_ptrs[m][layer] ||
          train->ind[m][n] >= rinfo->layer_ptrs[m][layer + 1]) {
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

    ws->tiles[tile_id]->vals[nnzs[tile_id]] = train->vals[n];
    ws->tiles[tile_id]->ind[0][nnzs[tile_id]] = train->ind[0][n];
    int stratum_temp = tile_id;
    for(idx_t m=1; m < nmodes; ++m) {
      int stratum_offset = stratum_temp%npes;
      stratum_temp /= npes;

      if(stratum_offset != 0) {
        int owner = (stratum_offset + rank)%npes;

        assert(sgd_comm->global_to_locals[tile_id][m].find(train->ind[m][n]) != sgd_comm->global_to_locals[tile_id][m].end());
        ws->tiles[tile_id]->ind[m][nnzs[tile_id]] = rinfo->layer_ptrs[m][owner] + sgd_comm->global_to_locals[tile_id][m][train->ind[m][n]];
      }
      else {
        ws->tiles[tile_id]->ind[m][nnzs[tile_id]] = train->ind[m][n];
      }
    }
    ++nnzs[tile_id];
  }
}

static void p_map_global_to_local(
  sgd_comm_t *sgd_comm,
  std::set<idx_t> **uniq_ids)
{
  int nmodes = sgd_comm->model->nmodes;
  int npes = sgd_comm->rinfo->npes;
  int rank = sgd_comm->rinfo->rank;
  int nstratum = p_get_nstratum(nmodes, npes);

  sgd_comm->local_to_globals = (idx_t ***)splatt_malloc(sizeof(idx_t **)*nstratum);
  sgd_comm->local_to_global_lens = (idx_t **)splatt_malloc(sizeof(idx_t *)*nstratum);

  sgd_comm->remote_to_globals = (idx_t ***)splatt_malloc(sizeof(idx_t **)*nstratum);
  sgd_comm->remote_to_global_lens = (idx_t **)splatt_malloc(sizeof(idx_t *)*nstratum);

  sgd_comm->global_to_locals = (unordered_map<idx_t, idx_t> **)splatt_malloc(sizeof(unordered_map<idx_t, idx_t> *)*nstratum);

  for(idx_t stratum=0; stratum < nstratum; ++stratum) {
    sgd_comm->local_to_globals[stratum] = (idx_t **)splatt_malloc(sizeof(idx_t *)*nmodes);
    sgd_comm->local_to_global_lens[stratum] = (idx_t *)splatt_malloc(sizeof(idx_t)*nmodes);

    sgd_comm->remote_to_globals[stratum] = (idx_t **)splatt_malloc(sizeof(idx_t *)*nmodes);
    sgd_comm->remote_to_global_lens[stratum] = (idx_t *)splatt_malloc(sizeof(idx_t)*nmodes);

    sgd_comm->global_to_locals[stratum] = new unordered_map<idx_t, idx_t>[nmodes];

    for(int m=1; m < nmodes; ++m) {
      sgd_comm->local_to_globals[stratum][m] = (idx_t *)splatt_malloc(sizeof(idx_t)*uniq_ids[stratum][m].size());
      sgd_comm->local_to_global_lens[stratum][m] = uniq_ids[stratum][m].size();
    }
  }
  for(idx_t stratum=0; stratum < nstratum; ++stratum) {
    for(int m=1; m < nmodes; ++m) {
      idx_t i = 0;
      for(std::set<idx_t>::iterator itr = uniq_ids[stratum][m].begin(); itr != uniq_ids[stratum][m].end(); ++itr, ++i) {
        sgd_comm->local_to_globals[stratum][m][i] = *itr;
        sgd_comm->global_to_locals[stratum][m][*itr] = i;
      }
    }
  }

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

      MPI_Request send_request, recv_request;

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
        &sgd_comm->local_to_globals[stratum][m][0],
        send_len,
        SPLATT_MPI_IDX,
        owner,
        m,
        MPI_COMM_WORLD,
        &send_request);

      sgd_comm->remote_to_globals[stratum][m] = (idx_t *)splatt_malloc(sizeof(idx_t)*recv_len);
      MPI_Irecv(
        &sgd_comm->remote_to_globals[stratum][m][0],
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
}

static void p_setup_sgd_persistent_comm(sgd_comm_t *sgd_comm)
{
  tc_model *model = sgd_comm->model;
  int nmodes = model->nmodes;
  int nfactors = model->rank;

  int nstratum = sgd_comm->nstratum;
  int npes = sgd_comm->rinfo->npes;
  int rank = sgd_comm->rinfo->rank;

  sgd_comm->send_recv_buf = (val_t **)splatt_malloc(sizeof(val_t *) * nmodes);
  for(int m=1; m < nmodes; ++m) {
    idx_t maximum = 0;
    for(int stratum=0; stratum < nstratum; ++stratum) {
      maximum = SS_MAX(maximum, sgd_comm->local_to_global_lens[stratum][m]);
      maximum = SS_MAX(maximum, sgd_comm->remote_to_global_lens[stratum][m]);
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

  for(int stratum=0; stratum < nstratum; ++stratum) {
    int request_cnt1 = 0, request_cnt2 = 0;

    int stratum_temp = stratum;
    for(int m=1; m < nmodes; ++m) {
      int stratum_offset = stratum_temp%npes;
      stratum_temp /= npes;
      if(stratum_offset == 0) continue;

      int owner = (stratum_offset + rank)%npes;
      MPI_Recv_init(
        model->factors[m] + sgd_comm->rinfo->layer_ptrs[m][owner]*nfactors,
        sgd_comm->local_to_global_lens[stratum][m]*nfactors,
        SPLATT_MPI_VAL,
        owner,
        m,
        MPI_COMM_WORLD,
        sgd_comm->requests[0][stratum] + request_cnt1);
      ++request_cnt1;

      int leaser = (npes - stratum_offset + rank)%npes;
      MPI_Send_init(
        sgd_comm->send_recv_buf[m],
        sgd_comm->remote_to_global_lens[stratum][m]*nfactors,
        SPLATT_MPI_VAL,
        leaser,
        m,
        MPI_COMM_WORLD,
        sgd_comm->requests[0][stratum] + request_cnt1);
      ++request_cnt1;

      MPI_Send_init(
        model->factors[m] + sgd_comm->rinfo->layer_ptrs[m][owner]*nfactors,
        sgd_comm->local_to_global_lens[stratum][m]*nfactors,
        SPLATT_MPI_VAL,
        owner,
        m,
        MPI_COMM_WORLD,
        sgd_comm->requests[1][stratum] + request_cnt2);
      ++request_cnt2;

      MPI_Recv_init(
        sgd_comm->send_recv_buf[m],
        sgd_comm->remote_to_global_lens[stratum][m]*nfactors,
        SPLATT_MPI_VAL,
        leaser,
        m,
        MPI_COMM_WORLD,
        sgd_comm->requests[1][stratum] + request_cnt2);
      ++request_cnt2;
    }
    assert(request_cnt1 == request_cnt2);
    sgd_comm->request_lens[stratum] = request_cnt1;
  }
}

void p_free_sgd_comm(sgd_comm_t *sgd_comm)
{
  for(int stratum=0; stratum < sgd_comm->nstratum; ++stratum) {
    for(int m=1; m < sgd_comm->model->nmodes; ++m) {
      splatt_free(sgd_comm->local_to_globals[stratum][m]);
    }
    splatt_free(sgd_comm->local_to_globals[stratum]);
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

  splatt_free(sgd_comm->local_to_globals);
  splatt_free(sgd_comm->local_to_global_lens);
  splatt_free(sgd_comm->remote_to_globals);
  splatt_free(sgd_comm->remote_to_global_lens);
  splatt_free(sgd_comm->global_to_locals);

  splatt_free(sgd_comm->requests[0]);
  splatt_free(sgd_comm->requests[1]);
  splatt_free(sgd_comm->request_lens);
  splatt_free(sgd_comm->send_recv_buf);
}
#endif /* SPLATT_USE_MPI */

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

  sgd_comm_t sgd_comm;
  sgd_comm.nstratum = nstratum;
  sgd_comm.rinfo = &sgd_rinfo;
  sgd_comm.model = model;

  /* count nnz of each tile */
  idx_t *nnzs;
  std::set<idx_t> **uniq_ids = p_count_nnz_of_tiles(&nnzs, &sgd_comm, train);

  /* mapping from global to local compacted indices within each tile */
  p_map_global_to_local(&sgd_comm, uniq_ids);

  for(int stratum=0; stratum < nstratum; ++stratum) {
    delete[] uniq_ids[stratum];
  }
  splatt_free(uniq_ids);

  /* populate each tile */
  p_populate_tiles(ws, train, &sgd_comm, nnzs);
  splatt_free(nnzs);

  /* set up persistent communication */
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
    if (ws->rand_per_iteration || e == 1) {
      timer_start(&ws->shuffle_time);
#pragma omp parallel for
      for(idx_t i = 0; i < nstratum; ++i) {
        shuffle_idx(tile_perms[i], ws->tiles[i]->nnz);
      }
      timer_stop(&ws->shuffle_time);
    }

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

      /* each owner compacts models before send */
      for(int m=1; m < nmodes; ++m) {
#define SPLATT_DBG_SGD
#ifdef SPLATT_DBG_SGD
        /* for error checking, initialize the part of factor matrices not owned by myself as NaN */
        for(idx_t i=0; i < sgd_rinfo.layer_ptrs[m][rank]*nfactors; ++i) {
          model->factors[m][i] = NAN;
        }
        for(idx_t i=sgd_rinfo.layer_ptrs[m][rank + 1]*nfactors; i < rinfo->global_dims[m]*nfactors; ++i) {
          model->factors[m][i] = NAN;
        }
#endif

        if(stratum_offset[m] == 0) continue;

#pragma omp parallel for
        for(idx_t i=0; i < sgd_comm.remote_to_global_lens[real_stratum][m]; ++i) {
          for(idx_t f=0; f < nfactors; ++f) {
            sgd_comm.send_recv_buf[m][i*nfactors + f] = model->factors[m][sgd_comm.remote_to_globals[real_stratum][m][i]*nfactors + f];
#ifdef SPLATT_DBG_SGD
            assert(!isnan(sgd_comm.send_recv_buf[m][i*nfactors + f]));
#endif
          }
        }
      }

      /* recv models from the owners */
      MPI_Startall(sgd_comm.request_lens[real_stratum], sgd_comm.requests[0][real_stratum]);
      MPI_Waitall(sgd_comm.request_lens[real_stratum], sgd_comm.requests[0][real_stratum], MPI_STATUSES_IGNORE);

#ifdef SPLATT_DBG_SGD
      /* check if we received NaN */
      for(int m=1; m < nmodes; ++m) {
        if(stratum_offset[m] == 0) continue;

        int owner = (stratum_offset[m] + rank)%npes;
        //printf("[%d]", rank);
        for(idx_t i=0; i < sgd_comm.local_to_global_lens[real_stratum][m]; ++i) {
          //if(m == 2) {
            //printf(" %ld:%g", sgd_rinfo.layer_ptrs[m][owner] + i, model->factors[m][(sgd_rinfo.layer_ptrs[m][owner] + i)*nfactors]);
          //}
          for(idx_t f=0; f < nfactors; ++f) {
            assert(!isnan(model->factors[m][(sgd_rinfo.layer_ptrs[m][owner] + i)*nfactors + f]));
          }
        }
        //printf("\n");
      }
#endif

      timer_stop(&comm_with_owner_time);

      /* update model for this stratum */
      if(ws->hogwild) {
#pragma omp parallel for
        for(idx_t n=0; n < ws->tiles[real_stratum]->nnz; ++n) {
          idx_t real_n = tile_perms[real_stratum][n];
          p_update_model(ws->tiles[real_stratum], real_n, model, ws);
        }
      }
      else {
        for(idx_t n=0; n < ws->tiles[real_stratum]->nnz; ++n) {
          idx_t real_n = tile_perms[real_stratum][n];
          p_update_model(ws->tiles[real_stratum], real_n, model, ws);
        }
      }

      timer_start(&comm_with_owner_time);

      /* send models back to the owners */
      MPI_Startall(sgd_comm.request_lens[real_stratum], sgd_comm.requests[1][real_stratum]);
      MPI_Waitall(sgd_comm.request_lens[real_stratum], sgd_comm.requests[1][real_stratum], MPI_STATUSES_IGNORE);

      /* each owner scatters received compact data to their original locations */
      /* TODO: a owner may receive multiple updates when we replicate models for short modes */
      for(int m=1; m < nmodes; ++m) {
        if (stratum_offset[m] == 0) continue;

#pragma omp parallel for
        for(idx_t i=0; i < sgd_comm.remote_to_global_lens[real_stratum][m]; ++i) {
          for(idx_t f=0; f < nfactors; ++f) {
            model->factors[m][sgd_comm.remote_to_globals[real_stratum][m][i]*nfactors + f] = sgd_comm.send_recv_buf[m][i*nfactors + f];
#ifdef SPLATT_DBG_SGD
            assert(!isnan(model->factors[m][sgd_comm.remote_to_globals[real_stratum][m][i]*nfactors + f]));
#endif
          }
        }
      }

      timer_stop(&comm_with_owner_time);
    } /* for each stratum */

    /* broadcast models from the owners */
    /* TODO: this step is required only for tc_converge 
     * (don't need this for model update).
     * Once we can check convergence with distributed model,
     * this step can go away.
     */
    timer_start(&bcast_time);
    for(idx_t p=0; p < npes; ++p) {
      for(idx_t m=1; m < nmodes; ++m) {

        assert(sgd_rinfo.layer_ptrs[m][p + 1] >= sgd_rinfo.layer_ptrs[m][p]);

        MPI_Bcast(
          model->factors[m] + sgd_rinfo.layer_ptrs[m][p]*nfactors,
          (sgd_rinfo.layer_ptrs[m][p + 1] - sgd_rinfo.layer_ptrs[m][p])*nfactors,
          SPLATT_MPI_VAL,
          p,
          MPI_COMM_WORLD);
      }
    }
    timer_stop(&bcast_time);

    timer_stop(&ws->train_time);
#endif

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


