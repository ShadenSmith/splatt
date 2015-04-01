
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../splatt_mpi.h"


/******************************************************************************
 * PRIVATE DEFINES
 *****************************************************************************/
static int const MSG_FINISHED  = 0;
static int const MSG_TRYCLAIM  = 1;
static int const MSG_MUSTCLAIM = 2;
static int const MSG_SENDBACK  = 3;
static int const MSG_STANDBY   = 4;
static int const MSG_UPDATES   = 5;



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Computes a factor matrix distribution using a naive method. Each
*        layer is defined as N / (p^0.3) slices is distributed in a contiguous
*        fashion to all processes with nonzeros in that layer.
*
* @param rinfo The MPI rank information to fill in.
* @param tt The distributed tensor.
* @param perm The resulting permutation (set to identity).
*/
static void __naive_mat_distribution(
  rank_info * const rinfo,
  sptensor_t const * const tt,
  permutation_t * const perm)
{
  MPI_Comm lcomm;
  int lrank;
  int npes;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    lcomm = rinfo->layer_comm[m];
    MPI_Comm_size(lcomm, &npes);
    MPI_Comm_rank(lcomm, &lrank);

    /* slice start/end of layer */
    idx_t const start = rinfo->layer_starts[m];
    idx_t const end = rinfo->layer_ends[m];

    /* target nrows = layer_size / npes in a layer */
    idx_t const psize = (end - start) / npes;

    rinfo->mat_start[m] = lrank * psize;
    rinfo->mat_end[m]   = (lrank + 1) * psize;
    /* account for being the last process in a layer */
    if(lrank == npes - 1) {
      rinfo->mat_end[m] = end - start;
    }
  }

  /* set perm to identity */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    for(idx_t i=0; i < tt->dims[m]; ++i) {
      perm->perms[m][i] = i;
      perm->iperms[m][i] = i;
    }
  }
}


/**
* @brief Generate a 'job' (an order to select rows for ownership) for a light
*        rank in my lower. This function is the producer for the work queue
*        and called by the root node in the layer.
*
* @param npes The number of ranks in the layer.
* @param lastp The most recent process to be given work.
* @param pvols An array of communication volumes.
* @param rinfo MPI rank information.
* @param comm The layer communicator.
* @param mustclaim A flag marking whether the last job was successful (whether
*                  any rows were claimed).
* @param left How many unclaimed rows are left.
*
* @return The selected rank.
*/
static int __make_job(
  int const npes,
  int const lastp,
  idx_t * const pvols,
  rank_info * const rinfo,
  MPI_Comm const comm,
  int const mustclaim,
  idx_t const left)
{
  /* grab 2 smallest processes */
  int p0 = (lastp+1) % npes;
  int p1 = (lastp+2) % npes;
  for(int p = 0; p < npes; ++p) {
    if(pvols[p] < pvols[p0]) {
      p1 = p0;
      p0 = p;
    }
  }

  rinfo->worksize = SS_MIN(pvols[p1] - pvols[p0], left);
  if(rinfo->worksize == 0) {
    rinfo->worksize = SS_MAX(left / npes, 1);
  }

  if(!mustclaim) {
    MPI_Isend(&MSG_TRYCLAIM, 1, MPI_INT, p0, 0, comm, &(rinfo->req));
  } else {
    MPI_Isend(&MSG_MUSTCLAIM, 1, MPI_INT, p0, 0, comm, &(rinfo->req));
  }

  MPI_Isend(&(rinfo->worksize), 1, SS_MPI_IDX, p0, 0, comm, &(rinfo->req));

  for(int p=0; p < npes; ++p) {
    if(p != p0) {
      MPI_Isend(&MSG_STANDBY, 1, MPI_INT, p, 0, comm, &(rinfo->req));
    }
  }

  return p0;
}


/**
* @brief Receive the latest claimed rows and update all other ranks.
*
* @param npes The number of ranks in my layer.
* @param pvols An array of process communication volumes.
* @param rinfo MPI rank information.
* @param comm The layer communicator.
* @param rowbuf A buffer to receive the claimed rows.
* @param left A pointer to update, we subtract the newly claimed rows from what
*             is left.
*
* @return The number of rows that were claimed.
*/
static idx_t __check_job(
  int const npes,
  idx_t * const pvols,
  rank_info * const rinfo,
  MPI_Comm const comm,
  idx_t * const rowbuf,
  idx_t * const left)
{
  MPI_Probe(MPI_ANY_SOURCE, 0, comm, &(rinfo->status));

  int const proc_up = rinfo->status.MPI_SOURCE;
  idx_t nclaimed;
  MPI_Recv(&nclaimed, 1, SS_MPI_IDX, proc_up, 0, comm, &(rinfo->status));
  MPI_Recv(rowbuf, nclaimed, SS_MPI_IDX, proc_up, 0, comm, &(rinfo->status));

  pvols[proc_up] += nclaimed;

  *left -= nclaimed;
  /* send new status message */
  for(int p=0; p < npes; ++p) {
    if(*left == 0) {
      MPI_Isend(&MSG_FINISHED, 1, MPI_INT, p, 0, comm, &(rinfo->req));
    } else {
      MPI_Isend(&MSG_UPDATES, 1, MPI_INT, p, 0, comm, &(rinfo->req));
    }
  }

  return nclaimed;
}



/**
* @brief Claim up to 'amt' rows that are unclaimed and found in my local
*        tensor.
*
* @param amt The maximum (desired) rows to claim.
* @param inds Indices local to my tensor.
* @param localdim The size of 'inds'
* @param rinfo MPI rank information.
* @param claimed An array marking which rows have been claimed.
* @param layerdim The dimension of the layer.
* @param newclaims An array of newly claimed rows.
*
* @return The number of claimed rows.
*/
static idx_t __tryclaim_rows(
  idx_t const amt,
  idx_t const * const inds,
  idx_t const localdim,
  rank_info const * const rinfo,
  char * const claimed,
  idx_t const layerdim,
  idx_t * const newclaims)
{
  idx_t newrows = 0;

  /* find at most amt unclaimed rows in my partition */
  for(idx_t i=0; i < localdim; ++i) {
    assert(inds[i] < layerdim);
    if(claimed[inds[i]] == 0) {
      newclaims[newrows++] = inds[i];
      claimed[inds[i]] = 1;

      if(newrows == amt) {
        break;
      }
    }
  }

  return newrows;
}


/**
* @brief Claim exactly 'amt' rows, first attempting to grab them from my local
*        tensor.
*
* @param amt The number of rows I must claim.
* @param inds Indices local to my tensor.
* @param localdim The size of 'inds'.
* @param rinfo MPI rank information.
* @param claimed An array marking which rows have been claimed.
* @param layerdim The dimension of the layer.
* @param newclaims An array of newly claimed rows.
*
* @return The number of claimed rows.
*/
static idx_t __mustclaim_rows(
  idx_t const amt,
  idx_t const * const inds,
  idx_t const localdim,
  rank_info const * const rinfo,
  char * const claimed,
  idx_t const layerdim,
  idx_t * const newclaims)
{
  /* first try local rows */
  idx_t newrows = __tryclaim_rows(amt, inds, localdim, rinfo, claimed,
      layerdim, newclaims);

  if(newrows == amt) {
    return newrows;
  }

  /* just grab the first amt unclaimed rows */
  for(idx_t i=0; i < layerdim; ++i) {
    if(claimed[i] == 0) {
      newclaims[newrows++] = i;
      claimed[i] = 1;
      if(newrows == amt) {
        break;
      }
    }
  }
  assert(newrows == amt);

  return newrows;
}


static void __distribute_u3_rows(
  idx_t const m,
  int const * const pcount,
  idx_t * const pvols,
  idx_t const * const rconns,
  idx_t * const mine,
  idx_t * const nrows,
  idx_t const * const inds,
  idx_t const localdim,
  rank_info * const rinfo)
{
  MPI_Comm const comm = rinfo->layer_comm[m];
  MPI_Request req;
  int const rank = rinfo->layer_rank[m];
  int npes;
  MPI_Comm_size(comm, &npes);
  int msg;
  idx_t amt;

  idx_t left = rconns[2];
  idx_t const dim = rinfo->layer_ends[m] - rinfo->layer_starts[m];

  /* mark if row claimed[i] has been claimed */
  char * claimed = (char *) calloc(dim, sizeof(char));

  /* a list of all rows I just claimed */
  idx_t * myclaims = (idx_t *) malloc(left * sizeof(idx_t));

  /* incoming new assignments */
  idx_t * bufclaims = (idx_t *) malloc(left * sizeof(idx_t));

  /* mark the rows already claimed */
  for(idx_t i=0; i < *nrows; ++i) {
    assert(mine[i] < dim);
    claimed[mine[i]] = 1;
  }

  /* Everyone gets a consistent set of claimed rows */
  MPI_Allreduce(MPI_IN_PLACE, claimed, dim, MPI_CHAR, MPI_SUM, comm);
  for(idx_t i=0; i < dim; ++i) {
    assert(claimed[i] <= 1);
  }

  /* lets root know which process was chosen last for grabbing rows */
  int newp = 0;

  int mustclaim = 0;
  idx_t nclaimed = 0;

  while(1) {
    if(rank == 0) {
      newp = __make_job(npes, newp, pvols, rinfo, comm, mustclaim, left);
    }

    MPI_Recv(&msg, 1, MPI_INT, 0, 0, comm, &(rinfo->status));
    if(msg == MSG_TRYCLAIM || msg == MSG_MUSTCLAIM) {
      /* get target number of rows */
      MPI_Recv(&amt, 1, SS_MPI_IDX, 0, 0, comm, &(rinfo->status));
      /* see how many I can claim */
      if(msg == MSG_TRYCLAIM) {
        nclaimed = __tryclaim_rows(amt, inds, localdim, rinfo, claimed, dim,
            myclaims);
      } else {
        nclaimed = __mustclaim_rows(amt, inds, localdim, rinfo, claimed, dim,
            myclaims);
      }

      /* send new claims to root process */
      MPI_Isend(&nclaimed, 1, SS_MPI_IDX, 0, 0, comm, &req);
      MPI_Isend(myclaims, nclaimed, SS_MPI_IDX, 0, 0, comm, &req);
      /* now mark as mine */
      for(idx_t i=0; i < nclaimed; ++i) {
        mine[(*nrows)++] = myclaims[i];
      }
    }

    /* check for updated rows, completion, etc. */
    if(rank == 0) {
      amt = __check_job(npes, pvols, rinfo, comm, bufclaims, &left);
      /* force claim next turn if no progress made this time */
      mustclaim = (amt > 0);
    }

    MPI_Recv(&msg, 1, MPI_INT, 0, 0, comm, &(rinfo->status));
    if(msg == MSG_UPDATES) {
      /* get new rows */
      MPI_Bcast(&amt, 1, SS_MPI_IDX, 0, comm);
      MPI_Bcast(bufclaims, amt, SS_MPI_IDX, 0, comm);

      /* mark as claimed */
      for(idx_t i=0; i < amt; ++i) {
        claimed[bufclaims[i]] = 1;
      }
    } else if(msg == MSG_FINISHED) {
      break;
    }
  }

  free(bufclaims);
  free(myclaims);
  free(claimed);

  MPI_Barrier(comm);
}



/**
* @brief Fill communication volume statistics (connectivity factor rows) and
*        store in rconns.
*
* @param m The mode to operate on.
* @param pcount An array of size 'ldim' which stores the count of how many
*               ranks have a nonzero in this slice.
* @param ldim The size (number of slices) of my layer.
* @param rinfo MPI rank information.
* @param rconns Row connectivity information. rconns[0] stores the number of
*               rows that only appear in 1 rank, rconns[1] stores the number of
*               rows that appear in 2 ranks, and rconns[2] stores the number of
*               rows that appear in >2 ranks.
*/
static void __fill_volume_stats(
  idx_t const m,
  int const * const pcount,
  idx_t const ldim,
  rank_info const * const rinfo,
  idx_t * const rconns)
{
  /* count u=1; u=2, u > 2 */
  rconns[0] = rconns[1] = rconns[2] = 0;
  int tot = 0;
  for(idx_t i=0; i < ldim; ++i) {
    assert(pcount[i] <= (rinfo->np13 * rinfo->np13));
    tot += pcount[i];
    switch(pcount[i]) {
    case 0:
      /* this only happens with empty slices */
      break;
    case 1:
      rconns[0] += 1;
      break;
    case 2:
      //rconns[1] += 1;
    default:
      rconns[2] += 1;
      break;
    }
  }
}


/**
* @brief Computes a factor matrix distribution using a greedy method. Each rank
*        claims all rows foind only in its own partition and contested rows are
*        given in a greedy manner which attempts to minimize total volume.
*
*        NOTE: Since ranks can end up with non-contiguous partitions we reorder
*        the tensor after distribution to have nice contiguous blocks of the
*        factor matrices! tt->indmap will be updated accordingly.
*
* @param rinfo The MPI rank information to fill in.
* @param tt The distributed tensor which MAY be reordered.
*/
static void __greedy_mat_distribution(
  rank_info * const rinfo,
  sptensor_t const * const tt,
  permutation_t * const perm)
{
  /* get the maximum dimension size for my layer */
  idx_t max_dim = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    idx_t const dsize = rinfo->layer_ends[m] - rinfo->layer_starts[m];
    if(dsize > max_dim) {
      max_dim = dsize;
    }
  }

  /* count of appearances for each idx across all ranks */
  idx_t rconns[3];
  int * pcount = (int *) malloc(max_dim * sizeof(int));
  idx_t * mine = (idx_t *) malloc(max_dim * sizeof(idx_t));

  int lnpes; /* npes in layer */
  idx_t * pvols; /* volumes of each rank */

  for(idx_t m=0; m < tt->nmodes; ++m) {
    /* layer dimensions */
    idx_t const layerdim = tt->dims[m];

    /* get local idxs */
    idx_t localdim;
    idx_t * inds = tt_get_slices(tt, m, &localdim);

    memset(pcount, 0, layerdim * sizeof(int));
    memset(mine, 0, layerdim * sizeof(idx_t));

    /* mark all idxs that are local to me */
    for(idx_t i=0; i < localdim; ++i) {
      pcount[inds[i]] = 1;
    }

    /* sum appearances to get communication volume */
    MPI_Allreduce(MPI_IN_PLACE, pcount, layerdim, MPI_INT, MPI_SUM,
        rinfo->layer_comm[m]);

    /* communication volume */
    idx_t myvol = 0;

    /* number of rows I own */
    idx_t nrows = 0;

    /* claim all rows that are entirely local to me */
    for(idx_t i=0; i < localdim; ++i) {
      switch(pcount[inds[i]]) {
      case 0:
        break;
      case 1:
        mine[nrows++] = inds[i];
        break;
      default:
        ++myvol;
        break;
      }
    }

    /* get size of layer and allocate volumes */
    MPI_Comm_size(rinfo->layer_comm[m], &lnpes);
    pvols = (idx_t *) malloc(lnpes * sizeof(idx_t));

    /* root process gathers all communication volumes */
    MPI_Gather(&myvol, 1, SS_MPI_IDX, pvols, 1, SS_MPI_IDX,
      0, rinfo->layer_comm[m]);

    /* now distribute rows with >=3 pcount in a greedy fashion */
    __fill_volume_stats(m, pcount, layerdim, rinfo, rconns);
    __distribute_u3_rows(m, pcount, pvols, rconns, mine, &nrows,
        inds, localdim, rinfo);

    /* prefix sum to get our new mat_start */
    idx_t rowoffset;
    MPI_Scan(&nrows, &rowoffset, 1, SS_MPI_IDX, MPI_SUM, rinfo->layer_comm[m]);

    /* ensure all rows are accounted for */
    if(rinfo->layer_rank[m] == (rinfo->np13 * rinfo->np13) - 1) {
      assert(rowoffset == rinfo->layer_ends[m] - rinfo->layer_starts[m]);
    }
    rowoffset -= nrows;

    /* assign new labels - IPERM is easier to fill first.
     * newlabels[newindex] = oldindex */
    idx_t * const newlabels = perm->iperms[m];
    idx_t * const inewlabels = perm->perms[m];
    memset(newlabels, 0, layerdim * sizeof(idx_t));
    for(idx_t i=0; i < nrows; ++i) {
      assert(rowoffset+i < layerdim);
      assert(mine[i] < layerdim);
      newlabels[rowoffset+i] = mine[i];
    }

    MPI_Allreduce(MPI_IN_PLACE, newlabels, layerdim, SS_MPI_IDX, MPI_SUM,
        rinfo->layer_comm[m]);

    /* fill perm: inewlabels[oldlayerindex] = newlayerindex */
    for(idx_t i=0; i < layerdim; ++i) {
      assert(newlabels[i] < layerdim);
      inewlabels[newlabels[i]] = i;
    }

    /* store matrix info */
    rinfo->mat_start[m] = rowoffset;
    rinfo->mat_end[m] = SS_MIN(rinfo->mat_start[m] + nrows, layerdim);

    free(inds);
    free(pvols);
    MPI_Barrier(rinfo->layer_comm[m]);
  } /* foreach mode */

  free(pcount);
  free(mine);
}


/**
* @brief Allocate + fill mat_ptrs, an array marking the start index for each
*        rank. Indices are local to the layer.
*
* @param rinfo The structure containing MPI information.
*/
static void __setup_mat_ptrs(
  idx_t const mode,
  MPI_Comm const comm,
  rank_info * const rinfo)
{
  /* number of procs in layer */
  int npes;
  int rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  /* allocate space for start/end idxs */
  rinfo->mat_ptrs[mode] = (idx_t *) calloc(npes + 1, sizeof(idx_t));
  idx_t * const mat_ptrs = rinfo->mat_ptrs[mode];

  mat_ptrs[rank] = rinfo->mat_start[mode];
  mat_ptrs[npes] = rinfo->layer_ends[mode] - rinfo->layer_starts[mode];

  /* Doing a reduce instead of a gather lets us set location mode_rank
   * instead of the rank in this communicator */
  MPI_Allreduce(MPI_IN_PLACE, mat_ptrs, npes, SS_MPI_IDX, MPI_SUM, comm);

  assert(rinfo->mat_ptrs[mode][rank    ] == rinfo->mat_start[mode]);
  assert(rinfo->mat_ptrs[mode][rank + 1] == rinfo->mat_end[mode]);
}



/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

permutation_t * mpi_distribute_mats(
  rank_info * const rinfo,
  sptensor_t * const tt,
  idx_t const distribution)
{
  permutation_t * perm = perm_identity(tt->dims, tt->nmodes);
  switch(distribution) {
  case 1:
    /* assign simple 1D matrix distribution */
    for(idx_t m=0; m < tt->nmodes; ++m) {
      rinfo->mat_start[m] = 0;
      rinfo->mat_end[m] = rinfo->layer_ends[m] - rinfo->layer_starts[m];
    }
    break;

  case 2:
    break;

  case 3:
    __greedy_mat_distribution(rinfo, tt, perm);
    perm_apply(tt, perm->perms);

    for(idx_t m=0; m < tt->nmodes; ++m) {
      __setup_mat_ptrs(m, rinfo->layer_comm[m], rinfo);
    }
    break;
  }

  return perm;
}


void mpi_find_owned(
  sptensor_t const * const tt,
  rank_info * const rinfo)
{
  for(idx_t m=0; m < tt->nmodes; ++m) {
    idx_t const start = rinfo->mat_start[m];
    idx_t const end = rinfo->mat_end[m];
    idx_t const * const indmap = tt->indmap[m];

    rinfo->ownstart[m]= tt->dims[m];
    rinfo->ownend[m] = 0;
    rinfo->nowned[m] = 0;
    for(idx_t i=0; i < tt->dims[m]; ++i) {
      idx_t gi = (indmap == NULL) ? i : indmap[i];
      if(gi >= start && gi < end) {
        rinfo->nowned[m] += 1;
        rinfo->ownstart[m] = SS_MIN(rinfo->ownstart[m], i);
        rinfo->ownend[m] = SS_MAX(rinfo->ownend[m], i);
      }
    }
    rinfo->ownend[m] += 1;

    /* sanity check to ensure owned rows are contiguous */
    if(indmap != NULL) {
      for(idx_t i=rinfo->ownstart[m]+1; i < rinfo->ownend[m]; ++i) {
        assert(indmap[i] >= start && indmap[i] < end);
        assert(indmap[i] == indmap[i-1]+1);
      }
    }
  }
}


