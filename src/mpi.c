
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "mpi.h"
#include "io.h"
#include "sort.h"
#include <string.h>
#include <math.h>


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
 * PRIVATE FUNCTONS
 *****************************************************************************/

/**
* @brief Write a tensor to file <rank>.part. All local indices are converted to
*        global.
*
* @param tt The tensor to write.
* @param perm Any permutations that have been done on the tensor
*             (before compression).
* @param rinfo MPI rank information.
*/
static void __write_part(
  sptensor_t const * const tt,
  permutation_t const * const perm,
  rank_info const * const rinfo)
{
  /* file name is <rank>.part */
  char name[256];
  sprintf(name, "%d.part", rinfo->rank);

  FILE * fout = open_f(name, "w");
  for(idx_t n=0; n < tt->nnz; ++n) {
    for(idx_t m=0; m < tt->nmodes; ++m) {
      /* map idx to original global coordinate */
      idx_t idx = tt->ind[m][n];
      if(tt->indmap[m] != NULL) {
        idx = tt->indmap[m][idx];
      }
      if(perm->iperms[m] != NULL) {
        idx = perm->iperms[m][idx];
      }
      idx += rinfo->layer_starts[m];

      /* write index */
      fprintf(fout, "%"SS_IDX" ", 1+idx);
    }
    fprintf(fout, "%"SS_VAL"\n", tt->vals[n]);
  }
  fclose(fout);
}


/**
* @brief Find the start/end slices in each mode for my partition of X.
*
* @param ssizes The nnz in each slice.
* @param nmodes The number of modes of X.
* @param nnz The number of nonzeros in total.
* @param rinfo MPI information.
*/
static void __find_my_slices(
  idx_t ** const ssizes,
  idx_t const nmodes,
  idx_t const nnz,
  rank_info * const rinfo)
{
  idx_t const pnnz = nnz / rinfo->np13; /* nnz in a layer */
  idx_t const * const dims = rinfo->global_dims;

  /* find start/end slices for my partition */
  for(idx_t m=0; m < nmodes; ++m) {
    /* current processor */
    int currp  = 0;
    idx_t lastn = 0;
    idx_t nnzcnt = 0;

    rinfo->layer_starts[m] = 0;
    rinfo->layer_ends[m] = dims[m];

    for(idx_t s=0; s < dims[m]; ++s) {
      if(nnzcnt >= lastn + pnnz) {
        lastn = nnzcnt;
        ++currp;
        if(currp == rinfo->coords_3d[m]) {
          rinfo->layer_starts[m] = s;
        } else if(currp == rinfo->coords_3d[m]+1 && currp != rinfo->np13) {
          /* only set layer_end if we aren't at the end of the tensor */
          rinfo->layer_ends[m] = s;
          break;
        }
      }
      nnzcnt += ssizes[m][s];
    }
  }
}


/**
* @brief Count the nonzero values in a partition of X.
*
* @param fname The name of the file containing X.
* @param nmodes The number of modes of X.
*
* @return The number of nonzeros in the intersection of all sstarts and sends.
*/
static idx_t __count_my_nnz(
  char const * const fname,
  idx_t const nmodes,
  idx_t const * const sstarts,
  idx_t const * const sends)
{
  FILE * fin = open_f(fname, "r");

  char * ptr = NULL;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  /* count nnz in my partition */
  idx_t mynnz = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      int mine = 1;
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10) - 1;
        if(ind < sstarts[m] || ind >= sends[m]) {
          mine = 0;
        }
      }
      if(mine) {
        ++mynnz;
      }
      /* skip over tensor val */
      strtod(ptr, &ptr);
    }
  }
  fclose(fin);

  free(line);

  return mynnz;
}


/**
* @brief Read a partition of X into tt.
*
* @param fname The file containing X.
* @param tt The tensor structure (must be pre-allocated).
* @param sstarts Array of starting slices, inclusive (one for each mode).
* @param sends Array of ending slices, exclusive (one for each mode).
*/
static void __read_tt_part(
  char const * const fname,
  sptensor_t * const tt,
  idx_t const * const sstarts,
  idx_t const * const sends)
{
  idx_t const nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;

  char * ptr = NULL;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  FILE * fin = open_f(fname, "r");
  idx_t nnzread = 0;
  while(nnzread < nnz && (read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      int mine = 1;
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10) - 1;
        tt->ind[m][nnzread] = ind;
        if(ind < sstarts[m] || ind >= sends[m]) {
          mine = 0;
        }
      }
      val_t const v = strtod(ptr, &ptr);
      tt->vals[nnzread] = v;
      if(mine) {
        ++nnzread;
      }
    }
  }
  fclose(fin);
  free(line);
}


/**
* @brief Read my portion of X from a file.
*
* @param fname The file containing X.
* @param ssizes The nonzero counts in each slice.
* @param nmodes The number of modes in X.
* @param rinfo MPI information (nnz, 3D comm, etc.).
*
* @return My portion of the sparse tensor read from fname.
*/
static sptensor_t * __read_tt(
  char const * const fname,
  idx_t ** const ssizes,
  idx_t const nmodes,
  rank_info * const rinfo)
{
  int const rank = rinfo->rank_3d;
  int const size = rinfo->npes;
  int const p13 = rinfo->np13;
  idx_t const nnz = rinfo->global_nnz;
  idx_t const * const dims = rinfo->global_dims;

  /* find start/end slices for my partition */
  __find_my_slices(ssizes, nmodes, nnz, rinfo);

  /* count nnz in my partition and allocate */
  idx_t const mynnz = __count_my_nnz(fname, nmodes, rinfo->layer_starts, rinfo->layer_ends);
  sptensor_t * tt = tt_alloc(mynnz, nmodes);

  /* compute partition balance */
  idx_t maxnnz;
  MPI_Reduce(&mynnz, &maxnnz, 1, SS_MPI_IDX, MPI_MAX, 0, rinfo->comm_3d);
  if(rank == 0) {
    idx_t target = nnz/size;
    double diff = 100. * ((double)(maxnnz - target)/(double)target);
    printf("nnz: %lu\ttargetnnz: %lu\tmaxnnz: %lu\t(%0.02f%% diff)\n",
        nnz, target, maxnnz, diff);
  }

  /* now actually load values */
  __read_tt_part(fname, tt, rinfo->layer_starts, rinfo->layer_ends);

  return tt;
}



/**
* @brief Count the nonzeros in each slice of X.
*
* @param fname The filename containing nonzeros.
* @param ssizes A 2D array for counting slice 'sizes'.
* @param nmodes The number of modes in X.
* @param rinfo MPI information (containing global dims, nnz, etc.).
*/
static void __fill_ssizes(
  char const * const fname,
  idx_t ** const ssizes,
  idx_t const nmodes,
  rank_info const * const rinfo)
{
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  idx_t const nnz = rinfo->global_nnz;
  idx_t const * const dims = rinfo->global_dims;

  /* compute start/end nnz for counting */
  idx_t const start = rank * nnz / size;
  idx_t end = (rank + 1) * nnz / size;
  if(end > nnz) {
    end = nnz;
  }

  char * ptr = NULL;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  /* skip to start */
  idx_t nlines = 0;
  FILE * fin = open_f(fname, "r");
  while(nlines < start && (read = getline(&line, &len, fin)) != -1) {
    if(read > 1 && line[0] != '#') {
      ++nlines;
    }
  }

  /* start filling ssizes */
  while(nlines < end && (read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ++nlines;
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10) - 1;
        ssizes[m][ind] += 1;
      }
      /* skip over tensor val */
      strtod(ptr, &ptr);
    }
  }
  fclose(fin);

  free(line);

  /* reduce to get total slice counts */
  for(idx_t m=0; m < nmodes; ++m) {
    MPI_Allreduce(MPI_IN_PLACE, ssizes[m], (int) dims[m], SS_MPI_IDX, MPI_SUM,
        MPI_COMM_WORLD);

    idx_t count = 0;
    for(idx_t i=0; i < dims[m]; ++i) {
      count += ssizes[m][i];
    }
    assert(count == nnz);
  }
}



/**
* @brief Get dimensions and nonzero count from tensor file.
*
* @param fname Filename of tensor.
* @param outnnz Number of nonzeros found in X.
* @param outnmodes Number of modes found in X.
* @param outdims The dimensions found in X.
*/
static void __get_dims(
  char const * const fname,
  idx_t * const outnnz,
  idx_t * const outnmodes,
  idx_t * const outdims)
{
  FILE * fin = open_f(fname, "r");

  char * ptr = NULL;
  idx_t nnz = 0;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  /* first count modes in tensor */
  idx_t nmodes = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    if(read > 1 && line[0] != '#') {
      /* get nmodes from first nnz line */
      ptr = strtok(line, " \t");
      while(ptr != NULL) {
        ++nmodes;
        ptr = strtok(NULL, " \t");
      }
      break;
    }
  }
  --nmodes;

  for(idx_t m=0; m < nmodes; ++m) {
    outdims[m] = 0;
  }

  /* fill in tensor dimensions */
  rewind(fin);
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10);
        outdims[m] = (ind > outdims[m]) ? ind : outdims[m];
      }
      /* skip over tensor val */
      strtod(ptr, &ptr);
      ++nnz;
    }
  }
  *outnnz = nnz;
  *outnmodes = nmodes;

  free(line);

  fclose(fin);
}


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



static idx_t __tryclaim_rows(
  idx_t const amt,
  idx_t const * const inds,
  idx_t const localdim,
  rank_info const * const rinfo,
  idx_t const mode,
  char const * const claimed,
  idx_t const layerdim,
  idx_t * const newclaims)
{
  idx_t newrows = 0;

  /* find at most amt unclaimed rows in my partition */
  for(idx_t i=0; i < localdim; ++i) {
    assert(inds[i] < layerdim);
    if(claimed[inds[i]] == 0) {
      newclaims[newrows++] = inds[i];
      if(newrows == amt) {
        break;
      }
    }
  }

  return newrows;
}


static idx_t __mustclaim_rows(
  idx_t const amt,
  rank_info const * const rinfo,
  idx_t const mode,
  char const * const claimed,
  idx_t const layerdim,
  idx_t * const newclaims)
{
  idx_t newrows = 0;

  /* just grab the first amt unclaimed rows */
  for(idx_t i=0; i < layerdim; ++i) {
    if(claimed[i] == 0) {
      newclaims[newrows++] = i;
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
        nclaimed = __tryclaim_rows(amt, inds, localdim, rinfo, m, claimed, dim,
            myclaims);
      } else {
        nclaimed = __mustclaim_rows(amt, rinfo, m, claimed, dim, myclaims);
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
*               rows that appear in 2 ranks, and rconns[3] stores the number of
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
#if 0
  if(rinfo->layer_rank[m] == 0) {
    double pavg = (double) tot / (double) ldim;
    double pct1 = 100. * (double) rconns[0] / ldim;
    double pct2 = 100. * (double) rconns[1] / ldim;
    double pct3 = 100. * (double) rconns[2] / ldim;
    printf("layer: %d uavg: %0.1f  u1: %6lu (%4.1f%%)  u2: %6lu  (%4.1f%%)  u3: %6lu (%4.1f%%)\n",
      rinfo->coords_3d[m], pavg, rconns[0], pct1, rconns[1], pct2,
      rconns[2], pct3);
  }
#endif
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

    idx_t const justme = nrows;

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


static void __setup_mat_ptrs(
  rank_info * const rinfo,
  sptensor_t const * const tt)
{
  /* number of procs in layer */
  int npes;
  int lrank;
  MPI_Comm lcomm;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    lcomm = rinfo->layer_comm[m];
    MPI_Comm_size(lcomm, &npes);
    MPI_Comm_rank(lcomm, &lrank);

    /* allocate space for start/end idxs */
    rinfo->mat_ptrs[m] = (idx_t *) calloc(npes + 1, sizeof(idx_t));
    idx_t * const mat_ptrs = rinfo->mat_ptrs[m];

    mat_ptrs[lrank] = rinfo->mat_start[m];
    mat_ptrs[npes] = rinfo->layer_ends[m] - rinfo->layer_starts[m];

    /* Doing a reduce instead of a gather lets us set location mode_rank
     * instead of the rank in this communicator */
    MPI_Allreduce(MPI_IN_PLACE, mat_ptrs, npes, SS_MPI_IDX, MPI_SUM, lcomm);

    assert(rinfo->mat_ptrs[m][lrank] == rinfo->mat_start[m]);
    assert(rinfo->mat_ptrs[m][lrank + 1] == rinfo->mat_end[m]);
  }
}


/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

permutation_t * mpi_distribute_mats(
  rank_info * const rinfo,
  sptensor_t * const tt)
{
  permutation_t * perm = perm_alloc(tt->dims, tt->nmodes);

  __greedy_mat_distribution(rinfo, tt, perm);

  __setup_mat_ptrs(rinfo, tt);

  perm_apply(tt, perm->perms);

#if 0
  __write_part(tt, perm, rinfo);
#endif

  MPI_Barrier(MPI_COMM_WORLD);

  return perm;
}


void mpi_send_recv_stats(
  rank_info const * const rinfo,
  sptensor_t const * const tt)
{
  idx_t * psends = NULL;
  idx_t * precvs = NULL;

  if(rinfo->rank == 0) {
    printf("\n\n");
  }

  /* layer-specific MPI info */
  MPI_Comm comm;
  int rank;
  int size;

  /* count sends */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    comm = rinfo->layer_comm[m];
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    psends = (idx_t *) realloc(psends, size * sizeof(idx_t));
    precvs = (idx_t *) realloc(precvs, size * sizeof(idx_t));

    assert(psends != NULL);
    assert(precvs != NULL);

    memset(psends, 0, size * sizeof(idx_t));
    memset(precvs, 0, size * sizeof(idx_t));

    idx_t const max_sends = tt->dims[m];
    idx_t const max_recvs = tt->dims[m] * size;

    idx_t sends = 0;
    idx_t recvs = 0;
    idx_t local_rows = 0;

    idx_t const * const mat_ptrs = rinfo->mat_ptrs[m];

    int pdest = 0;

    for(idx_t i=0; i < tt->dims[m]; ++i) {
      /* grab global index */
      idx_t gi = i;
      if(tt->indmap[m] != NULL) {
        gi = tt->indmap[m][i];
      }
      assert(gi >= mat_ptrs[0]);
      assert(gi < mat_ptrs[size]);

      /* move to the next processor if necessary */
      while(gi >= mat_ptrs[pdest+1]) {
        ++pdest;
        assert(pdest < size);
      }

      assert(pdest < size);
      assert(gi >= mat_ptrs[pdest]);
      assert(gi < mat_ptrs[pdest+1]);

      /* if it is non-local */
      if(pdest != rank) {
        precvs[pdest] += 1;
        ++sends;
      } else {
        ++local_rows;
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, precvs, size, SS_MPI_IDX, MPI_SUM, comm);

    assert(rank < size);
    recvs = precvs[rank];

    MPI_Barrier(MPI_COMM_WORLD);

    double relsend = 100. * (double) sends / (double) max_sends;
    double relrecv = 100. * (double) recvs / (double) max_recvs;
    double pct_local = 100. * (double) local_rows / (double) tt->dims[m];
    printf("p: %d\tsends: %3lu (max: %3lu  %4.1f%%)\t"
                    "ratio: %0.1fx\t"
                    "recvs: %3lu (max: %3lu  %4.1f%%)\t"
                    "ratio: %0.1fx\t"
                    "local: %6lu (%4.1f%%)\n",
        rinfo->rank_3d,
        //rinfo->coords_3d[0], rinfo->coords_3d[1], rinfo->coords_3d[2],
        sends, max_sends, relsend,
        3. * (double) tt->nnz / (double) sends,
        recvs, max_recvs, relrecv,
        3. * (double) tt->nnz / (double) recvs,
        local_rows, pct_local);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) {
      printf("\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  free(psends);
  free(precvs);
}


void mpi_setup_comms(
  rank_info * const rinfo)
{
  MPI_Comm_size(MPI_COMM_WORLD, &(rinfo->npes));

  int * const dims_3d = rinfo->dims_3d;
  int periods[MAX_NMODES];

  /* get 3D cart dimensions - this can be improved! */
  int p13;
  int sqnpes = (int) sqrt(rinfo->npes) + 1;
  for(p13 = 1; p13 < sqnpes; ++p13) {
    if(p13 * p13 * p13 == rinfo->npes) {
      break;
    }
  }
  assert(p13 * p13 * p13 == rinfo->npes);
  rinfo->np13 = p13;

  dims_3d[0] = dims_3d[1] = dims_3d[2] = p13;
  periods[0] = periods[1] = periods[2] = p13;

  /* create new communicator and update global rank */
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims_3d, periods, 1, &(rinfo->comm_3d));
  MPI_Comm_rank(MPI_COMM_WORLD, &(rinfo->rank));
  MPI_Comm_rank(rinfo->comm_3d, &(rinfo->rank_3d));

  /* get 3d coordinates */
  MPI_Cart_coords(rinfo->comm_3d, rinfo->rank_3d, 3, rinfo->coords_3d);

  /* compute ranks relative to tensor mode */
  for(idx_t m=0; m < 3; ++m) {
    /* map coord within layer to 1D */
    int const coord1d = rinfo->coords_3d[(m+1) % 3] * rinfo->np13 +
                        rinfo->coords_3d[(m+2) % 3];
    int const layer_id = rinfo->coords_3d[m];
    /* relative rank in this mode */
    rinfo->mode_rank[m] = (rinfo->np13 * rinfo->np13 * layer_id) + coord1d;

    /* now split 3D communicator into layers */
    MPI_Comm_split(rinfo->comm_3d, layer_id, 0, &(rinfo->layer_comm[m]));
    MPI_Comm_rank(rinfo->layer_comm[m], &(rinfo->layer_rank[m]));
  }
}


sptensor_t * mpi_tt_read(
  char const * const ifname,
  rank_info * const rinfo)
{
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  idx_t nmodes;

  if(rank == 0) {
    /* get tensor stats */
    __get_dims(ifname, &(rinfo->global_nnz), &nmodes, rinfo->global_dims);
  }

  MPI_Bcast(&(rinfo->global_nnz), 1, SS_MPI_IDX, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nmodes, 1, SS_MPI_IDX, 0, MPI_COMM_WORLD);
  MPI_Bcast(rinfo->global_dims, nmodes, SS_MPI_IDX, 0, MPI_COMM_WORLD);

  idx_t * ssizes[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    ssizes[m] = (idx_t *) calloc(rinfo->global_dims[m], sizeof(idx_t));
  }

  __fill_ssizes(ifname, ssizes, nmodes, rinfo);

  /* actually parse tensor and then map to local (layer) coordinates  */
  sptensor_t * tt = __read_tt(ifname, ssizes, nmodes, rinfo);
  for(idx_t m=0; m < nmodes; ++m) {
    free(ssizes[m]);
    tt->dims[m] = rinfo->layer_ends[m] - rinfo->layer_starts[m];
    for(idx_t n=0; n < tt->nnz; ++n) {
      assert(tt->ind[m][n] >= rinfo->layer_starts[m]);
      assert(tt->ind[m][n] < rinfo->layer_ends[m]);
      tt->ind[m][n] -= rinfo->layer_starts[m];
    }
  }

  return tt;
}


void rank_free(
  rank_info rinfo,
  idx_t const nmodes)
{
  MPI_Comm_free(&rinfo.comm_3d);
  for(idx_t m=0; m < nmodes; ++m) {
    free(rinfo.mat_ptrs[m]);
    MPI_Comm_free(&rinfo.layer_comm[m]);
  }
}

