
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../splatt_mpi.h"
#include "../io.h"
#include "../timer.h"


/******************************************************************************
 * PRIVATE FUNCTONS
 *****************************************************************************/


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
  idx_t const * const dims = rinfo->global_dims;

  /* find start/end slices for my partition */
  for(idx_t m=0; m < nmodes; ++m) {
    idx_t const pnnz = nnz / rinfo->dims_3d[m]; /* nnz in a layer */
    /* current processor */
    int currp  = 0;
    idx_t lastn = 0;
    idx_t nnzcnt = 0;

    rinfo->layer_starts[m] = 0;
    rinfo->layer_ends[m] = dims[m];

#if 1
    for(idx_t s=0; s < dims[m]; ++s) {
      if(nnzcnt >= lastn + pnnz) {
        /* choose this slice or the previous, whichever is closer */
        if(s > 0) {
          idx_t const thisdist = nnzcnt - (lastn + pnnz);
          idx_t const prevdist = (lastn + pnnz) - (nnzcnt - ssizes[m][s-1]);
          if(prevdist < thisdist) {
            lastn = nnzcnt - ssizes[m][s-1];
          } else {
            lastn = nnzcnt;
          }
        } else {
          lastn = nnzcnt;
        }

        ++currp;
        if(currp == rinfo->coords_3d[m]) {
          rinfo->layer_starts[m] = s;
        } else if(currp == rinfo->coords_3d[m]+1
            && currp != rinfo->dims_3d[m]) {
          /* only set layer_end if we aren't at the end of the tensor */
          rinfo->layer_ends[m] = s;
          break;
        }
      }
      nnzcnt += ssizes[m][s];
    }
#else
    idx_t const ideal = dims[m] / rinfo->dims_3d[m];
    rinfo->layer_starts[m] = rinfo->coords_3d[m] * ideal;
    rinfo->layer_ends[m] = SS_MIN(dims[m], rinfo->coords_3d[m]+1 * ideal);
#endif

    /* it is possible to have a very small dimension and too many ranks */
    if(rinfo->dims_3d[m] > 1 && rinfo->layer_starts[m] == 0
        && rinfo->layer_ends[m] == dims[m]) {
      fprintf(stderr, "SPLATT: rank: %d too many MPI ranks for mode %lu.\n",
          rinfo->rank, m+1);
      rinfo->layer_starts[m] = dims[m];
      rinfo->layer_ends[m] = dims[m];
      //abort();
    }
  }
}


static void __find_my_slices_1d(
  idx_t ** const ssizes,
  idx_t const nmodes,
  idx_t const nnz,
  rank_info * const rinfo)
{
  idx_t const pnnz = nnz / rinfo->npes;
  idx_t const * const dims = rinfo->global_dims;

  /* find start/end slices for my partition */
  for(idx_t m=0; m < nmodes; ++m) {
    /* current processor */
    int currp  = 0;
    idx_t lastn = 0;
    idx_t nnzcnt = 0;

    rinfo->layer_starts[m] = 0;
    rinfo->layer_ends[m] = dims[m];

    rinfo->mat_start[m] = 0;
    rinfo->mat_end[m] = dims[m];

    for(idx_t s=0; s < dims[m]; ++s) {
      if(nnzcnt >= lastn + pnnz) {
        /* choose this slice or the previous, whichever is closer */
        if(s > 0) {
          idx_t const thisdist = nnzcnt - (lastn + pnnz);
          idx_t const prevdist = (lastn + pnnz) - (nnzcnt - ssizes[m][s-1]);
          if(prevdist < thisdist) {
            lastn = nnzcnt - ssizes[m][s-1];
          } else {
            lastn = nnzcnt;
          }
        } else {
          lastn = nnzcnt;
        }

        ++currp;
        if(currp == rinfo->rank) {
          rinfo->mat_start[m] = s;
        } else if(currp == rinfo->rank+1 && currp != rinfo->npes) {
          /* only set mat_end if we aren't at the end of the tensor */
          rinfo->mat_end[m] = s;
          break;
        }
      }
      nnzcnt += ssizes[m][s];

      if(rinfo->rank == rinfo->npes-1) {
        assert(rinfo->mat_end[m] == rinfo->global_dims[m]);
      }
    }

    /* it is possible to have a very small dimension and too many ranks */
    if(rinfo->npes > 1 && rinfo->mat_start[m] == 0
        && rinfo->mat_end[m] == dims[m]) {
      fprintf(stderr, "SPLATT: rank: %d too many MPI ranks for mode %lu.\n",
          rinfo->rank, m+1);
      rinfo->mat_start[m] = dims[m];
      rinfo->mat_end[m] = dims[m];
      //abort();
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
* @brief Count the nonzero values in a partition of X.
*
* @param fname The name of the file containing X.
* @param nmodes The number of modes of X.
*
* @return The number of nonzeros in the intersection of all sstarts and sends.
*/
static idx_t __count_my_nnz_1d(
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
      int mine = 0;
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10) - 1;
        /* I own the nnz if it falls in any of my slices */
        if(ind >= sstarts[m] && ind < sends[m]) {
          mine = 1;
          break;
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
* @brief Read a partition of X into tt.
*
* @param fname The file containing X.
* @param tt The tensor structure (must be pre-allocated).
* @param sstarts Array of starting slices, inclusive (one for each mode).
* @param sends Array of ending slices, exclusive (one for each mode).
*/
static void __read_tt_part_1d(
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
      int mine = 0;
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10) - 1;
        tt->ind[m][nnzread] = ind;
        if(ind >= sstarts[m] && ind < sends[m]) {
          mine = 1;
        }
      }
      tt->vals[nnzread] = strtod(ptr, &ptr);
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
static sptensor_t * __read_tt_1d(
  char const * const fname,
  idx_t ** const ssizes,
  idx_t const nmodes,
  rank_info * const rinfo)
{
  int const rank = rinfo->rank;
  idx_t const nnz = rinfo->global_nnz;
  idx_t const * const dims = rinfo->global_dims;

  /* find start/end slices for my partition */
  __find_my_slices_1d(ssizes, nmodes, nnz, rinfo);

  /* count nnz in my partition and allocate */
  idx_t const mynnz = __count_my_nnz_1d(fname, nmodes, rinfo->mat_start,
      rinfo->mat_end);
  sptensor_t * tt = tt_alloc(mynnz, nmodes);

  /* now actually load values */
  __read_tt_part_1d(fname, tt, rinfo->mat_start, rinfo->mat_end);

  return tt;
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
static sptensor_t * __read_tt_3d(
  char const * const fname,
  idx_t ** const ssizes,
  idx_t const nmodes,
  rank_info * const rinfo)
{
  int const rank = rinfo->rank_3d;
  idx_t const nnz = rinfo->global_nnz;
  idx_t const * const dims = rinfo->global_dims;

  /* find start/end slices for my partition */
  __find_my_slices(ssizes, nmodes, nnz, rinfo);

  /* count nnz in my partition and allocate */
  idx_t const mynnz = __count_my_nnz(fname, nmodes, rinfo->layer_starts,
      rinfo->layer_ends);
  sptensor_t * tt = tt_alloc(mynnz, nmodes);

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
  int const rank = rinfo->rank;
  int const size = rinfo->npes;

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

/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

sptensor_t * mpi_tt_read(
  char const * const ifname,
  rank_info * const rinfo)
{
  timer_start(&timers[TIMER_IO]);
  idx_t nmodes;

  if(rinfo->rank == 0) {
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

  /* actually parse tensor */
  sptensor_t * tt = NULL;
  switch(rinfo->distribution) {
  case 1:
    tt = __read_tt_1d(ifname, ssizes, nmodes, rinfo);
    /* now fix tt->dims */
    for(idx_t m=0; m < tt->nmodes; ++m) {
      tt->dims[m] = 0;
      for(idx_t n=0; n < tt->nnz; ++n) {
        tt->dims[m] = SS_MAX(tt->dims[m], tt->ind[m][n] + 1);
      }
    }
    break;

  case 3:
    tt = __read_tt_3d(ifname, ssizes, nmodes, rinfo);
    /* now map tensor indices to local (layer) coordinates and fill in dims */
    for(idx_t m=0; m < nmodes; ++m) {
      free(ssizes[m]);
      tt->dims[m] = rinfo->layer_ends[m] - rinfo->layer_starts[m];
      for(idx_t n=0; n < tt->nnz; ++n) {
        assert(tt->ind[m][n] >= rinfo->layer_starts[m]);
        assert(tt->ind[m][n] < rinfo->layer_ends[m]);
        tt->ind[m][n] -= rinfo->layer_starts[m];
      }
    }
    break;
  }

  timer_stop(&timers[TIMER_IO]);

  return tt;
}


void mpi_filter_tt_1d(
  idx_t const mode,
  sptensor_t const * const tt,
  sptensor_t * const ftt,
  idx_t start,
  idx_t end)
{
  assert(ftt != NULL);

  for(idx_t m=0; m < ftt->nmodes; ++m) {
    ftt->dims[m] = tt->dims[m];
  }

  idx_t const olds = start;
  idx_t const olde = end;
  /* Adjust start and end if tt has been compressed. */
  assert(start != end);
  if(tt->indmap[mode] != NULL) {
    /* TODO: change this linear search into a binary one */
    for(idx_t i=0; i < tt->dims[mode]; ++i) {
      if(tt->indmap[mode][i] == start) {
        start = i;
      }
      if(tt->indmap[mode][i]+1 == end) {
        end = i+1;
        break;
      }
    }
  }

  idx_t nnz = 0;
  for(idx_t n=0; n < tt->nnz; ++n) {
    /* Copy the nonzero if we own the slice. */
    if(tt->ind[mode][n] >= start && tt->ind[mode][n] < end) {
      for(idx_t m=0; m < tt->nmodes; ++m) {
        ftt->ind[m][nnz] = tt->ind[m][n];
      }
      ftt->vals[nnz++] = tt->vals[n];
    }
  }

  /* update ftt dimensions and nnz */
  ftt->nnz = nnz;
  ftt->dims[mode] = end - start;

  /* now map mode coords to [0, end-start) */
  for(idx_t n=0; n < ftt->nnz; ++n) {
    assert(ftt->ind[mode][n] >= start);
    assert(ftt->ind[mode][n] < end);
    ftt->ind[mode][n] -= start;
  }

  /* create new indmap for mode */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(tt->indmap[m] == NULL) {
      break;
    }
    ftt->indmap[m] = (idx_t *) realloc(ftt->indmap[m],
        ftt->dims[m] * sizeof(idx_t));

    /* mode indices are shifted. otherwise just copy */
    if(m == mode) {
      for(idx_t i=0; i < ftt->dims[mode]; ++i) {
        ftt->indmap[mode][i] = tt->indmap[mode][i+start];
      }
    } else {
      memcpy(ftt->indmap[m], tt->indmap[m], tt->dims[m] * sizeof(idx_t));
    }
  }

  /* sanity check */
  for(idx_t i=0; i < ftt->dims[mode]; ++i) {
    assert(i + start < end);
  }
  for(idx_t n=0; n < ftt->nnz; ++n) {
    assert(ftt->ind[mode][n] < end - start);
  }
}


void mpi_write_mats(
  matrix_t ** mats,
  permutation_t const * const perm,
  rank_info const * const rinfo,
  char const * const basename,
  idx_t const nmodes)
{
  char * fname;
  idx_t const nfactors = mats[0]->J;

  MPI_Status status;

  idx_t maxdim = 0;
  idx_t maxlocaldim = 0;
  matrix_t * matbuf = NULL;
  val_t * vbuf = NULL;
  idx_t * loc_iperm = NULL;

  for(idx_t m=0; m < nmodes; ++m) {
    maxdim = SS_MAX(maxdim, rinfo->global_dims[m]);
    maxlocaldim = SS_MAX(maxlocaldim, mats[m]->I);
  }

  /* get the largest local dim */
  if(rinfo->rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &maxlocaldim, 1, SS_MPI_IDX, MPI_MAX, 0,
      rinfo->comm_3d);
  } else {
    MPI_Reduce(&maxlocaldim, NULL, 1, SS_MPI_IDX, MPI_MAX, 0,
      rinfo->comm_3d);
  }

  if(rinfo->rank == 0) {
    matbuf = mat_alloc(maxdim, nfactors);
    loc_iperm = (idx_t *) malloc(maxdim * sizeof(idx_t));
    vbuf = (val_t *) malloc(maxdim * nfactors * sizeof(val_t));
  }

  for(idx_t m=0; m < nmodes; ++m) {
    /* root handles the writing */
    if(rinfo->rank == 0) {
      asprintf(&fname, "%s%"SS_IDX".mat", basename, m);
      matbuf->I = rinfo->global_dims[m];

      /* copy root's matrix to buffer */
      for(idx_t i=0; i < mats[m]->I; ++i) {
        idx_t const gi = rinfo->layer_starts[m] + perm->iperms[m][i];
        for(idx_t f=0; f < nfactors; ++f) {
          matbuf->vals[f + (gi*nfactors)] = mats[m]->vals[f+(i*nfactors)];
        }
      }

      /* receive matrix from each rank */
      for(int p=1; p < rinfo->npes; ++p) {
        idx_t layerstart;
        idx_t nrows;
        MPI_Recv(&layerstart, 1, SS_MPI_IDX, p, 0, rinfo->comm_3d, &status);
        MPI_Recv(&nrows, 1, SS_MPI_IDX, p, 0, rinfo->comm_3d, &status);
        MPI_Recv(vbuf, nrows * nfactors, SS_MPI_VAL, p, 0, rinfo->comm_3d,
            &status);
        MPI_Recv(loc_iperm, nrows, SS_MPI_IDX, p, 0, rinfo->comm_3d, &status);

        /* permute buffer and copy into matbuf */
        for(idx_t i=0; i < nrows; ++i) {
          idx_t const gi = layerstart + loc_iperm[i];
          for(idx_t f=0; f < nfactors; ++f) {
            matbuf->vals[f + (gi*nfactors)] = vbuf[f+(i*nfactors)];
          }
        }
      }

      /* write the factor matrix to disk */
      mat_write(matbuf, fname);

      /* clean up */
      free(fname);
    } else {
      /* send matrix to root */
      MPI_Send(&(rinfo->layer_starts[m]), 1, SS_MPI_IDX, 0, 0, rinfo->comm_3d);
      MPI_Send(&(mats[m]->I), 1, SS_MPI_IDX, 0, 0, rinfo->comm_3d);
      MPI_Send(mats[m]->vals, mats[m]->I * mats[m]->J, SS_MPI_VAL, 0, 0,
          rinfo->comm_3d);
      MPI_Send(perm->iperms[m] + rinfo->mat_start[m], mats[m]->I, SS_MPI_IDX,
          0, 0, rinfo->comm_3d);
    }
  } /* foreach mode */


  if(rinfo->rank == 0) {
    mat_free(matbuf);
    free(vbuf);
    free(loc_iperm);
  }
}


void mpi_write_part(
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

      /* write index */
      fprintf(fout, "%"SS_IDX" ", 1+idx);
    }
    fprintf(fout, "%"SS_VAL"\n", tt->vals[n]);
  }
  fclose(fout);
}

