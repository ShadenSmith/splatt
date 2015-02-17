
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "mpi.h"
#include "io.h"
#include <string.h>
#include <math.h>



/******************************************************************************
 * PRIVATE FUNCTONS
 *****************************************************************************/

/**
* @brief Write a tensor to file <rank>.part. All local indices are converted to
*        global.
*
* @param tt The tensor to write.
*/
static void __write_part(
  sptensor_t const * const tt,
  int const rank)
{
  /* each process outputs their own view of X for testing */
  char name[256];
  sprintf(name, "%d.part", rank);
  FILE * fout = open_f(name, "w");
  for(idx_t n=0; n < tt->nnz; ++n) {
    for(idx_t m=0; m < tt->nmodes; ++m) {
      if(tt->indmap[m] != NULL) {
        fprintf(fout, "%lu\t", 1+tt->indmap[m][tt->ind[m][n]]);
      } else {
        fprintf(fout, "%lu\t", 1+tt->ind[m][n]);
      }
    }
    fprintf(fout, "%d\n", (int) tt->vals[n]);
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
* @param sstarts Array of slice starts, inclusive (one for each mode).
* @param sends Array of slice ends, exclusive (one for each mode).
*/
static void __find_my_slices(
  idx_t ** const ssizes,
  idx_t const nmodes,
  idx_t const nnz,
  rank_info const * const rinfo,
  idx_t * const sstarts,
  idx_t * const sends)
{
  idx_t const pnnz = nnz / rinfo->np13; /* nnz in a layer */
  idx_t const * const dims = rinfo->global_dims;

  /* find start/end slices for my partition */
  for(idx_t m=0; m < nmodes; ++m) {
    /* current processor */
    int currp  = 0;
    idx_t lastn = 0;
    idx_t nnzcnt = 0;

    sstarts[m] = 0;
    sends[m] = dims[m];

    for(idx_t s=0; s < dims[m]; ++s) {
      if(nnzcnt >= lastn + pnnz) {
        lastn = nnzcnt;
        ++currp;
        if(currp == rinfo->coords_3d[m]) {
          sstarts[m] = s;
        } else if(currp == rinfo->coords_3d[m]+1) {
          sends[m] = s;
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
* @param sstarts Array of slice starts, inclusive (one for each mode).
* @param sends Array of slice ends, exclusive (one for each mode).
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

  idx_t sstarts[MAX_NMODES];
  idx_t sends[MAX_NMODES];

  /* find start/end slices for my partition */
  __find_my_slices(ssizes, nmodes, nnz, rinfo, sstarts, sends);

  /* count nnz in my partition and allocate */
  idx_t const mynnz = __count_my_nnz(fname, nmodes, sstarts, sends);
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
  __read_tt_part(fname, tt, sstarts, sends);

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

  /* reduce to get total slice counts */
  for(idx_t m=0; m < nmodes; ++m) {
    MPI_Allreduce(MPI_IN_PLACE, ssizes[m], dims[m], SS_MPI_IDX, MPI_SUM,
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

  fclose(fin);
}



/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

void mpi_distribute_mats(
  rank_info * const rinfo,
  sptensor_t * const tt)
{
  if(rinfo->rank == 0) {
    printf("\n");
  }

#if 0
  for(idx_t m=0; m < tt->nmodes; ++m) {
    int const layer_id = rinfo->coords_3d[m];
    idx_t const layer_size = rinfo->global_dims[m] / rinfo->np13;
    idx_t const start = layer_id * layer_size;
    idx_t end = (layer_id + 1) * layer_size;
    /* account for last layer having extras */
    if(layer_id == rinfo->np13 - 1) {
      end = rinfo->global_dims[m];
    }

    /* target nrows = layer_size / npes in a layer */
    idx_t const psize = (end - start) / (rinfo->np13 * rinfo->np13);

    /* map coord within layer to 1D */
    int const coord1d = rinfo->coords_3d[(m+1)%tt->nmodes] * rinfo->np13 +
                        rinfo->coords_3d[(m+2)%tt->nmodes];

    rinfo->mat_start[m] = start + (coord1d * psize);
    rinfo->mat_end[m]   = start + ((coord1d + 1) * psize);
    if(coord1d == (rinfo->np13 * rinfo->np13) - 1) {
      rinfo->mat_end[m] = end;
    }
  }
#endif

  idx_t max_dim = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(rinfo->global_dims[m] > max_dim) {
      max_dim = rinfo->global_dims[m];
    }
  }

  idx_t * pop = (idx_t *) malloc(max_dim * sizeof(idx_t));
  idx_t * mine = (idx_t *) malloc(max_dim * sizeof(idx_t));
  for(idx_t m=0; m < tt->nmodes; ++m) {
    memset(pop, 0, rinfo->global_dims[m] * sizeof(idx_t));
    memset(mine, 0, rinfo->global_dims[m] * sizeof(idx_t));

    /* communication volume */
    idx_t vol = 0;

    /* number of rows I own */
    idx_t nrows = 0;

    /* mark all idxs that local to me */
    for(idx_t n=0; n < tt->nnz; ++n) {
      pop[tt->indmap[m][tt->ind[m][n]]] = 1;
    }
    /* sum appearances to get communication volume */
    MPI_Allreduce(MPI_IN_PLACE, pop, rinfo->global_dims[m], SS_MPI_IDX, MPI_SUM,
      rinfo->comm_3d);

    for(idx_t i=0; i < tt->dims[m]; ++i) {
      idx_t const gi = tt->indmap[m][i];
      /* claim all rows that are entirely local to me */
      if(pop[gi] == 1) {
        mine[nrows++] = gi;
      }
    }

#if 1
    /* count u=1; u=2, u > 2 */
    idx_t u1 = 1;
    idx_t u2 = 1;
    idx_t u3 = 1;
    for(idx_t i=0; i < rinfo->global_dims[m]; ++i) {
      switch(pop[i]) {
      case 0:
        break;
      case 1:
        ++u1;
        break;
      case 2:
        ++u2;
        break;
      default:
        ++u3;
        break;
      }
    }
    if(rinfo->rank == 0) {
      double pct1 = 100. * (double) u1 / rinfo->global_dims[m];
      double pct2 = 100. * (double) u2 / rinfo->global_dims[m];
      double pct3 = 100. * (double) u3 / rinfo->global_dims[m];
      printf("u1: %6lu (%4.1f%%)  u2: %6lu  (%4.1f%%)  u3: %6lu (%4.1f%%)\n",
        u1, pct1, u2, pct2, u3, pct3);
    }
#endif
  }
  free(pop);
  free(mine);
}



void mpi_send_recv_stats(
  rank_info const * const rinfo,
  sptensor_t const * const tt)
{
  int const rank = rinfo->rank_3d;
  int const size = rinfo->npes;

  idx_t max_sends = rinfo->np13 * rinfo->np13;
  idx_t max_recvs = rinfo->np13 * rinfo->np13;

  idx_t * psends = (idx_t *) malloc(size * sizeof(idx_t));
  idx_t * precvs = (idx_t *) malloc(size * sizeof(idx_t));

  if(rank == 0) {
    printf("\n\n");
  }

  /* count sends */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    memset(psends, 0, size * sizeof(idx_t));
    memset(precvs, 0, size * sizeof(idx_t));

    idx_t sends = 0;
    idx_t recvs = 0;
    idx_t local_rows = 0;

    /* lets us statically assign indices to ranks */
    idx_t const msize = rinfo->global_dims[m] / size;

    for(idx_t i=0; i < tt->dims[m]; ++i) {
      /* grab global index */
      idx_t const gi = tt->indmap[m][i];

      /* see if it can't be found locally */
      if(gi < rinfo->mat_start[m] || gi >= rinfo->mat_end[m]) {
        /* Compute the destination rank. The last rank handles the leftover
         * indices, so account for that. */
        int pdest = (int) (gi / msize);
        if(pdest == size) {
          --pdest;
        }
        assert(pdest < size);
        if(psends[pdest]++ == 0) {
          ++sends;
        }

        precvs[pdest] = 1;
      } else {
        ++local_rows;
      }
    }

    if(local_rows == 0) {
      printf("NO LOCALS: %d,%d,%d\t\t A: (%6lu - %6lu) X: (%6lu - %6lu)\n",
        rinfo->coords_3d[0], rinfo->coords_3d[1], rinfo->coords_3d[2],
        rinfo->mat_start[m], rinfo->mat_end[m],
        tt->indmap[m][0], tt->indmap[m][tt->dims[m]-1]);
    }

    MPI_Allreduce(MPI_IN_PLACE, precvs, size, SS_MPI_IDX, MPI_SUM,
        rinfo->comm_3d);

    recvs = precvs[rank];

    double relsend = 100. * (double) sends / (double) max_sends;
    double relrecv = 100. * (double) recvs / (double) max_recvs;
    double pct_local = 100. * (double) local_rows / (double) tt->dims[m];
    printf("p: %d,%d,%d\t\tsends: %3lu (max: %3lu  %4.1f%%)\t"
                    "recvs: %3lu (max: %3lu  %4.1f%%)\t"
                    "local: %6lu (%4.1f%%)\n",
        rinfo->coords_3d[0], rinfo->coords_3d[1], rinfo->coords_3d[2],
        sends, max_sends, relsend,
        recvs, max_recvs, relrecv,
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

  /* actually parse tensor */
  sptensor_t * tt = __read_tt(ifname, ssizes, nmodes, rinfo);
  for(idx_t m=0; m < nmodes; ++m) {
    free(ssizes[m]);
    tt->dims[m] = rinfo->global_dims[m];
  }

  /* clean up tensor */
  tt_remove_dups(tt);
  tt_remove_empty(tt);

  return tt;
}



