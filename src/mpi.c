
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "mpi.h"
#include "io.h"
#include <string.h>



/******************************************************************************
 * PRIVATE FUNCTONS
 *****************************************************************************/

static sptensor_t * __read_tt(
  char const * const fname,
  idx_t ** const ssizes,
  idx_t const nnz,
  idx_t const nmodes,
  idx_t const * const dims)
{
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  idx_t sstarts[MAX_NMODES];
  idx_t sends[MAX_NMODES];

  int const p13 = size / 2;
  int const p23 = size / 4;
  idx_t const pnnz = nnz / 2;

  if(rank == 0) {
    printf("pnnz: %lu target nnz: %lu\n", pnnz, nnz/size);
  }
  MPI_Barrier(MPI_COMM_WORLD);

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
        if(currp == rank/p13) {
          sstarts[m] = s;
        } else if(currp == (rank/p13)+1) {
          sends[m] = s;
          break;
        }
      }
      nnzcnt += ssizes[m][s];
    }
  }

  FILE * fin = open_f(fname, "r");

  char * ptr = NULL;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  sptensor_t * tt = NULL;
  idx_t mynnz = 0;

  /* count nnz in my partition */
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

  printf("p: %d (%lu x %lu x %lu) (%lu x %lu x %lu) -> %lu\n", rank, 
    sstarts[0], sstarts[1], sstarts[2],
    sends[0], sends[1], sends[2],
    mynnz);

  return tt;
}



static void __fill_ssizes(
  char const * const fname,
  idx_t ** const ssizes,
  idx_t const nnz,
  idx_t const nmodes,
  idx_t const * const dims)
{
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

sptensor_t * mpi_tt_read(
  char const * const ifname)
{
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  idx_t nmodes;
  idx_t nnz;
  idx_t dims[MAX_NMODES];

  if(rank == 0) {
    /* get tensor stats */
    __get_dims(ifname, &nnz, &nmodes, dims);
    printf("found %lu nnz and %lu %lu %lu\n", nnz, dims[0], dims[1], dims[2]);
  }

  MPI_Bcast(&nnz, 1, SS_MPI_IDX, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nmodes, 1, SS_MPI_IDX, 0, MPI_COMM_WORLD);
  MPI_Bcast(dims, nmodes, SS_MPI_IDX, 0, MPI_COMM_WORLD);

  idx_t * ssizes[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    ssizes[m] = (idx_t *) calloc(dims[m], sizeof(idx_t));
  }

  __fill_ssizes(ifname, ssizes, nnz, nmodes, dims);
  sptensor_t * tt = __read_tt(ifname, ssizes, nnz, nmodes, dims);


  for(idx_t m=0; m < nmodes; ++m) {
    free(ssizes[m]);
  }
  return tt;
}



void tt_distribute_stats(
  sptensor_t * const tt,
  idx_t const nprocs)
{
  idx_t const nnz = tt->nnz;
  idx_t const pnnz = nnz / nprocs;

  idx_t * ssizes[MAX_NMODES]; /* nnz per slice */
  idx_t * pmks[MAX_NMODES];   /* start idxs of each partition */
  idx_t * psizes = (idx_t *) calloc(nprocs * nprocs * nprocs, sizeof(idx_t));

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ssizes[m] = (idx_t *) calloc(tt->dims[m], sizeof(idx_t));
    pmks[m] = (idx_t *) malloc((nprocs+1) * sizeof(idx_t));
    pmks[m][0] = 0;
  }

  /* fill ssizes */
  for(idx_t n=0; n < nnz; ++n) {
    for(idx_t m=0; m < tt->nmodes; ++m) {
      ssizes[m][tt->ind[m][n]] += 1;
    }
  }

  for(idx_t m=0; m < tt->nmodes; ++m) {
    idx_t lastn  = 0;
    idx_t currp  = 0; /* current processor */
    idx_t nnzcnt = 0;

    for(idx_t s=0; s < tt->dims[m]; ++s) {
      if(nnzcnt >= lastn + pnnz) {
        lastn = nnzcnt;
        ++currp;
        pmks[m][currp] = s;
      }
      nnzcnt += ssizes[m][s];
    }
    pmks[m][nprocs] = tt->dims[m];
  }


  /* now count psizes */
  for(idx_t n=0; n < nnz; ++n) {
    /* find which proc nnz belongs to */
    idx_t myproci = 0;
    idx_t myprocj = 0;
    idx_t myprock = 0;

    while(tt->ind[0][n] > pmks[0][myproci+1]) {
      ++myproci;
    }
    while(tt->ind[1][n] > pmks[1][myprocj+1]) {
      ++myprocj;
    }
    while(tt->ind[2][n] > pmks[2][myprock+1]) {
      ++myprock;
    }

    assert(myproci <= nprocs);
    assert(myprocj <= nprocs);
    assert(myprock <= nprocs);

    psizes[myprock + (myprocj*nprocs) + (myproci*nprocs*nprocs)] += 1;
  }

  idx_t totn = 0;
  idx_t minp = nnz;
  idx_t maxp = 0;
  for(idx_t p=0; p < nprocs * nprocs * nprocs; ++p) {
    totn += psizes[p];
    if(psizes[p] < minp) {
      minp = psizes[p];
    }
    if(psizes[p] > maxp) {
      maxp = psizes[p];
    }
  }
  assert(totn == nnz);

  idx_t const p3 = nnz / (nprocs * nprocs * nprocs);
  printf("np: %5lu    pnnz: %8lu    min: %8lu    max: %8lu (%0.2f%% diff)\n",
    nprocs * nprocs * nprocs,
    p3,
    minp,
    maxp, 100. * ((double)maxp - (double)p3)/(double)maxp);

  free(psizes);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    free(pmks[m]);
    free(ssizes[m]);
  }
}


