
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "mpi.h"
#include "io.h"
#include <string.h>

static void __get_dims(
  char const * const fname,
  idx_t * const outnnz,
  idx_t * const outdims)
{
  FILE * fin;
  if((fin = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n", fname);
    exit(1);
  }

  /* first count nnz in tensor */
  char * ptr = NULL;
  idx_t nnz = 0;
  idx_t nmodes = 0;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    if(read > 1 && line[0] != '#') {
      /* get nmodes from first nnz line */
      if(nnz == 0) {
        ptr = strtok(line, " \t");
        while(ptr != NULL) {
          ++nmodes;
          ptr = strtok(NULL, " \t");
        }
      }
      ++nnz;
    }
  }
  --nmodes;

  *outnnz = nnz;

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
    }
  }
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

  idx_t dims[MAX_NMODES];
  idx_t nnz;

  if(rank == 0) {
    /* get tensor stats */
    __get_dims(ifname, &nnz, dims);
    printf("found %lu nnz and %lu %lu %lu\n", nnz, dims[0], dims[1], dims[2]);
  } else {

  }

  sptensor_t * tt = tt_read(ifname);

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
    ssizes[0][tt->ind[0][n]] += 1;
    ssizes[1][tt->ind[1][n]] += 1;
    ssizes[2][tt->ind[2][n]] += 1;
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


