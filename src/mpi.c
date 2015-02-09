
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "mpi.h"


/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/
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


