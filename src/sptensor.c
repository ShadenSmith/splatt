

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "matrix.h"
#include "sort.h"
#include "io.h"


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Remove the duplicate entries of a tensor. Duplicate values are
*        repeatedly averaged.
*
* @param tt The modified tensor to work on. NOTE: data structures are not
*           resized!
*/
static void __tt_remove_dups(
  sptensor_t * const tt)
{
  tt_sort(tt, 0, NULL);

  idx_t nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;

  for(idx_t n=0; n < nnz - 1; ++n) {
    int same = 1;
    for(idx_t m=0; m < nmodes; ++m) {
      if(tt->ind[m][n] != tt->ind[m][n+1]) {
        same = 0;
        break;
      }
    }
    if(same) {
      tt->vals[n] = (tt->vals[n] + tt->vals[n+1]) / 2;
      for(idx_t m=0; m < nmodes; ++m) {
        memmove(&(tt->ind[m][n]), &(tt->ind[m][n+1]),
          (nnz-n-1)*sizeof(idx_t));
      }
      --n;
      nnz -= 1;
    }
  }
  tt->nnz = nnz;
}



static void __tt_distribute_stats(
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

/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/
sptensor_t * tt_read(
  char const * const ifname)
{
  sptensor_t * tt = tt_read_file(ifname);
  __tt_remove_dups(tt);

  /* XXX */
  for(idx_t p=2; p <= 22; p += 1) {
    __tt_distribute_stats(tt, p);
  }

  printf("\n");

  return tt;
}


sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes)
{
  sptensor_t * tt = (sptensor_t*) malloc(sizeof(sptensor_t));
  tt->tiled = 0;

  tt->nnz = nnz;
  tt->vals = (val_t*) malloc(nnz * sizeof(val_t));

  tt->nmodes = nmodes;
  tt->type = (nmodes == 3) ? SPLATT_3MODE : SPLATT_NMODE;

  tt->dims = (idx_t*) malloc(nmodes * sizeof(idx_t));
  tt->ind = (idx_t**) malloc(nmodes * sizeof(idx_t*));
  for(idx_t m=0; m < nmodes; ++m) {
    tt->ind[m] = (idx_t*) malloc(nnz * sizeof(idx_t));
  }

  return tt;
}

void tt_free(
  sptensor_t * tt)
{
  tt->nnz = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    free(tt->ind[m]);
  }
  tt->nmodes = 0;
  free(tt->dims);
  free(tt->ind);
  free(tt->vals);
  free(tt);
}

spmatrix_t * tt_unfold(
  sptensor_t * const tt,
  idx_t const mode)
{
  idx_t nrows = tt->dims[mode];
  idx_t ncols = 1;

  for(idx_t m=1; m < tt->nmodes; ++m) {
    ncols *= tt->dims[(mode + m) % tt->nmodes];
  }

  /* sort tt */
  tt_sort(tt, mode, NULL);

  /* allocate and fill matrix */
  spmatrix_t * mat = spmat_alloc(nrows, ncols, tt->nnz);
  idx_t * const rowptr = mat->rowptr;
  idx_t * const colind = mat->colind;
  val_t * const mvals  = mat->vals;

  /* make sure to skip ahead to the first non-empty slice */
  idx_t row = 0;
  for(idx_t n=0; n < tt->nnz; ++n) {
    /* increment row and account for possibly empty ones */
    while(row <= tt->ind[mode][n]) {
      rowptr[row++] = n;
    }
    mvals[n] = tt->vals[n];

    idx_t col = 0;
    idx_t mult = 1;
    for(idx_t m = 0; m < tt->nmodes; ++m) {
      idx_t const off = tt->nmodes - 1 - m;
      if(off == mode) {
        continue;
      }
      col += tt->ind[off][n] * mult;
      mult *= tt->dims[off];
    }

    colind[n] = col;
  }
  /* account for any empty rows at end, too */
  for(idx_t r=row; r <= nrows; ++r) {
    rowptr[r] = tt->nnz;
  }

  return mat;
}

