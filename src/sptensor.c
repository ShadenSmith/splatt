

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
static void __tt_remove_dups(
  sptensor_t * const tt)
{
  tt_sort(tt, 0, NULL);

  for(idx_t n=0; n < tt->nnz - 1; ++n) {
    int same = 1;
    for(idx_t m=0; m < tt->nmodes; ++m) {
      if(tt->ind[m][n] != tt->ind[m][n+1]) {
        same = 0;
        break;
      }
    }
    if(same) {
      tt->vals[n] = (tt->vals[n] + tt->vals[n+1]) / 2;
      for(idx_t m=0; m < tt->nmodes; ++m) {
        memmove(&(tt->ind[m][n]), &(tt->ind[m][n+1]),
          (tt->nnz-n-1)*sizeof(idx_t));
      }
      --n;
      tt->nnz -= 1;
    }
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
  return tt;
}


sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes)
{
  sptensor_t * tt = (sptensor_t*) malloc(sizeof(sptensor_t));

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

  idx_t row = 0;
  rowptr[row++] = 0;
  for(idx_t n=0; n < tt->nnz; ++n) {
    /* increment row and account for possibly empty ones */
    while(tt->ind[mode][n] != row-1) {
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
  rowptr[nrows] = tt->nnz;

  return mat;
}

