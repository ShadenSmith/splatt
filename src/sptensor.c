

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "matrix.h"
#include "sort.h"
#include "io.h"
#include "mpi.h"
#include "timer.h"


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


/**
* @brief Relabel tensor indices to remove empty slices. Local -> global mapping
*        is written to tt->indmap.
*
* @param tt The tensor to relabel.
*/
static void __tt_remove_empty(
  sptensor_t * const tt)
{
  idx_t dim_sizes[MAX_NMODES];

  /* Allocate indmap */
  idx_t const nmodes = tt->nmodes;
  idx_t const nnz = tt->nnz;

  for(idx_t m=0; m < nmodes; ++m) {
    dim_sizes[m] = 0;
    tt->indmap[m] = (idx_t *) calloc(tt->dims[m], sizeof(idx_t));

    /* Fill in indmap */
    for(idx_t n=0; n < tt->nnz; ++n) {
      /* keep track of #unique slices */
      if(tt->indmap[m][tt->ind[m][n]] == 0) {
        ++dim_sizes[m];
      }
      tt->indmap[m][tt->ind[m][n]] = 1;
    }

    /* move on if no remapping is necessary */
    if(dim_sizes[m] == tt->dims[m]) {
      free(tt->indmap[m]);
      tt->indmap[m] = NULL;
      continue;
    }

    /* Now scan to remove empty slices */
    idx_t ptr = 0;
    for(idx_t i=0; i < tt->dims[m]; ++i) {
      /* move ptr to the next non-empty slice */
      while(tt->indmap[m][ptr] == 0) {
        ++ptr;
      }
      tt->indmap[m][i] = ptr;
    }

    /* relabel all indices in mode m */
    tt->dims[m] = dim_sizes[m];
    for(idx_t n=0; n < tt->nnz; ++n) {
      tt->ind[m][n] = tt->indmap[m][tt->ind[m][n]];
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

  /* remove duplicates and empty slices */
  __tt_remove_dups(tt);
  __tt_remove_empty(tt);

#if 0
  /* XXX */
  for(idx_t p=2; p <= 22; p += 1) {
    tt_distribute_stats(tt, p);
  }
  printf("\n");
#endif

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
    tt->indmap[m] = NULL;
  }

  return tt;
}

void tt_free(
  sptensor_t * tt)
{
  tt->nnz = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    free(tt->ind[m]);
    free(tt->indmap[m]);
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

