

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

idx_t tt_remove_dups(
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
      printf("DUPLICATE NONZERO: ");
      tt->vals[n] = (tt->vals[n] + tt->vals[n+1]) / 2;
      for(idx_t m=0; m < nmodes; ++m) {
        printf(SS_IDX" ", tt->ind[m][n]);
        memmove(&(tt->ind[m][n]), &(tt->ind[m][n+1]),
          (nnz-n-1)*sizeof(idx_t));
      }
      --n;
      nnz -= 1;
      printf("\n");
    }
  }
  idx_t const diff = tt->nnz - nnz;
  tt->nnz = nnz;
  return diff;
}


idx_t tt_remove_empty(
  sptensor_t * const tt)
{
  idx_t dim_sizes[MAX_NMODES];

  idx_t nremoved = 0;

  /* Allocate indmap */
  idx_t const nmodes = tt->nmodes;
  idx_t const nnz = tt->nnz;

  idx_t maxdim = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    maxdim = tt->dims[m] > maxdim ? tt->dims[m] : maxdim;
  }
  idx_t * scounts = (idx_t *) malloc(maxdim * sizeof(idx_t));

  for(idx_t m=0; m < nmodes; ++m) {
    dim_sizes[m] = 0;
    memset(scounts, 0, maxdim * sizeof(idx_t));

    /* Fill in indmap */
    for(idx_t n=0; n < tt->nnz; ++n) {
      /* keep track of #unique slices */
      if(scounts[tt->ind[m][n]] == 0) {
        scounts[tt->ind[m][n]] = 1;
        ++dim_sizes[m];
      }
    }

    /* move on if no remapping is necessary */
    if(dim_sizes[m] == tt->dims[m]) {
      tt->indmap[m] = NULL;
      continue;
    }

    nremoved += tt->dims[m] - dim_sizes[m];

    /* Now scan to remove empty slices */
    idx_t ptr = 0;
    for(idx_t i=0; i < tt->dims[m]; ++i) {
      if(scounts[i] == 1) {
        scounts[i] = ptr++;
      }
    }

    tt->indmap[m] = (idx_t *) malloc(dim_sizes[m] * sizeof(idx_t));

    /* relabel all indices in mode m */
    tt->dims[m] = dim_sizes[m];
    for(idx_t n=0; n < tt->nnz; ++n) {
      idx_t const global = tt->ind[m][n];
      idx_t const local = scounts[global];
      assert(local < dim_sizes[m]);
      tt->indmap[m][local] = global; /* store local -> global mapping */
      tt->ind[m][n] = local;
    }
  }

  free(scounts);
  return nremoved;
}



/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/
sptensor_t * tt_read(
  char const * const ifname)
{
  sptensor_t * tt = tt_read_file(ifname);
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

