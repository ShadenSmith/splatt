

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "matrix.h"
#include "sort.h"
#include "io.h"
#include "timer.h"

#include <math.h>


/******************************************************************************
 * PRIVATE FUNCTONS
 *****************************************************************************/
static inline int p_same_coord(
  sptensor_t const * const tt,
  idx_t const i,
  idx_t const j)
{
  idx_t const nmodes = tt->nmodes;
  if(nmodes == 3) {
    return (tt->ind[0][i] == tt->ind[0][j]) &&
           (tt->ind[1][i] == tt->ind[1][j]) &&
           (tt->ind[2][i] == tt->ind[2][j]);
  } else {
    for(idx_t m=0; m < nmodes; ++m) {
      if(tt->ind[m][i] != tt->ind[m][j]) {
        return 0;
      }
    }
    return 1;
  }
}




/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

val_t tt_normsq(sptensor_t const * const tt)
{
  val_t norm = 0.0;
  val_t const * const restrict tv = tt->vals;
  for(idx_t n=0; n < tt->nnz; ++n) {
    norm += tv[n] * tv[n];
  }
  return norm;
}


double tt_density(
  sptensor_t const * const tt)
{
  double root = pow((double)tt->nnz, 1./(double)tt->nmodes);
  double density = 1.0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    density *= root / (double)tt->dims[m];
  }

  return density;
}


idx_t * tt_get_slices(
  sptensor_t const * const tt,
  idx_t const m,
  idx_t * nunique)
{
  /* get maximum number of unique slices */
  idx_t minidx = tt->dims[m];
  idx_t maxidx = 0;

  idx_t const nnz = tt->nnz;
  idx_t const * const inds = tt->ind[m];

  /* find maximum number of uniques */
  for(idx_t n=0; n < nnz; ++n) {
    minidx = SS_MIN(minidx, inds[n]);
    maxidx = SS_MAX(maxidx, inds[n]);
  }
  /* +1 because maxidx is inclusive, not exclusive */
  idx_t const maxrange = 1 + maxidx - minidx;

  /* mark slices which are present and count uniques */
  idx_t * slice_mkrs = (idx_t *) calloc(maxrange, sizeof(idx_t));
  idx_t found = 0;
  for(idx_t n=0; n < nnz; ++n) {
    assert(inds[n] >= minidx);
    idx_t const idx = inds[n] - minidx;
    if(slice_mkrs[idx] == 0) {
      slice_mkrs[idx] = 1;
      ++found;
    }
  }
  *nunique = found;

  /* now copy unique slices */
  idx_t * slices = (idx_t *) splatt_malloc(found * sizeof(idx_t));
  idx_t ptr = 0;
  for(idx_t i=0; i < maxrange; ++i) {
    if(slice_mkrs[i] == 1) {
      slices[ptr++] = i + minidx;
    }
  }

  free(slice_mkrs);

  return slices;
}


idx_t tt_remove_dups(
  sptensor_t * const tt)
{
  tt_sort(tt, 0, NULL);

  idx_t const nmodes = tt->nmodes;

  idx_t newnnz = 0;
  for(idx_t nnz = 1; nnz < tt->nnz; ++nnz) {
    /* if the two nnz are the same, average them */
    if(p_same_coord(tt, newnnz, nnz)) {
      tt->vals[newnnz] += tt->vals[nnz];
    } else {
      /* new another nnz */
      ++newnnz;
      for(idx_t m=0; m < nmodes; ++m) {
        tt->ind[m][newnnz] = tt->ind[m][nnz];
      }
      tt->vals[newnnz] = tt->vals[nnz];
    }
  }
  ++newnnz;

  idx_t const diff = tt->nnz - newnnz;
  tt->nnz = newnnz;
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
  /* slice counts */
  idx_t * scounts = (idx_t *) splatt_malloc(maxdim * sizeof(idx_t));

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

    tt->indmap[m] = (idx_t *) splatt_malloc(dim_sizes[m] * sizeof(idx_t));

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
  int l = strlen(ifname);
  sptensor_t * tt;
  if (l > 4 && strcmp(ifname + l - 4, ".bin") == 0) {
    tt = tt_read_binary_file(ifname);
  }
  else {
    tt = tt_read_file(ifname);
  }
  return tt;
}


sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes)
{
  sptensor_t * tt = (sptensor_t*) splatt_malloc(sizeof(sptensor_t));
  tt->tiled = SPLATT_NOTILE;

  tt->nnz = nnz;
  tt->vals = (val_t*) splatt_malloc(nnz * sizeof(val_t));

  tt->nmodes = nmodes;
  tt->type = (nmodes == 3) ? SPLATT_3MODE : SPLATT_NMODE;

  tt->dims = (idx_t*) splatt_malloc(nmodes * sizeof(idx_t));
  tt->ind = (idx_t**) splatt_malloc(nmodes * sizeof(idx_t*));
  for(idx_t m=0; m < nmodes; ++m) {
    tt->ind[m] = (idx_t*) splatt_malloc(nnz * sizeof(idx_t));
    tt->indmap[m] = NULL;
  }

  return tt;
}


void tt_fill(
  sptensor_t * const tt,
  idx_t const nnz,
  idx_t const nmodes,
  idx_t ** const inds,
  val_t * const vals)
{
  tt->tiled = SPLATT_NOTILE;
  tt->nnz = nnz;
  tt->vals = vals;
  tt->ind = inds;

  tt->nmodes = nmodes;
  tt->type = (nmodes == 3) ? SPLATT_3MODE : SPLATT_NMODE;

  tt->dims = (idx_t*) splatt_malloc(nmodes * sizeof(idx_t));
  for(idx_t m=0; m < nmodes; ++m) {
    tt->indmap[m] = NULL;

    tt->dims[m] = inds[m][0];
    for(idx_t i=1; i < nnz; ++i) {
      tt->dims[m] = SS_MAX(tt->dims[m], inds[m][i]);
    }
  }
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

