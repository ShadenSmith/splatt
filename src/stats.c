
#include "base.h"
#include "stats.h"
#include "sptensor.h"
#include "ftensor.h"
#include "io.h"

#include <math.h>


/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/
static double __tt_density(
  sptensor_t const * const tt)
{
  double root = pow((double)tt->nnz, 1./(double)tt->nmodes);
  double density = 1.0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    density *= root / (double)tt->dims[m];
  }

  return density;
}

static void __stats_basic(
  sptensor_t const * const tt,
  char const * const ifname)
{
  printf("Tensor information ---------------------------------------------\n");
  printf("FILE=%s\n", ifname);
  printf("DIMS="SS_IDX, tt->dims[0]);
  for(idx_t m=1; m < tt->nmodes; ++m) {
    printf("x" SS_IDX, tt->dims[m]);
  }
  printf(" NNZ=" SS_IDX, tt->nnz);
  printf(" DENSITY=%e" , __tt_density(tt));
  printf("\n\n");
}


static void __stats_hparts(
  sptensor_t * const tt,
  idx_t const mode,
  char const * const pfname)
{
  if(pfname == NULL) {
    fprintf(stderr, "SPLATT ERROR: analysis type requires partition file\n");
    exit(1);
  }

  ftensor_t * ft = ften_alloc(tt);

  /* read partition info */
  idx_t const nvtxs = ft->nfibs[mode];
  idx_t * parts = idx_read(pfname, nvtxs);
  idx_t nparts = 1;
  for(idx_t v=0; v < nvtxs; ++v) {
    if(parts[v] + 1 > nparts) {
      nparts = parts[v] + 1;
    }
  }

  /* pptr marks the size of each partition (in vtxs, not nnz) */
  idx_t * pptr = (idx_t *) calloc(nparts+1, sizeof(idx_t));
  for(idx_t v=0; v < nvtxs; ++v) {
    pptr[1+parts[v]]++;
  }

  /* prefix sum of pptr */
  idx_t saved = pptr[1];
  pptr[1] = 0;
  for(idx_t p=2; p <= nparts; ++p) {
    idx_t tmp = pptr[p];
    pptr[p] = pptr[p-1] + saved;
    saved = tmp;
  }

  idx_t * plookup = (idx_t *) malloc(nvtxs * sizeof(idx_t));
  for(idx_t f=0; f < nvtxs; ++f) {
    idx_t const index = pptr[1+parts[f]]++;
    plookup[index] = f;
  }

  /* get stats on partition sizes */
  idx_t minp = tt->nnz;
  idx_t maxp = 0;
  for(idx_t p=0; p < nparts; ++p) {
    idx_t nnz = 0;
    for(idx_t f=pptr[p]; f < pptr[p+1]; ++f) {
      idx_t const findex = plookup[f];
      nnz += ft->fptr[mode][findex+1] - ft->fptr[mode][findex];
    }
    if(nnz < minp) {
      minp = nnz;
    }
    if(nnz > maxp) {
      maxp = nnz;
    }
  }

  idx_t nhedges = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    nhedges += tt->dims[m];
  }

  printf("Partition information ------------------------------------------\n");
  printf("FILE=%s\n", pfname);
  printf("NVTXS="SS_IDX" NHEDGES="SS_IDX"\n", nvtxs, nhedges);
  printf("NPARTS="SS_IDX" LIGHTEST="SS_IDX" HEAVIEST="SS_IDX" AVG=%0.1f\n",
    nparts, minp, maxp, (val_t)(ft->nnz) / (val_t) nparts);

  idx_t * unique[MAX_NMODES];
  idx_t nunique[MAX_NMODES];
  for(idx_t m=0; m < ft->nmodes; ++m) {
    unique[m] = (idx_t *) malloc(ft->dims[ft->dim_perms[mode][m]]
      * sizeof(idx_t));
  }

  /* now track unique ind info for each partition */
  for(idx_t p=0; p < nparts; ++p) {
    for(idx_t m=0; m < ft->nmodes; ++m) {
      memset(unique[m], 0, ft->dims[ft->dim_perms[mode][m]] * sizeof(idx_t));
      nunique[m] = 0;
    }

    idx_t nnz = 0;
    for(idx_t f=pptr[p]; f < pptr[p+1]; ++f) {
      idx_t const findex = plookup[f];
      nnz += ft->fptr[mode][findex+1] - ft->fptr[mode][findex];
      /* mark unique fids */
      if(unique[1][ft->fids[mode][f]] == 0) {
        ++nunique[1];
        unique[1][ft->fids[mode][f]] = 1;
      }

      for(idx_t j=ft->fptr[mode][findex]; j < ft->fptr[mode][findex+1]; ++j) {
        idx_t const jind = ft->inds[mode][j];
        /* mark unique inds */
        if(unique[2][jind] == 0) {
          ++nunique[2];
          unique[2][jind] = 1;
        }
      }
    }

    printf("nnz: %5lu (%4.1f%%)  ", nnz, 100. * (val_t)nnz / (val_t) tt->nnz);
    printf("I: %5lu (%4.1f%%)  ", nunique[0],
      100. * (val_t)nunique[0] / (val_t) ft->dims[mode]);
    printf("K: %5lu (%4.1f%%)  ", nunique[1],
      100. * (val_t)nunique[1] / (val_t) ft->dims[ft->dim_perms[mode][1]]);
    printf("J: %5lu (%4.1f%%)\n", nunique[2],
      100. * (val_t)nunique[2] / (val_t) ft->dims[ft->dim_perms[mode][2]]);
  }


  for(idx_t m=0; m < ft->nmodes; ++m) {
    free(unique[m]);
  }
  free(parts);
  free(plookup);
  free(pptr);
  ften_free(ft);
}

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void stats_tt(
  sptensor_t * const tt,
  char const * const ifname,
  splatt_stats_type const type,
  idx_t const mode,
  char const * const pfile)
{
  switch(type) {
  case STATS_BASIC:
    __stats_basic(tt, ifname);
    break;
  case STATS_HPARTS:
    __stats_hparts(tt, mode, pfile);
    break;
  default:
    fprintf(stderr, "SPLATT ERROR: analysis type not implemented\n");
    exit(1);
  }
}




