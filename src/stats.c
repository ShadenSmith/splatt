
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
  FILE * pfile;
  idx_t * parts;
  idx_t nparts;
  if(pfname == NULL) {
    fprintf(stderr, "SPLATT ERROR: analysis type requires partition file\n");
    exit(1);
  }
  if((pfile = fopen(pfname, "r")) == NULL) {
    fprintf(stderr, "SPLATT ERROR: unable to open '%s'\n", pfname);
    exit(1);
  }

  ftensor_t * ft = ften_alloc(tt);

  /* read partition info */
  idx_t const nvtxs = ft->nfibs[mode];
  parts = (idx_t *) malloc(nvtxs * sizeof(idx_t));
  idx_t ret;
  nparts = 1;
  for(idx_t v=0; v < nvtxs; ++v) {
    if((ret = fscanf(pfile, SS_IDX, &(parts[v]))) == 0) {
      fprintf(stderr, "SPLATT ERROR: not enough vertices in file\n");
      exit(1);
    }
    if(parts[v] + 1 > nparts) {
      nparts = parts[v] + 1;
    }
  }
  fclose(pfile);

  idx_t * psizes = (idx_t *) malloc((nparts+1) * sizeof(idx_t));
  memset(psizes, 0, (nparts+1) * sizeof(idx_t));
  for(idx_t v=0; v < nvtxs; ++v) {
    psizes[1+parts[v]]++;
  }

  /* prefix sum of psizes */
  idx_t saved = psizes[1];
  psizes[1] = 0;
  for(idx_t p=2; p < nparts; ++p) {
    idx_t tmp = psizes[p];
    psizes[p] = psizes[p-1] + saved;
    saved = tmp;
  }

  idx_t * plookup = (idx_t *) malloc(nvtxs * sizeof(idx_t));
  for(idx_t f=0; f < nvtxs; ++f) {
    idx_t const index = psizes[1+parts[f]]++;
    plookup[index] = f;
  }
  psizes[nparts] = nvtxs;

  /* get stats on partition sizes */
  idx_t minp = tt->nnz;
  idx_t maxp = 0;

  idx_t * unique[MAX_NMODES];
  idx_t nunique[MAX_NMODES];
  for(idx_t m=0; m < ft->nmodes; ++m) {
    unique[m] = (idx_t *) malloc(ft->dims[ft->dim_perms[mode][m]]
      * sizeof(idx_t));
  }

  /* now lets track unique ind info for each partition */
  for(idx_t p=0; p < nparts; ++p) {
    for(idx_t m=0; m < ft->nmodes; ++m) {
      memset(unique[m], 0, ft->dims[ft->dim_perms[mode][m]] * sizeof(idx_t));
      nunique[m] = 0;
    }

    idx_t nnz = 0;
    for(idx_t f=psizes[p]; f < psizes[p+1]; ++f) {
      nnz += ft->fptr[mode][f+1] - ft->fptr[mode][f];
      /* mark unique fids */
      if(unique[1][ft->fids[mode][f]] == 0) {
        ++nunique[1];
        unique[1][ft->fids[mode][f]] = 1;
      }

      for(idx_t j=ft->fptr[mode][f]; j < ft->fptr[mode][f+1]; ++j) {
        idx_t const jind = ft->inds[mode][j];
        /* mark unique inds */
        if(unique[2][jind] == 0) {
          ++nunique[2];
          unique[2][jind] = 1;
        }
      }
    }

    printf("nnz: %5lu (%2.1f%%)  ", nnz, 100. * (val_t)nnz / (val_t) tt->nnz);
    printf("I: %5lu (%2.1f%%)  ", nunique[0],
      100. * (val_t)nunique[0] / (val_t) ft->dims[mode]);
    printf("J: %5lu (%2.1f%%)  ", nunique[1],
      100. * (val_t)nunique[1] / (val_t) ft->dims[ft->dim_perms[mode][1]]);
    printf("K: %5lu (%2.1f%%)\n", nunique[2],
      100. * (val_t)nunique[2] / (val_t) ft->dims[ft->dim_perms[mode][2]]);

    if(nnz < minp) {
      minp = nnz;
    }
    if(nnz > maxp) {
      maxp = nnz;
    }
  }

  printf("Partition information ------------------------------------------\n");
  printf("NPARTS=" SS_IDX " LIGHTEST=" SS_IDX " HEAVIEST=" SS_IDX " AVG=%0.1f\n",
    nparts, minp, maxp, (val_t)(ft->nnz) / (val_t) nparts);


  for(idx_t m=0; m < ft->nmodes; ++m) {
    free(unique[m]);
  }
  free(parts);
  free(plookup);
  free(psizes);
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




