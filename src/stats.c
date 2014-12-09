
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
  char const * const ifname,
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

  sptensor_t * tt = tt_read(ifname);
  ftensor_t * ft = ften_alloc(tt);
  tt_free(tt);

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

  idx_t const max_unique = ft->dims[ft->dim_perms[mode][2]];
  idx_t * unique = (idx_t *) malloc(max_unique * sizeof(idx_t));

  printf("nnz: " SS_IDX " unique: " SS_IDX " ratio: " SS_VAL "\n",
    ft->nnz, max_unique, ((val_t)ft->nnz / (val_t)max_unique));

  /* now get per-partition stats */
  for(idx_t p=0; p < nparts; ++p) {
    memset(unique, 0, max_unique * sizeof(idx_t));
    idx_t nnz = 0;
    idx_t nunique = 0;
    for(idx_t f=psizes[p]; f < psizes[p+1]; ++f) {
      nnz += ft->fptr[mode][f+1] - ft->fptr[mode][f];

      for(idx_t jj=ft->fptr[mode][f]; jj < ft->fptr[mode][f+1]; ++jj) {
        idx_t const ind = ft->inds[mode][jj];
        if(unique[ind] == 0) {
          ++nunique;
          unique[ind] = 1;
        }
      }
    }
    printf("nnz: " SS_IDX " unique: " SS_IDX " ratio: " SS_VAL "\n",
      nnz, nunique, ((val_t)nnz / (val_t)nunique));
  }

  free(unique);
  free(plookup);
  free(psizes);
  free(parts);
  ften_free(ft);
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void stats_tt(
  sptensor_t const * const tt,
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
    __stats_hparts(ifname, mode, pfile);
    break;
  default:
    fprintf(stderr, "SPLATT ERROR: analysis type not implemented\n");
    exit(1);
  }
}




