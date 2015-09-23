
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "cpd.h"
#include "matrix.h"
#include "mttkrp.h"
#include "sptensor.h"
#include "stats.h"
#include "timer.h"
#include "thd_info.h"
#include "tile.h"
#include "io.h"
#include "util.h"


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
int splatt_tucker_als(
    splatt_idx_t const * const nfactors,
    splatt_idx_t const nmodes,
    splatt_csf_t const * const tensors,
    double const * const options,
    splatt_tucker_t * factored)
{
  matrix_t * mats[MAX_NMODES+1];

  idx_t const nthreads = (idx_t) options[SPLATT_OPTION_NTHREADS];

  /* fill in factored */
  idx_t maxdim = 0;
  idx_t maxcols = 0;
  idx_t mincols = nfactors[0];
  idx_t csize = 1;
  factored->nmodes = nmodes;
  for(idx_t m=0; m < nmodes; ++m) {
    factored->rank[m] = nfactors[m];
    mats[m] = mat_rand(tensors[0].dims[m], nfactors[m]);
    factored->factors[m] = mats[m]->vals;

    csize *= nfactors[m];
    maxdim = SS_MAX(maxdim, tensors[0].dims[m]);
    maxcols = SS_MAX(maxcols, nfactors[m]);
    mincols = SS_MIN(mincols, nfactors[m]);
  }
  factored->core = (val_t *) calloc(csize, sizeof(val_t));

  val_t * gten = (val_t *) malloc(maxdim * (csize / mincols) * sizeof(val_t));

  /* thread structures */
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 1,
    (maxcols * sizeof(val_t)) + 64);

  ftensor_t const * const ft = tensors + 0;
  ttm_splatt(ft, mats, gten, 0, thds, nthreads);

  printf("G1:\n");
  for(idx_t s=0; s < ft->nslcs; ++s) {
    for(idx_t f=0; f < nfactors[1] * nfactors[2]; ++f) {
      printf("%0.2e ", gten[f + (s*nfactors[1]*nfactors[2])]);
    }
    printf("\n----\n");
  }

  free(gten);
  for(idx_t m=0; m < nmodes; ++m) {
    free(mats[m]);
  }
  return SPLATT_SUCCESS;
}


void splatt_free_tucker(
    splatt_tucker_t * factored)
{
  free(factored->core);
  for(idx_t m=0; m < factored->nmodes; ++m) {
    free(factored->factors[m]);
  }
}

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

