
#include "completion.h"

#include "csf.h"

void splatt_tc_als(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws)
{
  /* convert training data to CSF-ALLMODE */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLMODE;
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  splatt_csf * csf = csf_alloc(train, opts);
  assert(csf->ntiles == 1);

  #pragma omp parallel
  {
    for(idx_t e=0; e < ws->max_its; ++e) {
      for(idx_t m=0; m < train->nmodes; ++m) {

        csf_sparsity const * const pt = csf[m].pt;

        /* update each row in parallel */
        /* TODO: use CCP to statically schedule */
        #pragma omp for schedule(dynamic, 16)
        for(idx_t i=0; i < pt->nfibs[0]; ++i) {
          idx_t const fid = (sids == NULL) ? s : sids[s];

        }
      }
    }

  } /* end omp parallel */

  csf_free(csf, opts);
}

