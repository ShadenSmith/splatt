
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
  return SPLATT_SUCCESS;
}


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

