
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
  char const * const ifname)
{
  sptensor_t * tt = tt_read(ifname);

  printf("FILE=%s\n", ifname);
  printf("DIMS="SS_IDX, tt->dims[0]);
  for(idx_t m=1; m < tt->nmodes; ++m) {
    printf("x" SS_IDX, tt->dims[m]);
  }
  printf(" NNZ=" SS_IDX, tt->nnz);
  printf(" DENSITY= %e" , __tt_density(tt));
  printf("\n");

  tt_free(tt);
}

static void __stats_hparts(
  char const * const ifname,
  idx_t const mode,
  char const * const pfname)
{
  if(pfname == NULL) {
    fprintf(stderr, "SPLATT ERROR: analysis type requires partition file\n");
    exit(1);
  }

}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_stats(
  char const * const ifname,
  splatt_stats_type const type,
  idx_t const mode,
  char const * const pfile)
{
  switch(type) {
  case STATS_BASIC:
    __stats_basic(ifname);
    break;
  case STATS_HPARTS:
    __stats_hparts(ifname, mode, pfile);
    break;
  default:
    fprintf(stderr, "SPLATT ERROR: analysis type not implemented\n");
    exit(1);
  }
}

