
#include "base.h"
#include "sptensor.h"
#include "io.h"


#include "ftensor.h"

#include <math.h>

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

void tt_stats(
  char * const fname)
{
  sptensor_t * tt = tt_read(fname);

  printf("FILE=%s\n", fname);
  printf("DIMS="SS_IDX, tt->dims[0]);
  for(idx_t m=1; m < tt->nmodes; ++m) {
    printf("x" SS_IDX, tt->dims[m]);
  }
  printf(" NNZ=" SS_IDX, tt->nnz);
  printf(" DENSITY= %e" , __tt_density(tt));
  printf("\n");

  ftensor_t * ft = ften_alloc(tt);
  ften_free(ft);

  tt_free(tt);
}

