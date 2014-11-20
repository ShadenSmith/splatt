
#include "base.h"
#include "sptensor.h"
#include "io.h"
#include "sort.h"

#include "matrix.h"

#include <math.h>

void tt_stats(
  char * const fname)
{
  sptensor_t * tt = tt_read(fname);

  double root = pow((double)tt->nnz, 1./(double)tt->nmodes);
  double density = 1.0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    density *= root / (double)tt->dims[m];
  }

  printf("FILE=%s\n", fname);
  printf("DIMS="SS_IDX, tt->dims[0]);
  for(idx_t m=1; m < tt->nmodes; ++m) {
    printf("x" SS_IDX, tt->dims[m]);
  }
  printf(" NNZ=" SS_IDX, tt->nnz);
  printf(" DENSITY= %e" , density);
  printf("\n");

  tt_free(tt);
}

