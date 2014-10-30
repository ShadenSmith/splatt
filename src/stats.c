
#include "base.h"
#include "sptensor.h"
#include "io.h"

#include <math.h>

void tt_stats(
  char * const fname)
{
  sptensor_t * tt = tt_read(fname);

  printf("FILE=%s\n", fname);
  printf("DIMS="SS_IDX, tt->dims[0]);
  for(idx_t m=1; m < NMODES; ++m) {
    printf("x" SS_IDX, tt->dims[m]);
  }
  printf(" NNZ= " SS_IDX, tt->nnz);

  double root = pow((double)tt->nnz, 1./(double)NMODES);
  double density = 1.0;
  for(idx_t m=0; m < NMODES; ++m) {
    density *= root / (double)tt->dims[m];
  }
  printf(" DENSITY= %e" , density);
  printf("\n");
}
