
#include "bench.h"
#include "mttkrp.h"

void bench_splatt(
  sptensor_t * const tt,
  matrix_t ** mats,
  idx_t const niters,
  idx_t const nthreads,
  int const scale)
{
  ftensor_t * ft = ften_alloc(tt);
  for(idx_t i=0; i < niters; ++i) {
    for(idx_t m=0; m < tt->nmodes; ++m) {
      mttkrp_splatt(ft, mats, m);
    }
  }
  ften_free(ft);
}

void bench_giga(
  sptensor_t * const tt,
  matrix_t ** mats,
  idx_t const niters,
  idx_t const nthreads,
  int const scale)
{
  printf("bench giga!\n");
  val_t * scratch = (val_t *) malloc(tt->nnz * sizeof(val_t));
  spmatrix_t * unfolds[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    unfolds[m] = tt_unfold(tt, m);
  }

  for(idx_t i=0; i < niters; ++i) {
    for(idx_t m=0; m < tt->nmodes; ++m) {
      mttkrp_giga(unfolds[m], mats, m, scratch);
    }
  }

  for(idx_t m=0; m < tt->nmodes; ++m) {
    spmat_free(unfolds[m]);
  }
  free(scratch);
}

void bench_ttbox(
  sptensor_t * const tt,
  matrix_t ** mats,
  idx_t const niters,
  idx_t const nthreads,
  int const scale)
{
  printf("bench ttbox!\n");
  val_t * scratch = (val_t *) malloc(tt->nnz * sizeof(val_t));
  for(idx_t i=0; i < niters; ++i) {
    for(idx_t m=0; m < tt->nmodes; ++m) {
      mttkrp_ttbox(tt, mats, m, scratch);
    }
  }
  free(scratch);
}


