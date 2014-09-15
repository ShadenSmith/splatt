
#include "ftensor.h"
#include "matrix.h"

void ttkrao_ftensor(
  ftensor_t const * restrict const tt,
  idx_t const mode,
  matrix_t const * restrict const A,
  matrix_t const * restrict const B,
  matrix_t const * restrict const M1)
{

  idx_t const * restrict const partptr = tt->slptr[mode-1];
  idx_t const * restrict const slptr   = tt->slptr[mode-1];
  idx_t const * restrict const fptr    = tt->fptr[mode-1];
  idx_t const * restrict const fid     = tt->fid[mode-1];
  idx_t const * restrict const ind     = tt->ind[mode-1];
  val_t const * restrict const vals    = tt->vals[mode-1];

  /* direction determines which matrix we're factoring out -- swap pointers
   * if direction==1 */
  val_t const * restrict const vals_a =
    (tt->direction[mode-1] == 0 ? A->vals : B->vals);
  val_t const * restrict const vals_b =
    (tt->direction[mode-1] == 0 ? B->vals : A->vals);

  idx_t const I = M1->I;
  idx_t const J = M1->J;
  idx_t const rank = J;
  val_t * restrict const m1_vals = M1->vals;
  memset(m1_vals, 0, I * J * sizeof(val_t));

  val_t * restrict const accumF = (val_t*) malloc(rank * sizeof(val_t));

  for(idx_t i=0; i < I; ++i) {
    for(idx_t s=slptr[i]; s < slptr[i+1]; ++s) {
      for(idx_t f=fptr[s]; f < fptr[s+1]; ++f) {
        idx_t const jjfirst = fptr[f];
        idx_t const rowfirst = ind[jjfirst];
        val_t const vfirst = vals[jjfirst];
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] = vfirst * vals_a[r + (rowfirst * J)];
        }
        /* inner product of rest of fiber with column of A */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals_a[jj];
          idx_t const row = ind[jj];
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] += v * vals_a[r + (row * J)];
          }
        }
        /* update row of M1 with accumF */
        idx_t const row = fid[f];
        for(idx_t r=0; r < rank; ++r) {
          m1_vals[r + (i * J)] += accumF[r] * vals_b[r + (row * J)];
        }
      }
    }
  }
  free(accumF);
}

ftensor_t * alloc_ftensor(
  idx_t const dims[3],
  idx_t const nnz)
{
  ftensor_t *tt = (ftensor_t*) malloc(sizeof(ftensor_t));
  for(idx_t d=0; d < 3; ++d) {
    tt->ind[d]  = (idx_t*) malloc(nnz * sizeof(idx_t));
    tt->vals[d] = (val_t*) malloc(nnz * sizeof(val_t));
  }
  return tt;
}

void free_ftensor(
  ftensor_t *tt)
{

}
