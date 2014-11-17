
#include "sort.h"

#define MIN_QUICKSORT_SIZE 16

void insertion_sort(
  idx_t * const a,
  idx_t const n)
{
  for(size_t i=1; i < n; ++i) {
    idx_t b = a[i];
    ssize_t j = i;
    while (j > 0 &&  a[j-1] > b) {
      --j;
    }
    memmove(a+(j+1), a+j, sizeof(idx_t)*(i-j));
    a[j] = b;
  }
}

void quicksort(
  idx_t * const a,
  idx_t const n)
{
  if(n < MIN_QUICKSORT_SIZE) {
    insertion_sort(a, n);
  } else {
    size_t i = 1;
    size_t j = n-1;
    size_t k = n >> 1;
    idx_t mid = a[k];
    a[k] = a[0];
    while(i < j) {
      if(a[i] > mid) { /* a[i] is on the wrong side */
        if(a[j] <= mid) { /* swap a[i] and a[j] */
          idx_t tmp = a[i];
          a[i] = a[j];
          a[j] = tmp;
          ++i;
        }
        --j;
      } else {
        if(a[j] > mid) { /* a[j] is on the right side */
          --j;
        }
        ++i;
      }
    }

    if(a[i] > mid) {
      --i;
    }
    a[0] = a[i];
    a[i] = mid;

    if(i > 1) {
      quicksort(a,i);
    }
    ++i; /* skip the pivot element */
    if(n-i > 1) {
      quicksort(a+i, n-i);
    }
  }
}

static inline int __ttcmp(
  sptensor_t const * const tt,
  idx_t const * const cmplt,
  idx_t const i,
  idx_t const j)
{
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(tt->ind[cmplt[m]][i] < tt->ind[cmplt[m]][j]) {
      return -1;
    }
  }
  return 1;
}

static inline void __ttswap(
  sptensor_t * const tt,
  idx_t const i,
  idx_t const j)
{
  val_t vtmp = tt->vals[i];
  tt->vals[i] = tt->vals[j];
  tt->vals[j] = vtmp;

  idx_t itmp;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    itmp = tt->ind[m][i];
    tt->ind[m][i] = tt->ind[m][j];
    tt->ind[m][j] = itmp;
  }
}

static void __ttqsort(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{

}

void tt_sort(
  sptensor_t * const tt,
  idx_t const mode)
{
  idx_t cmplt[MAX_NMODES];
  cmplt[0] = mode;
  printf("cmplt: " SS_IDX " ", cmplt[0]);
  for(idx_t m=1; m < tt->nmodes; ++m) {
    cmplt[m] = (mode + m) % tt->nmodes;
    printf(SS_IDX " ", cmplt[m]);
  }
  printf("\n");

  __ttqsort(tt, cmplt, 0, tt->nnz);
}

