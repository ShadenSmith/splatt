
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
