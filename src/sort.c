
#include "sort.h"
#include "timer.h"

#define MIN_QUICKSORT_SIZE 8

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
    } else if(tt->ind[cmplt[m]][j] < tt->ind[cmplt[m]][i]) {
      return 1;
    }
  }
  return 0;
}

static inline int __ttqcmp(
  sptensor_t const * const tt,
  idx_t const * const cmplt,
  idx_t const i,
  idx_t const j[MAX_NMODES])
{
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(tt->ind[cmplt[m]][i] < j[cmplt[m]]) {
      return -1;
    } else if(j[cmplt[m]] < tt->ind[cmplt[m]][i]) {
      return 1;
    }
  }
  return 0;
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

static void __tt_insertionsort(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  idx_t * ind;
  val_t * const vals = tt->vals;
  idx_t const nmodes = tt->nmodes;

  val_t vbuf;
  idx_t ibuf;

  for(size_t i=start+1; i < end; ++i) {
    ssize_t j = i;
    while (j > 0 && __ttcmp(tt, cmplt, i, j-1) < 0) {
      --j;
    }

    vbuf = vals[i];

    /* shift all data */
    memmove(vals+j+1, vals+j, (i-j)*sizeof(val_t));
    vals[j] = vbuf;
    for(idx_t m=0; m < nmodes; ++m) {
      ind = tt->ind[m];
      ibuf = ind[i];
      memmove(ind+j+1, ind+j, (i-j)*sizeof(idx_t));
      ind[j] = ibuf;
    }
  }
}

static void __tt_quicksort(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  idx_t vmid;
  idx_t imid[MAX_NMODES];

  idx_t * ind;
  val_t * const vals = tt->vals;
  idx_t const nmodes = tt->nmodes;

  if((end-start) <= MIN_QUICKSORT_SIZE) {
    __tt_insertionsort(tt, cmplt, start, end);
  } else {
    size_t i = start+1;
    size_t j = end-1;
    size_t k = start + ((end - start) / 2);

    /* grab pivot */
    vmid = vals[k];
    vals[k] = vals[start];
    for(idx_t m=0; m < nmodes; ++m) {
      ind = tt->ind[m];
      imid[m] = ind[k];
      ind[k] = ind[start];
    }

    while(i < j) {
      /* if tt[i] > mid  -> tt[i] is on wrong side */
      if(__ttqcmp(tt,cmplt,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(__ttqcmp(tt,cmplt,j,imid) < 1) {
          __ttswap(tt,i,j);
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(__ttqcmp(tt,cmplt,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(__ttqcmp(tt,cmplt,i,imid) == 1) {
      --i;
    }
    vals[start] = vals[i];
    vals[i] = vmid;
    for(idx_t m=0; m < nmodes; ++m) {
      ind = tt->ind[m];
      ind[start] = ind[i];
      ind[i] = imid[m];
    }

    if(i > start + 1) {
      __tt_quicksort(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      __tt_quicksort(tt, cmplt, i, end);
    }
  }
}

void tt_sort(
  sptensor_t * const tt,
  idx_t const mode)
{
  sp_timer_t timer;

  idx_t cmplt[MAX_NMODES];
  cmplt[0] = mode;
  printf("cmplt: " SS_IDX " ", cmplt[0]);
  for(idx_t m=1; m < tt->nmodes; ++m) {
    cmplt[m] = (mode + m) % tt->nmodes;
    printf(SS_IDX " ", cmplt[m]);
  }
  printf("\n");

  timer_reset(&timer);
  timer_start(&timer);
  __tt_quicksort(tt, cmplt, 0, tt->nnz);
  //__tt_insertionsort(tt, cmplt, 0, tt->nnz);
  timer_stop(&timer);

  /* validate sort */
  for(idx_t n=0; n < tt->nnz - 1; ++n) {
    if(__ttcmp(tt, cmplt, n, n+1) == 1) {
      printf("fail (");
      for(idx_t m=0; m < tt->nmodes; ++m) {
        printf(SS_IDX " ", tt->ind[m][n]);
      }
      printf(")\n     (");
      for(idx_t m=0; m < tt->nmodes; ++m) {
        printf(SS_IDX " ", tt->ind[m][n+1]);
      }
      printf(")\n");
      break;
    }
  }

  printf("SORT: %0.3fs\n", timer.seconds);
}

