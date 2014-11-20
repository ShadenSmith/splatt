

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sort.h"
#include "timer.h"



/******************************************************************************
 * DEFINES
 *****************************************************************************/
#define MIN_QUICKSORT_SIZE 8



/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/
static inline int __ttqcmp3(
  idx_t const * const ind0,
  idx_t const * const ind1,
  idx_t const * const ind2,
  idx_t const i,
  idx_t const j[3])
{
  if(ind0[i] < j[0]) {
    return -1;
  } else if(j[0] < ind0[i]) {
    return 1;
  }
  if(ind1[i] < j[1]) {
    return -1;
  } else if(j[1] < ind1[i]) {
    return 1;
  }
  if(ind2[i] < j[2]) {
    return -1;
  } else if(j[2] < ind2[i]) {
    return 1;
  }

  return 0;
}

static inline int __ttcmp3(
  idx_t const * const ind0,
  idx_t const * const ind1,
  idx_t const * const ind2,
  idx_t const i,
  idx_t const j)
{
  if(ind0[i] < ind0[j]) {
    return -1;
  } else if(ind0[j] < ind0[i]) {
    return 1;
  }
  if(ind1[i] < ind1[j]) {
    return -1;
  } else if(ind1[j] < ind1[i]) {
    return 1;
  }
  if(ind2[i] < ind2[j]) {
    return -1;
  } else if(ind2[j] < ind2[i]) {
    return 1;
  }
  return 0;
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

static void __tt_insertionsort3(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  idx_t * const ind0 = tt->ind[cmplt[0]];
  idx_t * const ind1 = tt->ind[cmplt[1]];
  idx_t * const ind2 = tt->ind[cmplt[2]];
  val_t * const vals = tt->vals;

  val_t vbuf;
  idx_t ibuf;

  for(size_t i=start+1; i < end; ++i) {
    ssize_t j = i;
    while (j > 0 && __ttcmp3(ind0, ind1, ind2, i, j-1) < 0) {
      --j;
    }

    vbuf = vals[i];

    /* shift all data */
    memmove(vals+j+1, vals+j, (i-j)*sizeof(val_t));
    vals[j] = vbuf;
    ibuf = ind0[i];
    memmove(ind0+j+1, ind0+j, (i-j)*sizeof(idx_t));
    ind0[j] = ibuf;
    ibuf = ind1[i];
    memmove(ind1+j+1, ind1+j, (i-j)*sizeof(idx_t));
    ind1[j] = ibuf;
    ibuf = ind2[i];
    memmove(ind2+j+1, ind2+j, (i-j)*sizeof(idx_t));
    ind2[j] = ibuf;
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

static void __tt_quicksort3(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  idx_t vmid;
  idx_t imid[3];

  idx_t * const ind0 = tt->ind[cmplt[0]];
  idx_t * const ind1 = tt->ind[cmplt[1]];
  idx_t * const ind2 = tt->ind[cmplt[2]];
  val_t * const vals = tt->vals;

  if((end-start) <= MIN_QUICKSORT_SIZE) {
    __tt_insertionsort3(tt, cmplt, start, end);
  } else {
    size_t i = start+1;
    size_t j = end-1;
    size_t k = start + ((end - start) / 2);

    /* grab pivot */
    vmid = vals[k];
    vals[k] = vals[start];
    imid[0] = ind0[k];
    imid[1] = ind1[k];
    imid[2] = ind2[k];
    ind0[k] = ind0[start];
    ind1[k] = ind1[start];
    ind2[k] = ind2[start];

    while(i < j) {
      /* if tt[i] > mid  -> tt[i] is on wrong side */
      if(__ttqcmp3(ind0,ind1,ind2,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(__ttqcmp3(ind0,ind1,ind2,j,imid) < 1) {
          val_t vtmp = vals[i];
          vals[i] = vals[j];
          vals[j] = vtmp;
          idx_t itmp = ind0[i];
          ind0[i] = ind0[j];
          ind0[j] = itmp;
          itmp = ind1[i];
          ind1[i] = ind1[j];
          ind1[j] = itmp;
          itmp = ind2[i];
          ind2[i] = ind2[j];
          ind2[j] = itmp;
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(__ttqcmp3(ind0,ind1,ind2,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(__ttqcmp3(ind0,ind1,ind2,i,imid) == 1) {
      --i;
    }
    vals[start] = vals[i];
    vals[i] = vmid;
    ind0[start] = ind0[i];
    ind1[start] = ind1[i];
    ind2[start] = ind2[i];
    ind0[i] = imid[0];
    ind1[i] = imid[1];
    ind2[i] = imid[2];

    if(i > start + 1) {
      __tt_quicksort3(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      __tt_quicksort3(tt, cmplt, i, end);
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


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_sort(
  sptensor_t * const tt,
  idx_t const mode,
  idx_t * dim_perm)
{
  sp_timer_t timer;

  idx_t * cmplt;
  if(dim_perm == NULL) {
    cmplt = (idx_t*) malloc(tt->nmodes * sizeof(idx_t));
    cmplt[0] = mode;
    for(idx_t m=1; m < tt->nmodes; ++m) {
      cmplt[m] = (mode + m) % tt->nmodes;
    }
  } else {
    cmplt = dim_perm;
  }

  timer_reset(&timer);
  timer_start(&timer);
  switch(tt->type) {
  case SPLATT_NMODE:
    __tt_quicksort(tt, cmplt, 0, tt->nnz);
    break;
  case SPLATT_3MODE:
    __tt_quicksort3(tt, cmplt, 0, tt->nnz);
    break;
  }
  timer_stop(&timer);

  printf("SORT: %0.3fs\n", timer.seconds);

  if(dim_perm == NULL) {
    free(cmplt);
  }
}

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

