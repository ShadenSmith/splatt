

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sort.h"
#include "timer.h"
#include "io.h"

#include <omp.h>


/******************************************************************************
 * DEFINES
 *****************************************************************************/
/* switch to insertion sort past this point */
#define MIN_QUICKSORT_SIZE 8

/* don't bother spawning threads for small sorts */
#define SMALL_SORT_SIZE 1000


/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/

/**
* @brief Compares ind*[i] and ind*[j] for two-mode tensors.
*
* @param ind0 The primary mode to compare. Defer tie-breaks to ind1.
* @param ind1 The secondary mode to compare.
* @param i The index into ind*.
* @param j The second index into ind*.
*
* @return Returns -1 if ind[i] < ind[j], 1 if ind[i] > ind[j], and 0 if they
*         are equal.
*/
static inline int p_ttcmp2(
  idx_t const * const ind0,
  idx_t const * const ind1,
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
  return 0;
}

/**
* @brief Compares ind*[i] and j[*] for two-mode tensors.
*
* @param ind0 The primary mode to compare. Defer tie-breaks to ind1.
* @param ind1 The secondary mode to compare.
* @param i The index into ind*[]
* @param j[2] The indices we are comparing i against.
*
* @return Returns -1 if ind[i] < j, 1 if ind[i] > j, and 0 if they are equal.
*/
static inline int p_ttqcmp2(
  idx_t const * const ind0,
  idx_t const * const ind1,
  idx_t const i,
  idx_t const j[2])
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

  return 0;
}

/**
* @brief Perform insertion sort on a 2-mode tensor between start and end,
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_insertionsort2(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  idx_t * const ind0 = tt->ind[cmplt[0]];
  idx_t * const ind1 = tt->ind[cmplt[1]];
  val_t * const vals = tt->vals;

  val_t vbuf;
  idx_t ibuf;

  for(size_t i=start+1; i < end; ++i) {
    size_t j = i;
    while (j > start && p_ttcmp2(ind0, ind1, i, j-1) < 0) {
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
  }
}

/**
* @brief Perform quicksort on a 2-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_quicksort2(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  val_t vmid;
  idx_t imid[2];

  idx_t * const ind0 = tt->ind[cmplt[0]];
  idx_t * const ind1 = tt->ind[cmplt[1]];
  val_t * const vals = tt->vals;

  if((end-start) <= MIN_QUICKSORT_SIZE) {
    p_tt_insertionsort2(tt, cmplt, start, end);
  } else {
    size_t i = start+1;
    size_t j = end-1;
    size_t k = start + ((end - start) / 2);

    /* grab pivot */
    vmid = vals[k];
    vals[k] = vals[start];
    imid[0] = ind0[k];
    imid[1] = ind1[k];
    ind0[k] = ind0[start];
    ind1[k] = ind1[start];

    while(i < j) {
      /* if tt[i] > mid  -> tt[i] is on wrong side */
      if(p_ttqcmp2(ind0,ind1,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(p_ttqcmp2(ind0,ind1,j,imid) < 1) {
          val_t vtmp = vals[i];
          vals[i] = vals[j];
          vals[j] = vtmp;
          idx_t itmp = ind0[i];
          ind0[i] = ind0[j];
          ind0[j] = itmp;
          itmp = ind1[i];
          ind1[i] = ind1[j];
          ind1[j] = itmp;
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(p_ttqcmp2(ind0,ind1,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(p_ttqcmp2(ind0,ind1,i,imid) == 1) {
      --i;
    }
    vals[start] = vals[i];
    vals[i] = vmid;
    ind0[start] = ind0[i];
    ind1[start] = ind1[i];
    ind0[i] = imid[0];
    ind1[i] = imid[1];

    if(i > start + 1) {
      p_tt_quicksort2(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      p_tt_quicksort2(tt, cmplt, i, end);
    }
  }
}

/**
* @brief Compares ind*[i] and j[*] for three-mode tensors.
*
* @param ind0 The primary mode to compare. Defer tie-breaks to ind1.
* @param ind1 The secondary mode to compare. Defer tie-breaks to ind2.
* @param ind2 The final tie-breaking mode.
* @param i The index into ind*[]
* @param j[3] The indices we are comparing i against.
*
* @return Returns -1 if ind[i] < j, 1 if ind[i] > j, and 0 if they are equal.
*/
static inline int p_ttqcmp3(
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


/**
* @brief Compares ind*[i] and ind*[j] for three-mode tensors.
*
* @param ind0 The primary mode to compare. Defer tie-breaks to ind1.
* @param ind1 The secondary mode to compare. Defer tie-breaks to ind2.
* @param ind2 The final tie-breaking mode.
* @param i The index into ind*.
* @param j The second index into ind*.
*
* @return Returns -1 if ind[i] < ind[j], 1 if ind[i] > ind[j], and 0 if they
*         are equal.
*/
static inline int p_ttcmp3(
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


/**
* @brief Compares ind*[i] and ind*[j] for n-mode tensors.
*
* @param tt The tensor we are sorting.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param i The index into ind*.
* @param j The second index into ind*.
*
* @return Returns -1 if ind[i] < ind[j], 1 if ind[i] > ind[j], and 0 if they
*         are equal.
*/
static inline int p_ttcmp(
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


/**
* @brief Compares ind*[i] and ind*[j] for n-mode tensors.
*
* @param tt The tensor we are sorting.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param i The index into ind*.
* @param j The coordinate we are comparing against.
*
* @return Returns -1 if ind[i] < j, 1 if ind[i] > j, and 0 if they are equal.
*/
static inline int p_ttqcmp(
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


/**
* @brief Swap nonzeros i and j.
*
* @param tt The tensor to operate on.
* @param i The first nonzero to swap.
* @param j The second nonzero to swap with.
*/
static inline void p_ttswap(
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


/**
* @brief Perform insertion sort on a 3-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_insertionsort3(
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
    size_t j = i;
    while (j > start && p_ttcmp3(ind0, ind1, ind2, i, j-1) < 0) {
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


/**
* @brief Perform insertion sort on an n-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_insertionsort(
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
    size_t j = i;
    while (j > start && p_ttcmp(tt, cmplt, i, j-1) < 0) {
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


/**
* @brief Perform quicksort on a 3-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_quicksort3(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  val_t vmid;
  idx_t imid[3];

  idx_t * const ind0 = tt->ind[cmplt[0]];
  idx_t * const ind1 = tt->ind[cmplt[1]];
  idx_t * const ind2 = tt->ind[cmplt[2]];
  val_t * const vals = tt->vals;

  if((end-start) <= MIN_QUICKSORT_SIZE) {
    p_tt_insertionsort3(tt, cmplt, start, end);
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
      if(p_ttqcmp3(ind0,ind1,ind2,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(p_ttqcmp3(ind0,ind1,ind2,j,imid) < 1) {
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
        if(p_ttqcmp3(ind0,ind1,ind2,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(p_ttqcmp3(ind0,ind1,ind2,i,imid) == 1) {
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
      p_tt_quicksort3(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      p_tt_quicksort3(tt, cmplt, i, end);
    }
  }
}


/**
* @brief Perform quicksort on a n-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_quicksort(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  val_t vmid;
  idx_t imid[MAX_NMODES];

  idx_t * ind;
  val_t * const vals = tt->vals;
  idx_t const nmodes = tt->nmodes;

  if((end-start) <= MIN_QUICKSORT_SIZE) {
    p_tt_insertionsort(tt, cmplt, start, end);
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
      if(p_ttqcmp(tt,cmplt,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(p_ttqcmp(tt,cmplt,j,imid) < 1) {
          p_ttswap(tt,i,j);
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(p_ttqcmp(tt,cmplt,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(p_ttqcmp(tt,cmplt,i,imid) == 1) {
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
      p_tt_quicksort(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      p_tt_quicksort(tt, cmplt, i, end);
    }
  }
}


/**
* @brief Perform a simple parallel quicksort with OpenMP tasks.
*
* @param a The array to sort.
* @param n The length of the array.
*/
static void p_quicksort(
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
      p_quicksort(a,i);
    }
    ++i; /* skip the pivot element */
    if(n-i > 1) {
      p_quicksort(a+i, n-i);
    }
  }
}


/**
 * idx = idx2*dim1 + idx1
 * -> ret = idx1*dim2 + idx2
 *        = (idx%dim1)*dim2 + idx/dim1
 */
static inline idx_t p_transpose_idx(
    idx_t const idx,
    idx_t const dim1,
    idx_t const dim2)
{
  return idx%dim1*dim2 + idx/dim1;
}

/**
* @brief Perform a counting sort on the most significant mode (cmplt[0]) and
*        then parallel quicksorts on each of slices.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
*/
static void p_counting_sort_hybrid(
    sptensor_t * const tt,
    idx_t * const cmplt)
{
  idx_t m = cmplt[0];
  idx_t nslices = tt->dims[m];

  idx_t * new_ind[MAX_NMODES];
  for(idx_t i = 0; i < tt->nmodes; ++i) {
    if(i != m) {
      new_ind[i] = splatt_malloc(tt->nnz * sizeof(**new_ind));
    }
  }
  val_t * new_vals = splatt_malloc(tt->nnz * sizeof(*new_vals));

  idx_t * histogram_array = splatt_malloc(
      (nslices * omp_get_max_threads() + 1) * sizeof(*histogram_array));

  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    idx_t * histogram = histogram_array + (nslices * tid);
    memset(histogram, 0, nslices * sizeof(idx_t));

    idx_t j_per_thread = (tt->nnz + nthreads - 1)/nthreads;
    idx_t jbegin = SS_MIN(j_per_thread*tid, tt->nnz);
    idx_t jend = SS_MIN(jbegin + j_per_thread, tt->nnz);

    /* count */
    for(idx_t j = jbegin; j < jend; ++j) {
      idx_t idx = tt->ind[m][j];
      ++histogram[idx];
    }

    #pragma omp barrier

    /* prefix sum */
    for(idx_t j = (tid*nslices) + 1; j < (tid+1) * nslices; ++j) {
      idx_t transpose_j = p_transpose_idx(j, nthreads, nslices);
      idx_t transpose_j_minus_1 = p_transpose_idx(j - 1, nthreads, nslices);

      histogram_array[transpose_j] += histogram_array[transpose_j_minus_1];
    }

    #pragma omp barrier
    #pragma omp master
    {
      for(int t = 1; t < nthreads; ++t) {
        idx_t j0 = (nslices*t) - 1, j1 = nslices * (t+1) - 1;
        idx_t transpose_j0 = p_transpose_idx(j0, nthreads, nslices);
        idx_t transpose_j1 = p_transpose_idx(j1, nthreads, nslices);

        histogram_array[transpose_j1] += histogram_array[transpose_j0];
      }
    }
    #pragma omp barrier

    if (tid > 0) {
      idx_t transpose_j0 = p_transpose_idx(nslices*tid - 1, nthreads, nslices);

      for(idx_t j = tid*nslices; j < (tid+1) * nslices - 1; ++j) {

        idx_t transpose_j = p_transpose_idx(j, nthreads, nslices);

        histogram_array[transpose_j] += histogram_array[transpose_j0];
      }
    }

    #pragma omp barrier


    /* now copy values into new structures (but not the mode we are sorting */
    for(idx_t j_off = 0; j_off < (jend-jbegin); ++j_off) {
      /* we are actually going backwards */
      idx_t const j = jend - j_off - 1;

      idx_t idx = tt->ind[m][j];
      --histogram[idx];

      idx_t offset = histogram[idx];

      new_vals[offset] = tt->vals[j];
      for(idx_t mode=0; mode < tt->nmodes; ++mode) {
        if(mode != m) {
          new_ind[mode][offset] = tt->ind[mode][j];
        }
      }
    }
  } /* omp parallel */

  for(idx_t i = 0; i < tt->nmodes; ++i) {
    if(i != m) {
      splatt_free(tt->ind[i]);
      tt->ind[i] = new_ind[i];
    }
  }
  splatt_free(tt->vals);
  tt->vals = new_vals;


  histogram_array[nslices] = tt->nnz;

  /* for 3/4D, we can use quicksort on only the leftover modes */
  if(tt->nmodes == 3) {
    #pragma omp parallel for schedule(dynamic)
    for(idx_t i = 0; i < nslices; ++i) {
      p_tt_quicksort2(tt, cmplt+1, histogram_array[i], histogram_array[i + 1]);
      for(idx_t j = histogram_array[i]; j < histogram_array[i + 1]; ++j) {
        tt->ind[m][j] = i;
      }
    }

  } else if(tt->nmodes == 4) {
    #pragma omp parallel for schedule(dynamic)
    for(idx_t i = 0; i < nslices; ++i) {
      p_tt_quicksort3(tt, cmplt+1, histogram_array[i], histogram_array[i + 1]);
      for(idx_t j = histogram_array[i]; j < histogram_array[i + 1]; ++j) {
        tt->ind[m][j] = i;
      }
    }

  } else {
    /* shift cmplt left one time, then do normal quicksort */
    idx_t saved = cmplt[0];
    memmove(cmplt, cmplt+1, (tt->nmodes - 1) * sizeof(*cmplt));
    cmplt[tt->nmodes-1] = saved;

    #pragma omp parallel for schedule(dynamic)
    for(idx_t i = 0; i < nslices; ++i) {
      p_tt_quicksort(tt, cmplt, histogram_array[i], histogram_array[i + 1]);
      for(idx_t j = histogram_array[i]; j < histogram_array[i + 1]; ++j) {
        tt->ind[m][j] = i;
      }
    }

    /* undo cmplt changes */
    saved = cmplt[tt->nmodes-1];
    memmove(cmplt+1, cmplt, (tt->nmodes - 1) * sizeof(*cmplt));
    cmplt[0] = saved;
  }

  splatt_free(histogram_array);
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_sort(
  sptensor_t * const tt,
  idx_t const mode,
  idx_t * dim_perm)
{
  tt_sort_range(tt, mode, dim_perm, 0, tt->nnz);
}


void tt_sort_range(
  sptensor_t * const tt,
  idx_t const mode,
  idx_t * dim_perm,
  idx_t const start,
  idx_t const end)
{
  idx_t * cmplt;
  if(dim_perm == NULL) {
    cmplt = (idx_t*) splatt_malloc(tt->nmodes * sizeof(idx_t));
    cmplt[0] = mode;
    for(idx_t m=1; m < tt->nmodes; ++m) {
      cmplt[m] = (mode + m) % tt->nmodes;
    }
  } else {
    cmplt = dim_perm;
  }

  timer_start(&timers[TIMER_SORT]);
  if(start == 0 && end == tt->nnz) {
    p_counting_sort_hybrid(tt, cmplt);

  /* sort a subtensor */
  } else {
    switch(tt->type) {
    case SPLATT_NMODE:
      p_tt_quicksort(tt, cmplt, start, end);
      break;

    case SPLATT_3MODE:
      p_tt_quicksort3(tt, cmplt, start, end);
      break;
    }
  }


  if(dim_perm == NULL) {
    free(cmplt);
  }
  timer_stop(&timers[TIMER_SORT]);
}


void insertion_sort(
  idx_t * const a,
  idx_t const n)
{
  timer_start(&timers[TIMER_SORT]);
  for(size_t i=1; i < n; ++i) {
    idx_t b = a[i];
    size_t j = i;
    while (j > 0 &&  a[j-1] > b) {
      --j;
    }
    memmove(a+(j+1), a+j, sizeof(idx_t)*(i-j));
    a[j] = b;
  }
  timer_stop(&timers[TIMER_SORT]);
}


void quicksort(
  idx_t * const a,
  idx_t const n)
{
  timer_start(&timers[TIMER_SORT]);
  p_quicksort(a,n);
  timer_stop(&timers[TIMER_SORT]);
}

