

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "util.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
val_t rand_val(void)
{
  /* TODO: modify this to work based on the size of val_t */
  val_t v =  3.0 * ((val_t) rand() / (val_t) RAND_MAX);
  if(rand() % 2 == 0) {
    v *= -1;
  }
  return v;
}


idx_t rand_idx(void)
{
  /* TODO: modify this to work based on the size of idx_t */
  return (idx_t) (rand() << 16) | rand();
}


void fill_rand(
  val_t * const restrict vals,
  idx_t const nelems)
{
  for(idx_t i=0; i < nelems; ++i) {
    vals[i] = rand_val();
  }
}


char * bytes_str(
  size_t const bytes)
{
  double size = (double)bytes;
  int suff = 0;
  const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
  while(size > 1024 && suff < 5) {
    size /= 1024.;
    ++suff;
  }
  char * ret = NULL;
  if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    fprintf(stderr, "SPLATT: asprintf failed with%"SPLATT_PF_IDX" bytes.\n",
        bytes);
    ret = NULL;
  }
  return ret;
}



idx_t argmax_elem(
  idx_t const * const arr,
  idx_t const N)
{
  idx_t mkr = 0;
  for(idx_t i=1; i < N; ++i) {
    if(arr[i] > arr[mkr]) {
      mkr = i;
    }
  }
  return mkr;
}


idx_t argmin_elem(
  idx_t const * const arr,
  idx_t const N)
{
  idx_t mkr = 0;
  for(idx_t i=1; i < N; ++i) {
    if(arr[i] < arr[mkr]) {
      mkr = i;
    }
  }
  return mkr;
}

