

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
  /* TODO: modify this to work based on the size of idx_t */
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
  idx_t const bytes)
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
    fprintf(stderr, "SPLATT: asprintf failed with%"SPLATT_PF_IDX" bytes.\n", bytes);
    ret = NULL;
  }
  return ret;
}

