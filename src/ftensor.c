

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "ftensor.h"
#include "sort.h"


/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/
static void __create_fptr(
  ftensor_t * const ft,
  sptensor_t const * const tt,
  idx_t const mode)
{
  idx_t nfibs = 0;


}

static void __create_sptr(
  ftensor_t * const ft,
  sptensor_t const * const tt,
  idx_t const mode)
{

}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
ftensor_t * ften_alloc(
  sptensor_t * const tt)
{
  ftensor_t * ft = (ftensor_t *) malloc(sizeof(ftensor_t));

  ft->nnz = tt->nnz;
  ft->nmodes = tt->nmodes;

  idx_t perm[MAX_NMODES];

  /* allocate modal data */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    ft->inds[m] = (idx_t *) malloc(ft->nnz * sizeof(idx_t));
    ft->vals[m] = (val_t *) malloc(ft->nnz * sizeof(val_t));

    tt_sort(tt, m, NULL);
    __create_fptr(ft, tt, m);
    __create_sptr(ft, tt, m);
  }

  return ft;
}

void ften_free(
  ftensor_t * ft)
{
  for(idx_t m=0; m < ft->nmodes; ++m) {
    //free(ft->sptr[m]);
    //free(ft->fptr[m]);
    //free(ft->fids[m]);
    free(ft->inds[m]);
    free(ft->vals[m]);
  }
  free(ft);
}


