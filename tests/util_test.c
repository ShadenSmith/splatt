

#include "../src/util.h"
#include "ctest/ctest.h"

#include "splatt_test.h"

CTEST_DATA(util)
{
  idx_t llist;
  idx_t * forlist; /* increasing order */
  idx_t * revlist; /* decreasing order */
};


CTEST_SETUP(util)
{
  data->llist = 10;
  data->forlist = malloc(data->llist * sizeof(*(data->revlist)));
  data->revlist = malloc(data->llist * sizeof(*(data->revlist)));

  for(idx_t i=0; i < data->llist; ++i) {
    data->forlist[i] = i;
    data->revlist[i] = data->llist - i;
  }
}

CTEST_TEARDOWN(util)
{
  free(data->forlist);
  free(data->revlist);
}


CTEST2(util, max)
{
  idx_t const * const flist = data->forlist;
  idx_t const * const rlist = data->revlist;

  idx_t fmax = flist[0];
  idx_t rmax = rlist[0];
  for(idx_t i=0; i < data->llist; ++i) {
    fmax = SS_MAX(fmax, flist[i]);
    rmax = SS_MAX(rmax, rlist[i]);
  }

  for(idx_t i=0; i < data->llist; ++i) {
    if(rlist[i] > rmax) {
      ASSERT_FAIL();
    }

    if(flist[i] > fmax) {
      ASSERT_FAIL();
    }
  }
}


CTEST2(util, min)
{
  idx_t const * const flist = data->forlist;
  idx_t const * const rlist = data->revlist;

  idx_t fmin = flist[0];
  idx_t rmin = rlist[0];
  for(idx_t i=0; i < data->llist; ++i) {
    fmin = SS_MIN(fmin, flist[i]);
    rmin = SS_MIN(rmin, rlist[i]);
  }

  for(idx_t i=0; i < data->llist; ++i) {
    if(rlist[i] < rmin) {
      ASSERT_FAIL();
    }

    if(flist[i] < fmin) {
      ASSERT_FAIL();
    }
  }
}


CTEST2(util, argmax)
{
  idx_t const * const flist = data->forlist;
  idx_t const * const rlist = data->revlist;

  idx_t fmax = flist[argmax_elem(flist, data->llist)];
  idx_t rmax = rlist[argmax_elem(rlist, data->llist)];

  for(idx_t i=0; i < data->llist; ++i) {
    if(rlist[i] > rmax) {
      ASSERT_FAIL();
    }

    if(flist[i] > fmax) {
      ASSERT_FAIL();
    }
  }
}



CTEST2(util, argmin)
{
  idx_t const * const flist = data->forlist;
  idx_t const * const rlist = data->revlist;

  idx_t fmin = flist[argmin_elem(flist, data->llist)];
  idx_t rmin = rlist[argmin_elem(rlist, data->llist)];

  for(idx_t i=0; i < data->llist; ++i) {
    if(rlist[i] < rmin) {
      ASSERT_FAIL();
    }

    if(flist[i] < fmin) {
      ASSERT_FAIL();
    }
  }
}

