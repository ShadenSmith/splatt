
#include "../src/mttkrp.h"

#include "ctest/ctest.h"

#include "splatt_test.h"


CTEST_DATA(mttkrp)
{
  int ntensors;
  sptensor_t * tensors[128];
};

CTEST_SETUP(mttkrp)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(int i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}

CTEST_TEARDOWN(mttkrp)
{
  for(int i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}

CTEST2(mttkrp, ttbox)
{

}

CTEST2(mttkrp, giga)
{

}

