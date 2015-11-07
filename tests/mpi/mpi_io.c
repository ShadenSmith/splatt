
#include "../ctest/ctest.h"
#include "../splatt_test.h"

#include "../../src/sptensor.h"


CTEST_DATA(mpi_io)
{
  idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
  rank_info rinfo;
};

CTEST_SETUP(mpi_io)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}

CTEST_TEARDOWN(mpi_io)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}

CTEST2(mpi_io, simple_distribute)
{
  for(idx_t tt=0; tt < data->ntensors; ++tt) {

  }
}

