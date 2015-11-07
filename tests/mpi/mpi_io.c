
#include "../ctest/ctest.h"
#include "../splatt_test.h"

#include "../../src/sptensor.h"


CTEST_DATA(mpi_io)
{
  sptensor_t * tensors;
};

CTEST_SETUP(mpi_io)
{
  data->tensors = NULL;
}

CTEST_TEARDOWN(mpi_io)
{
  free(data->tensors);
}

CTEST2(mpi_io, hi)
{
  ASSERT_EQUAL(1, 1);
}

