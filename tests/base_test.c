
#include "../src/base.h"
#include "ctest/ctest.h"
#include "splatt_test.h"


CTEST(base, alloc_aligned)
{
  void * ptr = splatt_malloc(4096);
  ASSERT_EQUAL(0, (int)ptr % 64);
  splatt_free(ptr);
}

