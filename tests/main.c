
#define CTEST_MAIN
#define CTEST_SEGFAULT
#include "ctest/ctest.h"

#include "splatt_test.h"

int main(int argc, char const ** argv)
{
  printf("found %s\n", DATASET(small.tns));
  printf("found %s\n", DATASET(small4.tns));
  printf("found %s\n", DATASET(med.tns));
  printf("found %s\n", DATASET(small.tns));
  return ctest_main(argc, argv);
}

