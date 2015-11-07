
#define CTEST_MAIN
#define CTEST_SEGFAULT
#include "ctest/ctest.h"

#include "splatt_test.h"

#ifdef SPLATT_USE_MPI
#include <mpi.h>
#endif

int main(int argc, char ** argv)
{
#ifdef SPLATT_USE_MPI
  /* even serial tests with MPI enabled need this */
  MPI_Init(&argc, &argv);
#endif

  /* this cast is annoying */
  char const ** margv = (char const **)argv;
  int ret = ctest_main(argc, margv);

#ifdef SPLATT_USE_MPI
  MPI_Finalize();
#endif
  return ret;
}

