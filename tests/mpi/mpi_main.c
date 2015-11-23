
#define CTEST_MAIN
#define CTEST_SEGFAULT
#include "../ctest/ctest.h"

#include <mpi.h>

int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  /* this cast is annoying */
  char const ** margv = (char const **)argv;
  int ret = ctest_main(argc, margv);

  MPI_Finalize();
  return ret;
}

