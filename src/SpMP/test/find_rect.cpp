#include <cstdio>
#include "../mm_io.h"

int main(int argc, const char *argv[])
{
  FILE *fp = fopen(argv[1], "r");

  // read banner
  MM_typecode matcode;
  if (mm_read_banner (fp, &matcode) != 0) {
    fprintf(stderr, "Error: could not process Matrix Market banner of %s.\n", argv[1]);
    fclose(fp);
    return -1;
  }

  if (!mm_is_valid (matcode) ||
       mm_is_array (matcode) || mm_is_dense (matcode) ) {
    fprintf(stderr, "Error: only support sparse and real matrices of %s.\n", argv[1]);
    fclose(fp);
    return -1;
  }
  bool pattern = mm_is_pattern (matcode);

  // read sizes
  int m, n;
  int nnz; // # of non-zeros specified in the file
  if (mm_read_mtx_crd_size(fp, &m, &n, &nnz) !=0) {
    fprintf(stderr, "Error: could not read matrix size of %s.\n", argv[1]);
    fclose(fp);
    return -1;
  }

  if (m != n) {
    printf("%s is rectangular %d x %d\n", argv[1], m, n);
  }

  fclose(fp);

  return 0;
}
