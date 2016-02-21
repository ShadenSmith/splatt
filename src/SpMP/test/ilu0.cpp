#include <cstring>
#include <set>

#include "../LevelSchedule.hpp"
#include "../synk/barrier.hpp"

#include "test.hpp"

int main(int argc, char **argv)
{
  /////////////////////////////////////////////////////////////////////////////
  // Load input
  /////////////////////////////////////////////////////////////////////////////

  int m = argc > 1 ? atoi(argv[1]) : 64; // default input is 64^3 27-pt 3D Lap.
  if (argc < 2) {
    fprintf(
      stderr,
      "Using default 64^3 27-pt 3D Laplacian matrix\n"
      "-- Usage examples --\n"
      "  %s 128 : 128^3 27-pt 3D Laplacian matrix\n"
      "  %s inline_1.mtx: run with inline_1 matrix in matrix market format\n\n",
      argv[0], argv[0]);
  }
  char buf[1024];
  sprintf(buf, "%d", m);

  bool readFromFile = argc > 1 ? strcmp(buf, argv[1]) && !strstr(argv[1], ".mtx"): false;
  printf("input=%s\n", argc > 1 ? argv[1] : buf);

  CSR *A = new CSR(argc > 1 ? argv[1] : buf, true /*force symmetric*/);
  CSR *A2 = new CSR(A->m, A->m, A->getNnz() + A->m);
  int nnz = 0;
  A2->rowptr[0] = 0;
  for (int i = 0; i < A->m; ++i) {
    assert(is_sorted(A->colidx + A->rowptr[i], A->colidx + A->rowptr[i + 1]));
    bool diagAdded = false;
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {
      if (!diagAdded && A->colidx[j] > i) {
        A2->colidx[nnz] = i;
        A2->diagptr[i] = nnz;
        A->diagptr[i] = j;
        ++nnz;
        diagAdded = true;
      }
      if (A->colidx[j] == i) {
        diagAdded = true;
        A2->diagptr[i] = nnz;
        A->diagptr[i] = j;
      }
      A2->colidx[nnz] = A->colidx[j];
      ++nnz;
    }
    if (!diagAdded) {
      A2->colidx[nnz] = i;
      A2->diagptr[i] = nnz;
      A->diagptr[i] = A->rowptr[i + 1];
      ++nnz;
    }
    A2->rowptr[i + 1] = nnz;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Construct schedules
  /////////////////////////////////////////////////////////////////////////////

  LevelSchedule *barrierSchedule = new LevelSchedule;
  barrierSchedule->useBarrier = true;
  barrierSchedule->transitiveReduction = false;
  barrierSchedule->constructTaskGraph(*A2);

  printf("parallelism %f\n", (double)A->m/(barrierSchedule->levIndices.size() - 1));
  double initialParallelism = (double)A->m/(barrierSchedule->levIndices.size() - 1);

  set<pair<int, int> > edges;

  int *flags = (int *)calloc(A->getNnz(), sizeof(int));

  double totalCnt = 0, edgeCnt = 0;
  for (int i = 0; i < A->m; ++i) {
    for (int k = A->rowptr[i]; k < A->diagptr[i]; ++k) {
      int c = A->colidx[k];
      int j1 = k + 1, j2 = A->diagptr[c] + 1;

      bool hasEdge = false;
      flags[k] = 1;
      if (flags[A->diagptr[c]]) {
        hasEdge = true;
      }

      while (j1 < A->rowptr[i + 1] && j2 < A->rowptr[c + 1]) {
        if (A->colidx[j1] < A->colidx[j2]) ++j1;
        else if (A->colidx[j2] < A->colidx[j1]) ++j2;
        else {
          flags[j1] = 1;
          if (flags[j2]) {
            hasEdge = true;
          }
          ++j1; ++j2;
        }
      }

      ++totalCnt;
      if (hasEdge) {
        edges.insert(make_pair(i, c));
        edges.insert(make_pair(c, i));
        ++edgeCnt;
      }
      else {
        //printf("%d->%d\n", c, i);
      }
    }
  } // for each row

  printf("%g/%g = %g\n", edgeCnt, totalCnt, (double)edgeCnt/totalCnt);
  
  //A->print();

  auto itr = edges.begin();
  CSR *B = new CSR(A->m, A->m, edges.size() + A->m);
  int j = 0;
  B->rowptr[0] = 0;
  for (int i = 0; i < A->m; ++i) {
    bool diagAdded = false;
    set<int> refCols;
    for (int k = A2->rowptr[i]; k < A2->rowptr[i + 1]; ++k) {
      refCols.insert(A2->colidx[k]);
    }
    assert(refCols.find(i) != refCols.end());
    while (itr != edges.end() && itr->first == i) {
      if (!diagAdded && itr->second > i) {
        B->colidx[j] = i;
        B->diagptr[i] = j;
        ++j;
        diagAdded = true;
      }
      B->colidx[j] = itr->second;
      ++j;
      ++itr;
    }
    if (!diagAdded) {
      B->colidx[j] = i;
      B->diagptr[i] = j;
      ++j;
    }
    B->rowptr[i + 1] = j;

    set<int> cols;
    for (int k = B->rowptr[i]; k < B->rowptr[i + 1]; ++k) {
      cols.insert(B->colidx[k]);
    }
    assert(includes(refCols.begin(), refCols.end(), cols.begin(), cols.end()));
  } // for each row

  //B->print();

  barrierSchedule = new LevelSchedule;
  barrierSchedule->useBarrier = true;
  barrierSchedule->transitiveReduction = false;
  barrierSchedule->constructTaskGraph(*B);

  printf("parallelism %f\n", (double)A->m/(barrierSchedule->levIndices.size() - 1));
  double newParallelism = (double)A->m/(barrierSchedule->levIndices.size() - 1);
  printf("%s %g\n", argv[1], newParallelism/initialParallelism);

  return 0;
}
