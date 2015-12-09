
#include "svd.h"
#include "matrix.h"
#include "timer.h"
#include "util.h"

/**
* @brief Compute the optimal workspace size for LAPACK_SVD.
*
* @param nrows The number of rows.
* @param ncols The number of columns.
*
* @return The optimal workspace size.
*/
static int p_optimal_svd_work_size(
    idx_t const nrows,
    idx_t const ncols)
{
  char jobz = 'S';

  /* actually pass in A^T */
  int M = ncols;
  int N = nrows;
  int LDA = M;

  val_t * S = NULL; //malloc(SS_MIN(M,N) * sizeof(*S));

  /* NOTE: change these if we switch to jobz=O */
  int LDU = M;
  int LDVt = SS_MIN(M,N);

  val_t * U = NULL; //malloc(LDU * SS_MIN(M,N) * sizeof(*U));
  val_t * Vt = NULL; //malloc(LDVt * N * sizeof(*Vt));

  val_t work_size = 0;
  int * iwork = NULL; //malloc(8 * SS_MIN(M,N) * sizeof(*iwork));
  int info = 0;

  val_t * A = NULL;

  /* query */
  int lwork = -1;
  LAPACK_SVD(
      &jobz,
      &M, &N,
      A, &LDA,
      S,
      U, &LDU,
      Vt, &LDVt,
      &work_size, &lwork,
      iwork, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGESDD returned %d\n", info);
  }

  return (int) work_size;
}


void left_singulars(
    val_t const * const inmat,
    val_t * outmat,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const rank,
    svd_ws * const ws)
{
	timer_start(&timers[TIMER_SVD]);

  char jobz = 'S';

  /* actually pass in A^T */
  int M = ncols;
  int N = nrows;
  int LDA = M;
  int LDU = M;
  int LDVt = SS_MIN(M,N);

  int info = 0;

  memcpy(ws->A, inmat, M * N * sizeof(*(ws->A)));

  /* do the SVD */
  LAPACK_SVD(
      &jobz,
      &M, &N,
      ws->A, &LDA,
      ws->S,
      ws->U, &LDU,
      ws->Vt, &LDVt,
      ws->workspace, &(ws->lwork),
      ws->iwork, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGESDD returned %d\n", info);
  }


  /* copy matrix Vt to outmat */
  val_t const * const Vt = ws->Vt;
  for(idx_t r=0; r < nrows; ++r) {
    for(idx_t c=0; c < rank; ++c) {
      outmat[c + (r*rank)] = Vt[c + (r*LDVt)];
    }
  }

	timer_stop(&timers[TIMER_SVD]);
}



void alloc_svd_ws(
    svd_ws * const ws,
    idx_t const nmats,
    idx_t const * const nrows,
    idx_t const * const ncolumns)
{
  /* initialize */
  ws->A = NULL;
  ws->U = NULL;
  ws->S = NULL;
  ws->Vt = NULL;
  ws->workspace = NULL;
  ws->iwork = NULL;
  ws->lwork = 0;

  /* allocate SVD workspace */
  idx_t max_A = 0;
  int max_U = 0;
  int max_Vt = 0;
  int max_min = 0;
  int max_lwork = 0;
  for(idx_t m=0; m < nmats; ++m) {
    max_A = SS_MAX(max_A, nrows[m] * ncolumns[m]);

    int const M = (int) ncolumns[m];
    int const N = (int) nrows[m];
    int const mindim = SS_MIN(M, N);
    max_min = SS_MAX(max_min, mindim);

    max_U = SS_MAX(max_U, M * mindim);
    max_Vt = SS_MAX(max_Vt, N * mindim);

    int const lwork = p_optimal_svd_work_size(nrows[m], ncolumns[m]);
    max_lwork= SS_MAX(lwork, ws->lwork);
  }

  /* allocate matrices */
  ws->A = malloc(max_A * sizeof(*(ws->A)));
  ws->U = malloc(max_U * sizeof(*(ws->U)));
  ws->S = malloc(max_min * sizeof(*(ws->S)));
  ws->Vt = malloc(max_Vt * sizeof(*(ws->Vt)));

  /* allocate workspaces */
  ws->lwork = max_lwork;
  ws->workspace = malloc(max_lwork * sizeof(*(ws->workspace)));
  ws->iwork = malloc(8 * max_min * sizeof(*(ws->iwork)));
}


void free_svd_ws(
    svd_ws * const ws)
{
  free(ws->A);
  free(ws->S);
  free(ws->U);
  free(ws->Vt);
  free(ws->workspace);
  free(ws->iwork);
}

