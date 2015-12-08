
#include "svd.h"
#include "matrix.h"
#include "timer.h"
#include "util.h"

int optimal_svd_work_size(
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

  val_t work_size;
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

  printf("work_size: %d\n", (int) work_size);

  return (int) work_size;
}


void alloc_svd_ws(
    val_t ** svdbuf,
    val_t ** U,
    val_t ** S,
    val_t ** Vt,
    val_t ** lwork,
    int ** iwork,
    idx_t const nmats,
    idx_t const * const nrows,
    idx_t const * const ncolumns)
{
  /* allocate SVD workspace */
  idx_t maxmat = 0;
  for(idx_t m=0; m < nmats, ++m) {
    maxmat = SS_MAX(maxmat, nrows[m] * ncolumns[m]);
  }
  *svdbuf = malloc(maxmat * sizeof(**svdbuf));

  *svdbuf = NULL;
  *U = NULL;
  *S = NULL;
  *Vt = NULL;

  int max_lwork = 0;
  for(idx_t m=0; m < nmodes; ++m) {
    int work_size = optimal_svd_work_size(tensors->dims[m], ws->gten_cols[m]);
    max_lwork = SS_MAX(max_lwork, work_size);
  }
  ws->lwork = malloc(max_lwork * sizeof(*(ws->lwork)));
}


void left_singulars(
    val_t * inmat,
    val_t * outmat,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const rank)
{
	timer_start(&timers[TIMER_SVD]);

  char jobz = 'S';

  /* actually pass in A^T */
  int M = ncols;
  int N = nrows;
  int LDA = M;

  val_t * S = malloc(SS_MIN(M,N) * sizeof(*S));

  /* NOTE: change these if we switch to jobz=O */
  int LDU = M;
  int LDVt = SS_MIN(M,N);

  val_t * U = malloc(LDU * SS_MIN(M,N) * sizeof(*U));
  val_t * Vt = malloc(LDVt * N * sizeof(*Vt));

  val_t work_size;
  int lwork = -1;
  int * iwork = malloc(8 * SS_MIN(M,N) * sizeof(*iwork));
  int info = 0;

  /* query */
  LAPACK_SVD(
      &jobz,
      &M, &N,
      inmat, &LDA,
      S,
      U, &LDU,
      Vt, &LDVt,
      &work_size, &lwork,
      iwork, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGESDD returned %d\n", info);
  }

  lwork = work_size;
  val_t * workspace = malloc(lwork * sizeof(*workspace));

  /* do the SVD */
  LAPACK_SVD(
      &jobz,
      &M, &N,
      inmat, &LDA,
      S,
      U, &LDU,
      Vt, &LDVt,
      workspace, &lwork,
      iwork, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGESDD returned %d\n", info);
  }


  /* copy matrix Vt to outmat */
  for(idx_t r=0; r < nrows; ++r) {
    for(idx_t c=0; c < rank; ++c) {
      outmat[c + (r*rank)] = Vt[c + (r*LDVt)];
    }
  }

  free(workspace);
  free(iwork);

  free(S);
  free(Vt);
  free(U);

	timer_stop(&timers[TIMER_SVD]);
}


