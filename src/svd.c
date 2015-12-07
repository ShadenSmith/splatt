
#include "svd.h"
#include "matrix.h"
#include "timer.h"
#include "util.h"


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


