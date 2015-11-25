
extern "C"
{

#include "svd.h"
#include "timer.h"

}

#include <Eigen/Core>
#include <Eigen/SVD>


typedef Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    EMat;


void left_singulars(
    val_t * inmat,
    val_t * outmat,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const rank)
{
	timer_start(&timers[TIMER_SVD]);

  Eigen::Map<EMat> tmp(inmat, nrows, ncols);

  /* computeSVD */
  Eigen::JacobiSVD<EMat, Eigen::HouseholderQRPreconditioner>
      svd(tmp, Eigen::ComputeThinU);

  /* now copy into outmat */
  EMat left = svd.matrixU().leftCols(rank);
  memcpy(outmat, left.data(), nrows * rank * sizeof(*outmat));

	timer_stop(&timers[TIMER_SVD]);
}

