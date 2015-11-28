
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


void make_core(
    val_t * ttmc,
    val_t * lastmat,
    val_t * core,
    idx_t const nmodes,
    idx_t const mode,
    idx_t const * const nfactors,
    idx_t const nlongrows)
{
	timer_start(&timers[TIMER_MATMUL]);
  idx_t ncols = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    if(m != mode) {
      ncols *= nfactors[m];
    }
  }

  Eigen::Map<EMat> ttmc_mat(ttmc, nlongrows, ncols);
  Eigen::Map<EMat> last_factor(lastmat, nlongrows, nfactors[mode]);

  EMat core_mat = last_factor.transpose() * ttmc_mat;
  memcpy(core, core_mat.data(), nfactors[mode] * ncols * sizeof(*core));
	timer_stop(&timers[TIMER_MATMUL]);
}


