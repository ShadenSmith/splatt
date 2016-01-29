
#include "../ctest/ctest.h"
#include "../splatt_test.h"

#include "../../src/io.h"
#include "../../src/sptensor.h"
#include "../../src/sort.h"

static char const * const TMP_FILE = "tmp.bin";


CTEST_DATA(mpi_io)
{
  idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
  rank_info rinfo;
};

CTEST_SETUP(mpi_io)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}

CTEST_TEARDOWN(mpi_io)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}

CTEST2(mpi_io, splatt_mpi_coord_load)
{
  int rank, npes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  double * opts = splatt_default_opts();
  for(idx_t tt=0; tt < data->ntensors; ++tt) {
    splatt_idx_t nmodes;
    splatt_idx_t nnz;
    splatt_idx_t ** inds;
    splatt_val_t * vals;

    int ret = splatt_mpi_coord_load(datasets[tt], &nmodes, &nnz,  &inds, &vals,
        opts, MPI_COMM_WORLD);

    ASSERT_EQUAL(SPLATT_SUCCESS, ret);
    ASSERT_EQUAL(data->tensors[tt]->nmodes, nmodes);
    ASSERT_NOT_NULL(vals);
    ASSERT_NOT_NULL(inds);
    for(idx_t m=0; m < nmodes; ++m) {
      ASSERT_NOT_NULL(inds[m]);
    }

    /* check nnz */
    idx_t const global = data->tensors[tt]->nnz;
    idx_t const target = global / npes;
    if(rank == 0) {
      ASSERT_EQUAL(global-(target * (npes-1)), nnz);
    } else {
      ASSERT_EQUAL(target, nnz);
    }
    /* everyone has the correct, this is an extra sanity check */
    idx_t tot_nnz;
    MPI_Reduce(&nnz, &tot_nnz, 1, SPLATT_MPI_IDX, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0) {
      ASSERT_EQUAL(global, tot_nnz);
    }


    /* now check inds/vals */
    sptensor_t * tmp = NULL;
    if(rank == 0) {
      tmp = tt_alloc(global, nmodes);

      /* copy mine */
      memcpy(tmp->vals, vals, nnz * sizeof(*(tmp->vals)));
      for(idx_t m=0; m < nmodes; ++m) {
        memcpy(tmp->ind[m], inds[m], nnz * sizeof(**inds));
      }

      /* collect from every process */
      MPI_Status status;
      for(int p=1; p < npes; ++p) {
        MPI_Recv(tmp->vals + nnz + ((p-1) * target), target, SPLATT_MPI_VAL,
            p, nmodes, MPI_COMM_WORLD, &status);
        for(idx_t m=0; m < nmodes; ++m) {
          MPI_Recv(tmp->ind[m] + nnz + ((p-1) * target), target, SPLATT_MPI_IDX,
              p, m, MPI_COMM_WORLD, &status);
        }
      }

      tt_sort(tmp, 0, NULL);
      tt_sort(data->tensors[tt], 0, NULL);

      /* now compare */
      val_t const * const goldv = data->tensors[tt]->vals;
      for(idx_t n=0; n < tmp->nnz; ++n) {
        ASSERT_DBL_NEAR_TOL(goldv[n], tmp->vals[n], 0);
      }
      for(idx_t m=0; m < tmp->nmodes; ++m) {
        idx_t const * const goldi = data->tensors[tt]->ind[m];
        for(idx_t n=0; n < tmp->nnz; ++n) {
          ASSERT_EQUAL(goldi[n], tmp->ind[m][n]);
        }
      }

      tt_free(tmp);
    } else {

      /* non-root just sends tensor data */
      MPI_Send(vals, nnz, SPLATT_MPI_VAL, 0, nmodes, MPI_COMM_WORLD);
      for(idx_t m=0; m < nmodes; ++m) {
        MPI_Send(inds[m], nnz, SPLATT_MPI_IDX, 0, m, MPI_COMM_WORLD);
      }
    }
  } /* foreach tensor */
}


CTEST2(mpi_io, splatt_mpi_coord_load_binary)
{
  int rank, npes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  double * opts = splatt_default_opts();
  for(idx_t tt=0; tt < data->ntensors; ++tt) {

    /* convert to binary */
    if(rank == 0) {
      tt_write_binary(data->tensors[tt], TMP_FILE);
    }

    /* distribute binary tensor */
    splatt_idx_t nmodes;
    splatt_idx_t nnz;
    splatt_idx_t ** inds;
    splatt_val_t * vals;
    int ret = splatt_mpi_coord_load(TMP_FILE, &nmodes, &nnz,  &inds, &vals,
        opts, MPI_COMM_WORLD);

    /* make sure things were actually allocated */
    ASSERT_EQUAL(SPLATT_SUCCESS, ret);
    ASSERT_EQUAL(data->tensors[tt]->nmodes, nmodes);
    ASSERT_NOT_NULL(vals);
    ASSERT_NOT_NULL(inds);
    for(idx_t m=0; m < nmodes; ++m) {
      ASSERT_NOT_NULL(inds[m]);
    }

    /* now check distribution - exact comparison should be good (no sorting) */
    sptensor_t * gold = tt_read(TMP_FILE);
    idx_t const target_nnz = gold->nnz / npes;
    if(rank == 0) {
      ASSERT_EQUAL(gold->nnz - ((npes-1) * target_nnz), nnz);
    } else {
      ASSERT_EQUAL(target_nnz, nnz);
    }

    /* all nnz better be accounted for */
    idx_t total_nnz;
    MPI_Allreduce(&nnz, &total_nnz, 1, SPLATT_MPI_IDX, MPI_SUM,
        MPI_COMM_WORLD);
    ASSERT_EQUAL(gold->nnz, total_nnz);

    /* where my nonzeros start (relative to global tensor) */
    idx_t start_nnz = 0;
    if(rank > 0) {
      start_nnz = (rank-1) * target_nnz;
    } else {
      start_nnz = (npes-1) * target_nnz;
    }

    /* check inds */
    for(idx_t m=0; m < nmodes; ++m) {
      for(idx_t n=0; n < nnz; ++n) {
        ASSERT_EQUAL(gold->ind[m][n + start_nnz], inds[m][n]);
      }
      splatt_free(inds[m]);
    }
    /* check vals */
    for(idx_t n=0; n < nnz; ++n) {
      ASSERT_DBL_NEAR_TOL(gold->vals[n + start_nnz], vals[n], 0.);
    }

    splatt_free(inds);
    splatt_free(vals);
    tt_free(gold);
  }

  /* delete temporary file */
  if(rank == 0) {
    remove(TMP_FILE);
  }
}






