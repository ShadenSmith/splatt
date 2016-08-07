#include <vector>
#include <set>
#include <unordered_set>
#include <algorithm>

#include <omp.h>
//#include <tbb/concurrent_unordered_set.h>

#include "CSR.hpp"

#include "reorder.h"
#include "csf.h"

using namespace std;
using namespace SpMP;

permutation_t *perm_bfs_or_rcm(sptensor_t * const tt, int use_rcm)
{
  permutation_t *perm = perm_alloc(tt->dims, tt->nmodes);
  assert(false); // temporarily disable to avoid tbb dependency
#if 0

#define SPLATT_FIBER_BASED_BFS
#ifdef SPLATT_FIBER_BASED_BFS
  // (i1, i2) when i1 and i2 appear in a same fiber

  double *opts = splatt_default_opts();
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLMODE_ROUND_ROBIN;
  splatt_csf *csf = splatt_csf_alloc(tt, opts);

  idx_t max_dims = 0;
  for(int m=0; m < tt->nmodes; ++m) {
    max_dims = max(max_dims, tt->dims[m]);
  }
  omp_lock_t *locks = new omp_lock_t[max_dims];
  for(int i=0; i < max_dims; ++i) {
    omp_init_lock(locks + i);
  }

  for(int k=0; k < tt->nmodes; ++k) {
    idx_t mode = csf[k].dim_perm[tt->nmodes - 1];

    CSR A;
    A.m = tt->dims[mode];
    A.n = A.m;

    vector<vector<int> > g(A.m);

    csf_sparsity *tile = csf[k].pt;

    idx_t *fptr = tile->fptr[tt->nmodes - 2];
    idx_t *inds = tile->fids[tt->nmodes - 1];

    idx_t nfibers = tile->nfibs[tt->nmodes - 2];

    double t = omp_get_wtime();
    idx_t temp_cnt = 0;

#pragma omp parallel for reduction(+:temp_cnt)
    for(idx_t f=0; f < nfibers; ++f) {
      if (0 == omp_get_thread_num() && f%max((size_t)(nfibers/omp_get_num_threads()/100.), 1UL) == 0) {
        printf("%ld%% f=%ld fibers=%ld nnz=%ld elapsed_time=%f\n", f/max((size_t)(nfibers/omp_get_num_threads()/100.), 1UL), f, fptr[f + 1] - fptr[f], temp_cnt, omp_get_wtime() - t);
        t = omp_get_wtime();
      }
      for(idx_t i=fptr[f]; i < fptr[f + 1]; ++i) {
        idx_t src = inds[i];
        omp_set_lock(locks + src);
        for(idx_t j=fptr[f]; j < fptr[f + 1]; ++j) {
          idx_t dst = inds[j];
          if (dst != src) {
            vector<int>::iterator itr = lower_bound(g[src].begin(), g[src].end(), dst);
            if (itr == g[src].end() || *itr != dst) {
              g[src].insert(itr, dst);
              ++temp_cnt;
            }
          }
        }
        omp_unset_lock(locks + src);
      } // for each nnz pair in fiber
    } // for each fiber

    t = omp_get_wtime();

    A.rowptr = new idx_t[A.m + 1];
    idx_t nnz = 0;
    for(int i=0; i < A.m; ++i) {
      A.rowptr[i] = nnz;
      nnz += g[i].size();
    }
    A.rowptr[A.m] = nnz;
    printf("%s:%d nnz=%ld nnz_per_row=%f %f\n", __FILE__, __LINE__, nnz, (double)nnz/A.m, omp_get_wtime() - t);
    t = omp_get_wtime();

    A.colidx = new int[nnz];
#pragma omp parallel for
    for(int i=0; i < A.m; ++i) {
      copy(g[i].begin(), g[i].end(), A.colidx + A.rowptr[i]);
    }
    g.clear();

    printf("%s:%d %f\n", __FILE__, __LINE__, omp_get_wtime() - t);
    t = omp_get_wtime();

    int *temp_perm = new int[A.m];
    int *temp_iperm = new int[A.m];
    if (use_rcm) {
      A.getRCMPermutation(temp_perm, temp_iperm);
    }
    else {
      A.getBFSPermutation(temp_perm, temp_iperm);
    }

    printf("%s:%d %f\n", __FILE__, __LINE__, omp_get_wtime() - t);
    t = omp_get_wtime();

#pragma omp parallel for
    for(int i=0; i < A.m; ++i) {
      perm->perms[mode][i] = temp_perm[i];
      perm->iperms[mode][i] = temp_iperm[i];
    }

    printf("%s:%d %f\n", __FILE__, __LINE__, omp_get_wtime() - t);

    delete[] temp_perm;
    delete[] temp_iperm;

    delete[] A.rowptr;
    delete[] A.colidx;
  }

  for (int i=0; i < max_dims; ++i) {
    omp_destroy_lock(locks + i);
  }

  delete[] locks;

#else // slice based bfs
  // inter-mode graph for m1->m2 is a bipartite graph from ith mode to jth mode
  // for each non-zero at (x_1, ..., x_m1, ..., x_m2, ..., x_m),
  // inter_mode_graphs[m1][m2] has an edge (x_m1, x_m2)
  CSR *inter_mode_graphs[tt->nmodes*tt->nmodes];

  for(int m1=0; m1 < tt->nmodes; ++m1) {
    for(int m2=0; m2 < tt->nmodes; ++m2) {
      if (m1 == m2) continue;

      idx_t nrows = tt->dims[m1], nnz = 0;
      vector<tbb::concurrent_unordered_set<idx_t> > g(nrows);

      double t = omp_get_wtime();
#pragma omp parallel for
      for(idx_t i=0; i < tt->nnz; ++i) {
        idx_t src = tt->ind[m1][i];
        idx_t dst = tt->ind[m2][i];
        g[src].insert(dst);
      }

      printf("%s:%d [%d,%d] nnz=%ld %f\n", __FILE__, __LINE__, m1, m2, nnz, omp_get_wtime() - t);
      t = omp_get_wtime();

      CSR *A = new CSR;
      inter_mode_graphs[m1*tt->nmodes + m2] = A;

      A->rowptr = new idx_t[nrows + 1];
      for(idx_t i=0; i < nrows; ++i) {
        A->rowptr[i] = nnz;
        nnz += g[i].size();
      }
      A->rowptr[nrows] = nnz;
      A->colidx = new int[nnz];

#pragma omp parallel for
      for(idx_t i=0; i < nrows; ++i) {
        int j = A->rowptr[i];
        for(tbb::concurrent_unordered_set<idx_t>::iterator itr=g[i].begin(); itr != g[i].end(); ++itr, ++j) {
          A->colidx[j] = *itr;
        }
        sort(A->colidx + A->rowptr[i], A->colidx + A->rowptr[i + 1]);
      }

      g.clear();

      printf("%s:%d [%d,%d] nnz=%ld %f\n", __FILE__, __LINE__, m1, m2, nnz, omp_get_wtime() - t);
    }
  }

  for(int m1=0; m1 < tt->nmodes; ++m1) {
    printf("%s:%d [%d]\n", __FILE__, __LINE__, m1);

    int nrows = tt->dims[m1];
    int *visited = new int[nrows];

#pragma omp parallel for
    for(int i=0; i < nrows; ++i) {
      visited[i] = 0;
    }

    vector<int> q_in, q_out;
    int cnt = 0;
    int ncomponents = 0;

    double t = omp_get_wtime();
    for(idx_t i=0; i < nrows; ++i) {
      if (visited[i]) continue;

      int nlevel = 0;
      q_in.push_back(i);
      visited[i] = 1;

      int old_cnt = cnt;

      while (!q_in.empty()) {
        for(vector<int>::const_iterator itr=q_in.begin(); itr != q_in.end(); ++itr) {
          int u = *itr;

          perm->perms[m1][u] = cnt;
          ++cnt;
          if (cnt%max((size_t)(nrows/100.), 1UL) == 0) {
            printf("%ld%% level=%d elapsed_time=%f\n", cnt/max((size_t)(nrows/100.), 1UL), nlevel, omp_get_wtime() - t);
            t = omp_get_wtime();
          }

          for(int m2=0; m2 < tt->nmodes; ++m2) {
            if (m1 == m2) continue;

            CSR *g_out = inter_mode_graphs[m1*tt->nmodes + m2];
            CSR *g_in = inter_mode_graphs[m2*tt->nmodes + m1];

            for(idx_t j=g_out->rowptr[i]; j < g_out->rowptr[i + 1]; ++j) {
              int c = g_out->colidx[j];
              for(idx_t k=g_in->rowptr[c]; k < g_in->rowptr[c + 1]; ++k) {
                int v = g_in->colidx[k];
                if (!visited[v]) {
                  //printf("[%d->%d] %d->%d->%d\n", m1, m2, u, c, v);
                  q_out.push_back(v);
                  visited[v] = 1;
                }
              }
            }
          }
        } // for each visited vertex u

        printf("component %d level %d has %ld rows out of %d\n", ncomponents, nlevel, q_in.size(), nrows);

        q_in.swap(q_out);
        q_out.clear();

        ++nlevel;
      }

      printf("component %d has %d levels %d rows out of %d\n", ncomponents, nlevel, cnt - old_cnt, nrows);
      ++ncomponents;
    } // for each connected component

    assert(cnt == nrows);

    for(int i=0; i < nrows; ++i) {
      perm->iperms[m1][perm->perms[m1][i]] = i;
      if (perm->iperms[m1][i] != i) {
        printf("!!!%d->%ld\n", i, perm->iperms[m1][i]);
      }
    }
  }

  for(int m1=0; m1 < tt->nmodes; ++m1) {
    for(int m2=0; m2 < tt->nmodes; ++m2) {
      if (m1 == m2) continue;

      delete[] inter_mode_graphs[m1*tt->nmodes + m2]->rowptr;
      delete[] inter_mode_graphs[m1*tt->nmodes + m2]->colidx;
      delete inter_mode_graphs[m1*tt->nmodes + m2];
    }
  }
#endif // slice based bfs

  perm_apply(tt, perm->perms);
#endif

  return perm;
}

permutation_t *perm_bfs(sptensor_t * const tt)
{
  return perm_bfs_or_rcm(tt, 0);
}

permutation_t *perm_rcm(sptensor_t * const tt)
{
  return perm_bfs_or_rcm(tt, 1);
}
