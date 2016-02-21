#include <algorithm>
#include <deque>
#include <set>
#include <map>

#include <omp.h>

#include "reorder.h"
#include "csf.h"
#include "stats.h"

#include "Utils.hpp"

using namespace std;
using namespace SpMP;

int *parallel_max_element(int *begin, int *end)
{
  idx_t local_max_indices[omp_get_max_threads()];
  int *ret = end;

#pragma omp parallel
  {
    int n = end - begin;
    int i_begin, i_end;
    getSimpleThreadPartition(&i_begin, &i_end, n);

    local_max_indices[omp_get_thread_num()] =
      max_element(begin + i_begin, begin + i_end) - begin;

#pragma omp barrier
#pragma omp master
    {
      ret = begin + local_max_indices[0];
      idx_t maximum = *ret;
      for(int i = 1; i < omp_get_num_threads(); ++i) {
        if (begin[local_max_indices[i]] > maximum) {
          ret = begin + local_max_indices[i];
          maximum = *ret;
        }
      }
    }
  }

  //assert(ret == max_element(begin, end));
  return ret;
}

double hist_init_time = 0;
double hist_main_time = 0;
double hist_reduce_time = 0;

void populate_histogram(int *hist, const splatt_csf *csf, int m1, idx_t s)
{
  hist_init_time -= omp_get_wtime();

#pragma omp parallel for
  for(idx_t i = 0; i < csf->dims[m1]; ++i) {
    hist[i] = 0;
  }

  hist_init_time += omp_get_wtime();
  hist_main_time -= omp_get_wtime();

  // for each other mode
  for(int m2 = 0; m2 < 3; ++m2) {
    if (m1 == m2) continue;

    csf_sparsity *tile1 = NULL, *tile2 = NULL;
    for(int i = 0; i < 6; ++i) {
      if (csf[i].dim_perm[0] == m1 && csf[i].dim_perm[1] == m2) {
        assert(!tile1);
        tile1 = csf[i].pt;
      }
      if (csf[i].dim_perm[0] == m2 && csf[i].dim_perm[1] == m1) {
        assert(!tile2);
        tile2 = csf[i].pt;
      }
    }
    assert(tile1 && tile2);

    assert(!tile1->fids[0] && !tile2->fids[0]);

    idx_t *sptr1 = tile1->fptr[0];
    idx_t *fids1 = tile1->fids[1];
    idx_t *sptr2 = tile2->fptr[0];
    idx_t *fids2 = tile2->fids[1];

    // for each fiber (s, fids1[f1], *)
#pragma omp parallel for
    for(idx_t f1 = sptr1[s]; f1 < sptr1[s+1]; ++f1) {
      // for each fiber (fids2[f2], fids1[f1], *)
      for(idx_t f2 = sptr2[fids1[f1]]; f2 < sptr2[fids1[f1]+1]; ++f2) {
        __sync_fetch_and_add(hist + fids2[f2], 1);
//#pragma omp atomic
        //++hist[fids2[f2]];
      }
    }
  } // for each other mode

  hist_main_time += omp_get_wtime();
}

void populate_histogram_sample(int *hist, const splatt_csf *csf, int m1, idx_t s)
{
  hist_init_time -= omp_get_wtime();

#pragma omp parallel for
  for(idx_t i = 0; i < csf->dims[m1]; ++i) {
    hist[i] = 0;
  }

  hist_init_time += omp_get_wtime();
  hist_main_time -= omp_get_wtime();

  map<int, int> hist_map;

  // count # of fibers
  int total_cnt = 0;
  int cnts[3] = { 0 };
  for(int m2 = 0; m2 < 3; ++m2) {
    if (m1 == m2) continue;

    csf_sparsity *tile1 = NULL, *tile2 = NULL;
    for(int i = 0; i < 6; ++i) {
      if (csf[i].dim_perm[0] == m1 && csf[i].dim_perm[1] == m2) {
        assert(!tile1);
        tile1 = csf[i].pt;
      }
      if (csf[i].dim_perm[0] == m2 && csf[i].dim_perm[1] == m1) {
        assert(!tile2);
        tile2 = csf[i].pt;
      }
    }
    assert(tile1 && tile2);

    assert(!tile1->fids[0] && !tile2->fids[0]);

    idx_t *sptr1 = tile1->fptr[0];
    idx_t *fids1 = tile1->fids[1];
    idx_t *sptr2 = tile2->fptr[0];

    for(int f = sptr1[s]; f < sptr1[s+1]; ++f) {
      cnts[m2] += sptr2[fids1[f]+1] - sptr2[fids1[f]];
    }
    total_cnt += cnts[m2];
  }

  // distribute # of samples needed for each mode
  long long NSAMPLES = 65536LL;
  int samples[3] = { 0 };
  for(int m2 = 0; m2 < 3; ++m2) {
    if (m1 == m2) continue;
    if (m2 == 2 || m2 == 1 && m1 == 2) {
      samples[m2] = NSAMPLES;
      for (int m3 = 0; m3 < m2; ++m3) {
        samples[m2] -= samples[m3];
      }
    }
    else {
      samples[m2] = NSAMPLES*cnts[m2]/total_cnt;
      assert(samples[m2] >= 0);
    }
    printf("mode %d: %d samples out of %d\n", m2, samples[m2], cnts[m2]);
  }

  // for each other mode
  for(int m2 = 1; m2 < 3; ++m2) {
    if (m1 == m2) continue;

    csf_sparsity *tile1 = NULL, *tile2 = NULL;
    for(int i = 0; i < 6; ++i) {
      if (csf[i].dim_perm[0] == m1 && csf[i].dim_perm[1] == m2) {
        assert(!tile1);
        tile1 = csf[i].pt;
      }
      if (csf[i].dim_perm[0] == m2 && csf[i].dim_perm[1] == m1) {
        assert(!tile2);
        tile2 = csf[i].pt;
      }
    }
    assert(tile1 && tile2);

    assert(!tile1->fids[0] && !tile2->fids[0]);

    idx_t *sptr1 = tile1->fptr[0];
    idx_t *fids1 = tile1->fids[1];
    idx_t *sptr2 = tile2->fptr[0];
    idx_t *fids2 = tile2->fids[1];

    // for each fiber (s, fids1[f1], *)
#pragma omp parallel for
    for(idx_t f1 = sptr1[s]; f1 < sptr1[s+1]; ++f1) {
      // for each fiber (fids2[f2], fids1[f1], *)
      for(idx_t f2 = sptr2[fids1[f1]]; f2 < sptr2[fids1[f1]+1]; ++f2) {
        __sync_fetch_and_add(hist + fids2[f2], 1);
      }
    }

    int nfibers1 = sptr1[s+1] - sptr1[s];
    int *prefix_sum = new int[nfibers1 + 1];
    prefix_sum[0] = 0;
    for(int f1 = sptr1[s]; f1 < sptr1[s+1]; ++f1) {
      prefix_sum[f1 + 1 - sptr1[s]] = prefix_sum[f1 - sptr1[s]] + sptr2[fids1[f1]+1] - sptr2[fids1[f1]];
    }

    assert(prefix_sum[nfibers1] == cnts[m2]);

    for (int i = 0; i < samples[m2]; ++i) {
      int idx = rand()%cnts[m2];
      idx_t f1 = upper_bound(prefix_sum, prefix_sum + nfibers1, idx) - prefix_sum - 1;
      assert(idx >= prefix_sum[f1] && idx < prefix_sum[f1 + 1]);
      int f2 = idx - prefix_sum[f1] + sptr2[fids1[f1 + sptr1[s]]];
      int bin = fids2[f2];
      if (hist_map.find(bin) == hist_map.end()) {
        hist_map[bin] = 0;
      }
      ++hist_map[bin];
    }
    delete[] prefix_sum;
  } // for each other mode

  hist_main_time += omp_get_wtime();

  int *max_ptr = parallel_max_element(hist, hist + csf->dims[m1]);

  int maximum = -1;
  int max_idx = -1;
  for (map<int, int>::iterator itr = hist_map.begin(); itr != hist_map.end(); ++itr) {
    if (itr->second > maximum) {
      max_idx = itr->first;
      maximum = itr->second;
    }
  }

  int rank = 0;
  for (int i = 0; i < csf->dims[m1]; ++i) {
    if (hist[i] > hist[max_idx]) ++rank;
  }
  printf("max = %d at %ld, sampled_max = %lld (real cnt = %d, rank = %d) at %d\n", *max_ptr, max_ptr - hist, maximum*total_cnt/NSAMPLES, hist[max_idx], rank, max_idx);
}

/* execution time for delicious with varying cur_max_bucket */
// cur_max_bucket = 1024
// 0% 1726 elapsed_time=76.529988 (histogram total 71.360496 avg 0.041344 hist_init 0.013766 hist_main 3.412220 hist_reduce 67.932765)
// cur_max_bucket = 2048
// 0% 1726 elapsed_time=69.571672 (histogram total 65.131212 avg 0.037713 hist_init 0.017320 hist_main 3.415375 hist_reduce 61.696797)
// cur_max_bucket = 4096
// 0% 1726 elapsed_time=65.628014 (histogram total 61.515216 avg 0.035620 hist_init 0.013407 hist_main 3.416017 hist_reduce 58.083945)
// cur_max_bucket = 8192
// 0% 1726 elapsed_time=62.120921 (histogram total 58.097406 avg 0.033641 hist_init 0.014290 hist_main 3.452460 hist_reduce 54.628824)
// cur_max_bucket = 16384
// 0% 1726 elapsed_time=58.669855 (histogram total 54.646743 avg 0.031643 hist_init 0.014900 hist_main 3.483956 hist_reduce 51.146099)
// cur_max_bucket = 32768
// 0% 1726 elapsed_time=54.783594 (histogram total 50.918607 avg 0.029484 hist_init 0.017119 hist_main 3.501318 hist_reduce 47.398419)
// cur_max_bucket = 65536
// 0% 1726 elapsed_time=50.854705 (histogram total 46.928303 avg 0.027173 hist_init 0.019171 hist_main 3.652035 hist_reduce 43.255368)
// cur_max_bucket = 131072
// 0% 1726 elapsed_time=46.915687 (histogram total 42.958090 avg 0.024874 hist_init 0.024755 hist_main 3.901233 hist_reduce 39.030430)
// cur_max_bucket = 262144
// 0% 1726 elapsed_time=42.426353 (histogram total 38.591844 avg 0.022346 hist_init 0.038141 hist_main 4.289968 hist_reduce 34.262135)
// cur_max_bucket = 524288
// 0% 1726 elapsed_time=37.761628 (histogram total 34.013813 avg 0.019695 hist_init 0.064410 hist_main 4.658353 hist_reduce 29.289435)
// cur_max_bucket = 1048576
// 0% 1726 elapsed_time=34.294597 (histogram total 30.453448 avg 0.017634 hist_init 0.122887 hist_main 4.954820 hist_reduce 25.374180)
// cur_max_bucket = 2097152
// 0% 1726 elapsed_time=32.567633 (histogram total 28.654423 avg 0.016592 hist_init 0.250997 hist_main 5.052103 hist_reduce 23.349777)
// cur_max_bucket = 4194304
// 0% 1726 elapsed_time=32.962280 (histogram total 29.192228 avg 0.016903 hist_init 0.429426 hist_main 5.105814 hist_reduce 23.655401)
// cur_max_bucket = 8388608
// 0% 1726 elapsed_time=37.704083 (histogram total 33.866515 avg 0.019610 hist_init 0.717942 hist_main 5.268225 hist_reduce 27.878736)
// cur_max_bucket = 16777216
// 0% 1726 elapsed_time=48.957511 (histogram total 45.162274 avg 0.026151 hist_init 1.635134 hist_main 5.248173 hist_reduce 38.277328)
// cur_max_bucket = 33554432
// 0% 1726 elapsed_time=72.396652 (histogram total 68.615778 avg 0.039731 hist_init 3.665499 hist_main 5.776651 hist_reduce 59.171651)
// cur_max_bucket = 67108864
// 0% 1726 elapsed_time=72.489443 (histogram total 68.676843 avg 0.039767 hist_init 3.414763 hist_main 5.732942 hist_reduce 59.527138)
// cur_max_bucket = 134217728
// 0% 1726 elapsed_time=72.638989 (histogram total 68.785730 avg 0.039830 hist_init 3.221596 hist_main 5.748549 hist_reduce 59.813431)
// cur_max_bucket = 268435456
// 0% 1726 elapsed_time=72.351387 (histogram total 68.515841 avg 0.039673 hist_init 2.897416 hist_main 5.784604 hist_reduce 59.831583)
// cur_max_bucket = 536870912
// 0% 1726 elapsed_time=72.113347 (histogram total 68.492705 avg 0.039660 hist_init 2.761415 hist_main 5.818204 hist_reduce 59.911029)
// cur_max_bucket = 1073741824
// 0% 1726 elapsed_time=73.770896 (histogram total 70.018840 avg 0.040544 hist_init 3.240197 hist_main 5.762401 hist_reduce 61.013997)
// cur_max_bucket = -2147483648
// 0% 1726 elapsed_time=74.219666 (histogram total 70.493101 avg 0.040818 hist_init 3.319347 hist_main 5.769216 hist_reduce 61.402342)

static const int MAX_BUCKETS = 1<<30;
int *buckets;
int cur_max_bucket = MAX_BUCKETS;

void populate_histogram_reduction(int *hist, const splatt_csf *csf, int m1, idx_t s)
{
  hist_init_time -= omp_get_wtime();

  /*int *temp_hist = new int[csf->dims[m1]];

#pragma omp parallel for
  for(idx_t i = 0; i < csf->dims[m1]; ++i) {
    temp_hist[i] = 0;
  }*/

  // each bucket size is (1 << p)
  int p = ceil(log2((csf->dims[m1] + cur_max_bucket - 1)/cur_max_bucket));
  // # of buckets used
  int nbuckets = (csf->dims[m1] + (1 << p) - 1)/(1 << p);
  assert(nbuckets <= cur_max_bucket);

#pragma omp parallel for
  for(int i = 0; i < nbuckets; ++i) {
    buckets[i] = 0;
  }
  hist_init_time += omp_get_wtime();
  hist_main_time -= omp_get_wtime();

  // for each other mode
  for(int m2 = 0; m2 < 3; ++m2) {
    if (m1 == m2) continue;

    csf_sparsity *tile1 = NULL, *tile2 = NULL;
    for(int i = 0; i < 6; ++i) {
      if (csf[i].dim_perm[0] == m1 && csf[i].dim_perm[1] == m2) {
        assert(!tile1);
        tile1 = csf[i].pt;
      }
      if (csf[i].dim_perm[0] == m2 && csf[i].dim_perm[1] == m1) {
        assert(!tile2);
        tile2 = csf[i].pt;
      }
    }
    assert(tile1 && tile2);

    assert(!tile1->fids[0] && !tile2->fids[0]);

    idx_t *sptr1 = tile1->fptr[0];
    idx_t *fids1 = tile1->fids[1];
    idx_t *sptr2 = tile2->fptr[0];
    idx_t *fids2 = tile2->fids[1];

    /*int cnt = 0;
    for(idx_t f1 = sptr1[s]; f1 < sptr1[s+1]; ++f1) {
      cnt += sptr2[fids1[f1]+1] - sptr2[fids1[f1]];
    }
    printf("%d %ld\n", cnt, csf->dims[m1]);*/

    // for each fiber (s, fids1[f1], *)
#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      int *hist_private = hist + tid*csf->dims[m1];

#pragma omp for
      for(idx_t f1 = sptr1[s]; f1 < sptr1[s+1]; ++f1) {
        // for each fiber (fids2[f2], fids1[f1], *)
        for(idx_t f2 = sptr2[fids1[f1]]; f2 < sptr2[fids1[f1]+1]; ++f2) {
          //assert(fids2[f2] < csf->dims[m1]);
          idx_t idx = fids2[f2];
          int bid = idx >> p;
          //assert(bid < nbuckets);
          if (!buckets[bid]) buckets[bid] = 1;
//#pragma omp atomic
          ++hist_private[idx];
        }
      }
    } // omp parallel

/*#pragma omp parallel for
    for(idx_t f1 = sptr1[s]; f1 < sptr1[s+1]; ++f1) {
      // for each fiber (fids2[f2], fids1[f1], *)
      for(idx_t f2 = sptr2[fids1[f1]]; f2 < sptr2[fids1[f1]+1]; ++f2) {
#pragma omp atomic
        ++temp_hist[fids2[f2]];
      }
    }*/
  } // for each other mode

  hist_main_time += omp_get_wtime();
  hist_reduce_time -= omp_get_wtime();

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    int used_bucket_cnt = 0;
    for(int bid = 0; bid < nbuckets; ++bid) {
      if(buckets[bid]) ++used_bucket_cnt;
    }

    //if (0 == tid) printf("used_bucket_cnt = %d out of %d\n", used_bucket_cnt, nbuckets);

    int bucket_per_thread = (used_bucket_cnt + nthreads - 1)/nthreads;
    int bucket_begin = min(bucket_per_thread*tid, nbuckets);
    int bucket_end = min(bucket_begin + bucket_per_thread, nbuckets);

    used_bucket_cnt = 0;
    for(int bid = 0; bid < nbuckets; ++bid) {
      if(buckets[bid]) {
        if(used_bucket_cnt >= bucket_begin && used_bucket_cnt < bucket_end) {
          int j_begin = min(bid << p, (int)csf->dims[m1]);
          int j_end = min((bid + 1) << p, (int)csf->dims[m1]);
          for(int j = j_begin; j < j_end; ++j) {
            for(int t = 1; t < nthreads; ++t) {
              hist[j] += hist[t*csf->dims[m1] + j];
              hist[t*csf->dims[m1] + j] = 0;
            }
          }
        }
        ++used_bucket_cnt;
      }
    }
  }

  /*for(int i = 0; i < csf->dims[m1]; ++i) {
    assert(hist[i] == temp_hist[i]);
    hist[i] = 0;
  }*/

  hist_reduce_time += omp_get_wtime();

  /*int cnt = 0;
  for(idx_t i = 0; i < csf->dims[m1]; ++i) {
    if (hist[i]) ++cnt;
  }
  printf("cnt = %d\n", cnt);*/

  //delete[] temp_hist;
}

permutation_t *perm_matching(sptensor_t * const tt)
{
  double *opts = splatt_default_opts();
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLPERMUTE;
  bool write = false;
  splatt_csf *csf;
  if (write) {
    csf = splatt_csf_alloc(tt, opts);
    splatt_csf_write(csf, "/home/jpark103/tensor/delicious.6.csf", 6);
  }
  else {
    csf = (splatt_csf *)malloc(sizeof(splatt_csf)*6);
    double read_time = omp_get_wtime();
    splatt_csf_read(csf, "/home/jpark103/tensor/delicious.6.csf", 6);
    printf("splatt_csf_read takes %f\n", omp_get_wtime() - read_time);

    stats_csf(csf, 1);
  }

  idx_t max_dims = 0;
  for(int m=0; m < csf->nmodes; ++m) {
    max_dims = max(max_dims, csf->dims[m]);
  }
  int *hist = new int[max_dims];

  pair<idx_t, idx_t> *sorted_by_connection_temp = new pair<idx_t, idx_t>[max_dims];
  idx_t *sorted_by_connection = new idx_t[max_dims];

  permutation_t *perm = perm_alloc(csf->dims, csf->nmodes);

  for(int m1 = 0; m1 < 3; ++m1) { // for each mode
    printf("\n\n\n--- working on mode %d ---\n\n", m1);

    // (head, first) is the pair with the biggest overlap
    // (head, second) is the pair with the second biggest overlap starting from head
    idx_t first = 0, first_idx = 0, second = 0, second_idx = 0, head = 0;
    double t = omp_get_wtime();
    double histogram_time = 0;
    int histogram_cnt = 0;

    for(idx_t s = 0; s < csf->dims[m1]; ++s) { // for each slice of mode m1
      if (s > 0 && s%max((size_t)(csf->dims[m1]/10.), 1UL) == 0) {
        printf("%ld%% %ld elapsed_time=%f (histogram total %f avg %f hist_init %f hist_main %f)\n", s/max((size_t)(csf->dims[m1]/100.), 1UL), s, omp_get_wtime() - t, histogram_time, histogram_time/histogram_cnt, hist_init_time, hist_main_time);

        t = omp_get_wtime();
        histogram_time = 0;
        histogram_cnt = 0;
        hist_init_time = 0;
        hist_main_time = 0;
        hist_reduce_time = 0;
      }

      histogram_time -= omp_get_wtime();
      populate_histogram(hist, csf, m1, s);
      histogram_time += omp_get_wtime();
      ++histogram_cnt;

      hist[s] = 0;

      idx_t local_first_idx = parallel_max_element(hist, hist + csf->dims[m1]) - hist;
      idx_t local_first = hist[local_first_idx];

      if (local_first > first) {
        head = s;

        first = local_first;
        first_idx = local_first_idx;

        hist[first_idx] = 0;
        second_idx = parallel_max_element(hist, hist + csf->dims[m1]) - hist;
        second = hist[second_idx];
        if (second_idx == first_idx) {
          second_idx = csf->dims[m1];
        }
      }

      sorted_by_connection_temp[s] = make_pair(local_first_idx, s);
    } // for each slice of first mode

    t = omp_get_wtime();
    sort(sorted_by_connection_temp, sorted_by_connection_temp + csf->dims[m1]);
    reverse(sorted_by_connection_temp, sorted_by_connection_temp + csf->dims[m1]);
    printf("sorting takes %f\n", omp_get_wtime() - t);

#pragma omp parallel for
    for(idx_t i = 0; i < csf->dims[m1]; ++i) {
      sorted_by_connection[i] = sorted_by_connection_temp[i].second;
    }

    map<idx_t, idx_t> unvisited;
    for(idx_t i = 0; i < csf->dims[m1]; ++i) {
      unvisited[sorted_by_connection[i]] = i;
    }

    deque<idx_t> dq;

    idx_t tail = first_idx;

    dq.push_back(head);
    dq.push_back(tail);

    idx_t head_next = second_idx;
    idx_t tail_next = csf->dims[m1];

    sorted_by_connection[unvisited[head]] = csf->dims[m1];
    sorted_by_connection[unvisited[tail]] = csf->dims[m1];
    sorted_by_connection[unvisited[head_next]] = csf->dims[m1];
    idx_t sort_idx = 0;

    unvisited.erase(head);
    unvisited.erase(tail);
    unvisited.erase(head_next);

    // head_next - head - tail - (tail_next)

    t = omp_get_wtime();
    histogram_time = 0;
    histogram_cnt = 0;

    while (dq.size() < csf->dims[m1]) {
      assert(head_next == csf->dims[m1] ^ tail_next == csf->dims[m1]);
      if (dq.size()%max((size_t)(csf->dims[m1]/10.), 1UL) == 0) {
        printf("%ld%% %ld elapsed_time=%f (histogram total %f avg %f)\n", dq.size()/max((size_t)(csf->dims[m1]/100.), 1UL), dq.size(), omp_get_wtime() - t, histogram_time, histogram_time/histogram_cnt);
        t = omp_get_wtime();
        histogram_time = 0;
        histogram_cnt = 0;
      }

      head = dq.front();
      tail = dq.back();

      idx_t s = head_next == csf->dims[m1] ? head : tail;

      histogram_time -= omp_get_wtime();
      populate_histogram(hist, csf, m1, s);
      histogram_time += omp_get_wtime();
      ++histogram_cnt;

      deque<idx_t>::iterator itr;
      for (itr = dq.begin(); itr != dq.end(); ++itr) {
        hist[*itr] = 0;
      }
      if (head_next == csf->dims[m1]) {
        hist[tail_next] = 0;
      }
      else {
        hist[head_next] = 0;
      }

      idx_t next = parallel_max_element(hist, hist + csf->dims[m1]) - hist;
      if (0 == hist[next] && dq.size() < csf->dims[m1] - 1) {
        for ( ; sorted_by_connection[sort_idx] == csf->dims[m1]; sort_idx++);
        next = sorted_by_connection[sort_idx];
        hist[next] = 0;
        printf("s=%ld no connection from here\n", s);
      }

      if (hist[next] > second) {
        if (head_next == csf->dims[m1]) {
          dq.push_front(next);
        }
        else {
          dq.push_back(next);
        }
      } else {
        if (head_next == csf->dims[m1]) {
          dq.push_back(tail_next);

          tail_next = csf->dims[m1];
          head_next = next;
        }
        else {
          dq.push_front(head_next);

          head_next = csf->dims[m1];
          tail_next = next;
        }
        second = hist[next];
      }
      
      if (dq.size() < csf->dims[m1]) {
        assert(unvisited.find(next) != unvisited.end());
        sorted_by_connection[unvisited[next]] = csf->dims[m1];
        unvisited.erase(next);
      }
    }

    deque<idx_t>::iterator itr;
    idx_t i = 0;
    for(itr = dq.begin(); itr != dq.end(); ++itr, ++i) {
      perm->perms[m1][*itr] = i;
      perm->iperms[m1][i] = *itr;
    }

#ifndef NDEBUG
    int *temp_perm = new int[csf->dims[m1]];
    for(int i = 0; i < csf->dims[m1]; ++i) {
      temp_perm[i] = perm->perms[m1][i];
    }
    assert(isPerm(temp_perm, csf->dims[m1]));
#endif
  } // for each mode

  delete[] hist;

  return perm;
}
