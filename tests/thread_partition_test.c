
#include "../src/thread_partition.h"
#include "../src/util.h"
#include "../src/sort.h"

#include "ctest/ctest.h"

#include "splatt_test.h"

#define NUM_PARTITION_TESTS 7


bool lprobe(
    idx_t const * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts,
    idx_t const bottleneck);


CTEST_DATA(partition)
{
  idx_t P;
  idx_t N;
  idx_t * unit_data;
  idx_t * rand_data;
  idx_t * sorted_data;
  idx_t * fororder_data;
  idx_t * revorder_data;
  idx_t * bigend_data;
  idx_t * fibonacci_data;

  idx_t * ptrs[NUM_PARTITION_TESTS];
};


CTEST_SETUP(partition)
{
  data->P = 31;

  data->N = 500;
  data->rand_data = malloc(data->N * sizeof(*(data->rand_data)));
  data->sorted_data = malloc(data->N * sizeof(*(data->sorted_data)));
  data->fororder_data = malloc(data->N * sizeof(*(data->fororder_data)));
  data->revorder_data = malloc(data->N * sizeof(*(data->revorder_data)));
  data->bigend_data = malloc(data->N * sizeof(*(data->bigend_data)));
  data->unit_data = malloc(data->N * sizeof(*(data->unit_data)));
  data->fibonacci_data = malloc(data->N * sizeof(*(data->fibonacci_data)));

  for(idx_t x=0; x < data->N; ++x) {
    data->unit_data[x] = 1;
    data->rand_data[x] = rand_idx() % 131;
    data->sorted_data[x] = rand_idx() % 131;
    data->bigend_data[x] = rand_idx() % 131;

    data->fororder_data[x] = x;
    data->revorder_data[x] = data->N - x;

    if(x < 2) {
      data->fibonacci_data[x] = 1;
    } else {
      if(x < 10) {
        data->fibonacci_data[x] = data->fibonacci_data[x-1] +
            data->fibonacci_data[x-2];
      } else {
        data->fibonacci_data[x] = data->fibonacci_data[x-1] + 1000;
      }
    }
  }


  splatt_quicksort(data->sorted_data, data->N);
  data->bigend_data[data->N - 1] = 999;

  data->ptrs[0] = data->rand_data;
  data->ptrs[1] = data->sorted_data;
  data->ptrs[2] = data->fororder_data;
  data->ptrs[3] = data->revorder_data;
  data->ptrs[4] = data->bigend_data;
  data->ptrs[5] = data->unit_data;
  data->ptrs[6] = data->fibonacci_data;
}

CTEST_TEARDOWN(partition)
{
  for(idx_t t=0; t < NUM_PARTITION_TESTS; ++t) {
    free(data->ptrs[t]);
  }
}


CTEST2(partition, prefix_sum_inc)
{
  idx_t * pref = malloc(data->N * sizeof(*pref));

  for(idx_t t=0; t < NUM_PARTITION_TESTS; ++t) {
    idx_t * const restrict weights = data->ptrs[t];

    idx_t total = 0;
    for(idx_t x=0; x < data->N; ++x) {
      total += weights[x];
    }

    /* make a copy */
    memcpy(pref, weights, data->N * sizeof(*pref));

    prefix_sum_inc(pref, data->N);

    ASSERT_EQUAL(total, pref[data->N - 1]);

    idx_t running = 0;
    for(idx_t x=0; x < data->N; ++x) {
      running += weights[x];
      ASSERT_EQUAL(running, pref[x]);
    }
  }
  free(pref);
}


CTEST2(partition, prefix_sum_exc)
{
  /* make a copy */
  idx_t * pref = malloc(data->N * sizeof(*pref));

  /* foreach test */
  for(idx_t t=0; t < NUM_PARTITION_TESTS; ++t) {
    idx_t * const restrict weights = data->ptrs[t];
    memcpy(pref, weights,  data->N * sizeof(*pref));

    prefix_sum_exc(pref, data->N);

    idx_t running = 0;
    for(idx_t x=0; x < data->N; ++x) {
      ASSERT_EQUAL(running, pref[x]);
      running += weights[x];
    }
  }

  free(pref);
}


CTEST2(partition, partition_weighted)
{
  /* foreach test */
  for(idx_t t=0; t < NUM_PARTITION_TESTS; ++t) {
    idx_t * const restrict weights = data->ptrs[t];

    idx_t bneck;
    idx_t * parts = partition_weighted(weights, data->N, data->P, &bneck);

    /* check bounds */
    ASSERT_EQUAL(0, parts[0]);
    ASSERT_EQUAL(data->N, parts[data->P]);

    /* check non-overlapping partitions */
    for(idx_t p=0; p < data->P; ++p) {
      /* if N < P, someone will have no work */
      if(parts[p] > parts[p+1]) {
        ASSERT_FAIL();
      }
    }

    /* check that bneck is not surpassed */
    for(idx_t p=0; p < data->P; ++p) {
      idx_t const left = SS_MIN(parts[p], data->N-1);
      /* -1 because exclusive bound */
      idx_t const right = SS_MIN(parts[p+1]-1, data->N-1);
      if(weights[right] - weights[left] > bneck) {
        ASSERT_FAIL();
      }
    }

    /* check actual optimality */
    bool success;
    success = lprobe(weights, data->N, parts, data->P, bneck);
    ASSERT_EQUAL(true, success);
    success = lprobe(weights, data->N, parts, data->P, bneck-1);
    ASSERT_EQUAL(false, success);

    splatt_free(parts);
  } /* end foreach test */
}


CTEST2(partition, probe)
{
  idx_t total = 0;
  for(idx_t x=0; x < data->N; ++x) {
    total += data->rand_data[x];
  }

  prefix_sum_exc(data->rand_data, data->N);

  idx_t * parts = splatt_malloc((data->P+1) * sizeof(*parts));

  bool result = lprobe(data->rand_data, data->N, parts, data->P,
      (total / data->P) - 1);
  ASSERT_EQUAL(false, result);

  /* find optimal */
  idx_t bottleneck = total / data->P;
  while(!result) {
    result = lprobe(data->rand_data, data->N, parts, data->P, bottleneck);
    ++bottleneck;
  }
  --bottleneck;

  /* check bounds */
  ASSERT_EQUAL(0, parts[0]);
  ASSERT_EQUAL(data->N, parts[data->P]);

  /* check non-overlapping partitions */
  for(idx_t p=1; p < data->P; ++p) {
    /* if N < P, someone will have no work */
    if(parts[p] <= parts[p-1]) {
      ASSERT_FAIL();
    }
  }

  /* check actual bneck */
  for(idx_t p=1; p < data->P; ++p) {
    /* if N < P, someone will have no work */
    if(parts[p] - parts[p-1] > bottleneck) {
      ASSERT_FAIL();
    }
  }
  splatt_free(parts);
}


CTEST2(partition, bigpart)
{
  idx_t const N = 25000000;
  idx_t const P = 24;

  idx_t * weights = splatt_malloc(N * sizeof(*weights));

  for(idx_t x=0; x < N; ++x) {
    weights[x] = rand_idx() % 100;
  }

  sp_timer_t part;
  timer_fstart(&part);
  idx_t bneck; 
  idx_t * parts = partition_weighted(weights, N, P, &bneck);
  timer_stop(&part);

  /* correctness */
  bool success;
  success = lprobe(weights, N, parts, P, bneck);
  ASSERT_EQUAL(true, success);
  success = lprobe(weights, N, parts, P, bneck-1);
  ASSERT_EQUAL(false, success);

  splatt_free(weights);
  splatt_free(parts);
}


CTEST2(partition, part_equalsize)
{
  idx_t const P = 24;
  idx_t const CHUNK = 10000;
  idx_t const N = P * CHUNK;

  idx_t * weights = splatt_malloc(N * sizeof(*weights));

  for(idx_t x=0; x < N; ++x) {
    weights[x] = 1;
  }

  idx_t bneck;
  idx_t * parts = partition_weighted(weights, N, P, &bneck);

  ASSERT_EQUAL(CHUNK, bneck);

  lprobe(weights, N, parts, P, bneck);

  for(idx_t p=0; p < P; ++p) {
    ASSERT_EQUAL(CHUNK * p, parts[p]);
  }
  ASSERT_EQUAL(N, parts[P]);

  splatt_free(weights);
  splatt_free(parts);
}


CTEST2(partition, part_simple)
{
  idx_t * parts;

  /* assign 1 to everyone */
  parts = partition_simple(4, 4);
  ASSERT_EQUAL(0, parts[0]);
  ASSERT_EQUAL(1, parts[1]);
  ASSERT_EQUAL(2, parts[2]);
  ASSERT_EQUAL(3, parts[3]);
  ASSERT_EQUAL(4, parts[4]);
  splatt_free(parts);

  /* assign 1 to everyone, 2 to last */
  parts = partition_simple(5, 4);
  ASSERT_EQUAL(0, parts[0]);
  ASSERT_EQUAL(1, parts[1]);
  ASSERT_EQUAL(2, parts[2]);
  ASSERT_EQUAL(3, parts[3]);
  ASSERT_EQUAL(5, parts[4]);
  splatt_free(parts);

  /* assign 2 to everyone */
  parts = partition_simple(8, 4);
  ASSERT_EQUAL(0, parts[0]);
  ASSERT_EQUAL(2, parts[1]);
  ASSERT_EQUAL(4, parts[2]);
  ASSERT_EQUAL(6, parts[3]);
  ASSERT_EQUAL(8, parts[4]);
  splatt_free(parts);
  
  /* primes */
  parts = partition_simple(7, 3);
  ASSERT_EQUAL(0, parts[0]);
  ASSERT_EQUAL(2, parts[1]);
  ASSERT_EQUAL(4, parts[2]);
  ASSERT_EQUAL(7, parts[3]);
  splatt_free(parts);

  /* fewer items than parts */
  parts = partition_simple(2, 3);
  ASSERT_EQUAL(0, parts[0]);
  ASSERT_EQUAL(1, parts[1]);
  ASSERT_EQUAL(2, parts[2]);
  ASSERT_EQUAL(2, parts[3]);
  splatt_free(parts);
}


