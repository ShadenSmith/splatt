/**
Copyright (c) 2015, Intel Corporation. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Intel Corporation nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL INTEL CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstring>
#include <algorithm>

#include "CSR.hpp"

using namespace std;

namespace SpMP
{

void CSR::permuteRowptr(CSR *ret, const int *reversePerm) const
{
  assert(isPerm(reversePerm, m));

  int base = getBase();

  idx_t rowPtrSum[omp_get_max_threads() + 1];

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    int iBegin, iEnd;
    getSimpleThreadPartition(&iBegin, &iEnd, m);

    ret->rowptr[iBegin] = 0;
    int i;
    for (i = iBegin; i < iEnd - 1; ++i) {
      int row = reversePerm ? reversePerm[i] : i;
      idx_t begin = rowptr[row], end = rowptr[row + 1];
      ret->rowptr[i + 1] = ret->rowptr[i] + end - begin;
      if (extptr) {
        ret->extptr[i] = ret->rowptr[i] + extptr[row] - rowptr[row];
      }
    }
    idx_t localNnz = 0;
    if (i < iEnd) {
      int row = reversePerm ? reversePerm[i] : i;
      idx_t begin = rowptr[row], end = rowptr[row + 1];
      localNnz = ret->rowptr[i] + end - begin;
      if (extptr) {
        ret->extptr[i] = ret->rowptr[i] + extptr[row] - rowptr[row];
      }
    }

    prefixSum(&localNnz, &ret->rowptr[m], rowPtrSum);
    localNnz += base;

    for (i = iBegin; i < iEnd; ++i) {
      ret->rowptr[i] += localNnz;
      if (ret->extptr) ret->extptr[i] += localNnz;
    }
  } // omp parallel

  ret->rowptr[m] += base;
  assert(ret->rowptr[m] == rowptr[m]);
}

CSR *CSR::permuteRowptr(const int *reversePerm) const
{
  CSR *ret = new CSR();

  ret->m = m;
  ret->n = n;
  ret->rowptr = MALLOC(idx_t, m + 1);
  assert(ret->rowptr);
  idx_t nnz = rowptr[m];
  ret->colidx = MALLOC(int, nnz);
  assert(ret->colidx);
  if (values) {
    ret->values = MALLOC(double, nnz);
    assert(ret->values);
  }
  if (idiag) {
    ret->idiag = MALLOC(double, m);
    assert(ret->idiag);
  }
  if (diagptr) {
    ret->diagptr = MALLOC(idx_t, m);
    assert(ret->diagptr);
  }
  if (extptr) {
    ret->extptr = MALLOC(idx_t, m);
    assert(ret->extptr);
  }
  
  permuteRowptr(ret, reversePerm);

  return ret;
}

template<class T, int BASE = 0>
void permuteColsInPlace_(CSR *A, const int *perm)
{
  assert(perm);

  const idx_t *rowptr = A->rowptr - BASE;
  const idx_t *extptr = A->extptr ? A->extptr - BASE: rowptr + 1;
  idx_t *diagptr = A->diagptr ? A->diagptr - BASE : NULL;

  int *colidx = A->colidx - BASE;
  T *values = A->values - BASE;

  int m = A->m;

#pragma omp parallel for
  for (int i = BASE; i < m + BASE; ++i) {
    int diagCol = -1;
    for (idx_t j = rowptr[i]; j < extptr[i]; ++j) {
      int c = colidx[j];
      assert(c >= 0 && c < m);
      colidx[j] = perm[c - BASE] + BASE;
      assert(colidx[j] - BASE >= 0 && colidx[j] - BASE < m);
      if (c == i) diagCol = perm[c - BASE] + BASE;
    }

    for (idx_t j = rowptr[i] + 1; j < extptr[i]; ++j) {
      int c = colidx[j];
      double v = values[j];

      idx_t k = j - 1;
      while (k >= rowptr[i] & colidx[k] > c) {
        colidx[k + 1] = colidx[k];
        values[k + 1] = values[k];
        --k;
      }

      colidx[k + 1] = c;
      values[k + 1] = v;
    }

    if (diagptr) {
      for (idx_t j = rowptr[i]; j < extptr[i]; ++j) {
        if (colidx[j] == diagCol) {
          diagptr[i] = j;
          break;
        }
      }
    }
  } // for each row
}

void CSR::permuteColsInPlace(const int *perm)
{
  if (0 == getBase()) {
    permuteColsInPlace_<double, 0>(this, perm);
  }
  else {
    assert(1 == getBase());
    permuteColsInPlace_<double, 1>(this, perm);
  }
}

template<class T, int BASE = 0, bool SORT = false>
static void permuteMain_(
  CSR *out, const CSR *in,
  const int *columnPerm, const int *rowInversePerm)
{
  int m = in->m;

  const idx_t *rowptr = in->rowptr - BASE;
  const idx_t *diagptr = in->diagptr ? in->diagptr - BASE : NULL;
  const idx_t *extptr = in->extptr ? in->extptr - BASE : rowptr + 1;
  const int *colidx = in->colidx - BASE;
  const T *values = in->values ? in->values - BASE : NULL;
  const T *idiag = in->idiag ? in->idiag - BASE : NULL;

  idx_t *newRowptr = out->rowptr - BASE;
  idx_t *newDiagptr = out->diagptr ? out->diagptr - BASE : NULL;
  idx_t *newExtptr = out->extptr ? out->extptr - BASE : newRowptr + 1;
  int *newColidx = out->colidx - BASE;
  T *newValues = out->values ? out->values - BASE : NULL;
  T *newIdiag = out->idiag ? out->idiag - BASE : NULL;

  columnPerm -= BASE;
  if (rowInversePerm) rowInversePerm -= BASE;

#pragma omp parallel
  {
    int iBegin, iEnd;
    getLoadBalancedPartition(&iBegin, &iEnd, newRowptr + BASE, m);
    iBegin += BASE;
    iEnd += BASE;

    if (rowInversePerm && values && !idiag && !diagptr && !SORT) {
      // a common case
      for (int i = iBegin; i < iEnd; ++i) {
        int row = rowInversePerm[i] + BASE;
        idx_t begin = rowptr[row], end = extptr[row];
        idx_t newBegin = newRowptr[i];

        idx_t k = newBegin;
        for (idx_t j = begin; j < end; ++j, ++k) {
          int colIdx = colidx[j];
          int newColIdx = columnPerm[colIdx] + BASE;

          newColidx[k] = newColIdx;
          newValues[k] = values[j];
        }
      } // for each row
    }
    else {
      for (int i = iBegin; i < iEnd; ++i) {
        int row = rowInversePerm ? rowInversePerm[i] + BASE : i;
        idx_t begin = rowptr[row], end = extptr[row];
        idx_t newBegin = newRowptr[i];

        int diagCol = -1;
        idx_t k = newBegin;
        for (idx_t j = begin; j < end; ++j, ++k) {
          int colIdx = colidx[j];
          int newColIdx = columnPerm[colIdx] + BASE;

          newColidx[k] = newColIdx;
          if (values) newValues[k] = values[j];

          if (diagptr && colidx[j] == row) {
            diagCol = newColIdx;
          }
        }
        assert(!diagptr || diagCol != -1);

        if (SORT) {
          // insertion sort
          for (idx_t j = newBegin + 1; j < newExtptr[i]; ++j) {
            int c = newColidx[j];
            double v = newValues[j];

            idx_t k = j - 1;
            while (k >= newBegin && newColidx[k] > c) {
              newColidx[k + 1] = newColidx[k];
              newValues[k + 1] = newValues[k];
              --k;
            }

            newColidx[k + 1] = c;
            newValues[k + 1] = v;
          }
        }

        if (idiag) newIdiag[i] = idiag[row];

        if (diagptr) {
          for (idx_t j = newBegin; j < newExtptr[i]; ++j) {
            if (newColidx[j] == diagCol) {
              newDiagptr[i] = j;
              break;
            }
          }
        } // if (diagptr)
      } // for each row
    }
  } // omp parallel
}

void CSR::permuteMain(
  CSR *out, const int *columnPerm, const int *rowInversePerm,
  bool sort /*=false*/) const
{
  if (0 == getBase()) {
    if (sort) {
      permuteMain_<double, 0, true>(out, this, columnPerm, rowInversePerm);
    }
    else {
      permuteMain_<double, 0, false>(out, this, columnPerm, rowInversePerm);
    }
  }
  else {
    assert(1 == getBase());
    if (sort) {
      permuteMain_<double, 1, true>(out, this, columnPerm, rowInversePerm);
    }
    else {
      permuteMain_<double, 1, false>(out, this, columnPerm, rowInversePerm);
    }
  }
}

template<class T, int BASE = 0>
static void permuteRowsMain_(
  CSR *out, const CSR *in, const int *rowInversePerm)
{
  const idx_t *rowptr = in->rowptr - BASE;
  const int *colidx = in->colidx - BASE;
  const idx_t *diagptr = in->diagptr ? in->diagptr - BASE : NULL;
  const T *values = in->values ? in->values - BASE : NULL;
  const T *idiag = in->idiag ? in->idiag - BASE : NULL;

  int m = in->m;

#pragma omp parallel
  {
    int iBegin, iEnd;
    getLoadBalancedPartition(&iBegin, &iEnd, out->rowptr + BASE, m);
    iBegin += BASE;
    iEnd += BASE;

    for (int i = iBegin; i < iEnd; ++i) {
      int row = rowInversePerm ? rowInversePerm[i - BASE] + BASE : i;
      idx_t begin = rowptr[row], end = rowptr[row + 1];
      idx_t newBegin = out->rowptr[i] - BASE;

      if (values)
        memcpy(out->values + newBegin, values + begin, (end - begin)*sizeof(double));
      memcpy(out->colidx + newBegin, colidx + begin, (end - begin)*sizeof(int));

      if (diagptr)
        out->diagptr[i] =
          out->rowptr[i] + (diagptr[row] - rowptr[row]);
      if (idiag) out->idiag[i] = idiag[row];
    }
  }
}

void CSR::permuteRowsMain(CSR *out, const int *rowInversePerm) const
{
  if (0 == getBase()) {
    permuteRowsMain_<double, 0>(out, this, rowInversePerm);
  }
  else {
    assert(1 == getBase());
    permuteRowsMain_<double, 1>(out, this, rowInversePerm);
  }
}

CSR *CSR::permute(const int *columnPerm, const int *rowInversePerm, bool sort /*=false*/) const
{
  CSR *ret = permuteRowptr(rowInversePerm);
  permuteMain(ret, columnPerm, rowInversePerm, sort);
  return ret;
}

CSR *CSR::permuteRows(const int *reversePerm) const
{
  CSR *ret = permuteRowptr(reversePerm);
  permuteRowsMain(ret, reversePerm);
  return ret;
}

} // namespace SpMP
