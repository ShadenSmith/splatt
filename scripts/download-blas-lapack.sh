#!/bin/bash

if [ "$#" -eq 0 ]; then
  echo "usage: <build directory> [int size]";
  exit 1;
fi

BUILD_DIR=$1
mkdir -p ${BUILD_DIR}

BISIZE=32
if [ "$#" -gt 1 ]; then
  BISIZE=$2
fi

#
# Build BLIS for its BLAS interface.
#
git clone https://github.com/flame/blis.git ${BUILD_DIR}/blis
pushd $1/blis
  CC=gcc ./configure -p .. --enable-threading=openmp --blas-int-size=${BISIZE} auto
  make -j
  make install
popd

#
# Build reference LAPACK implementation.
#
git clone https://github.com/Reference-LAPACK/lapack.git ${BUILD_DIR}/lapack
pushd ${BUILD_DIR}/lapack
  # Setup LAPACK build to use BLIS.
  BLIS_LIB=${BUILD_DIR}/lib/libblis.a
  sed "s@../../librefblas.a@${BLIS_LIB}@" make.inc.example > make.inc
  make -j lapacklib
  mv liblapack.a ../lib/
popd

