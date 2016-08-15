#!/bin/bash

LAPACK_VERSION=3.5.0
LAPACK_LOC=http://www.netlib.org/lapack/lapack-${LAPACK_VERSION}.tgz

if [ "$#" -eq 0 ]; then
  echo "usage: <build directory>";
  exit 1;
fi

wget ${LAPACK_LOC} --output-document=$1/lapack.tgz;

pushd $1/
tar xf lapack.tgz
rm lapack.tgz;

mv lapack-${LAPACK_VERSION} lapack
pushd lapack
cmake .
make -j

popd
popd

