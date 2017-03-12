#!/bin/bash

if [ "$#" -eq 0 ]; then
  echo "usage: <download directory>";
  exit 1;
fi

ARCH="$(uname -s)-$(uname -m)"
CMAKE_URL="https://cmake.org/files/v3.6/cmake-3.6.3-${ARCH}.tar.gz"
CMAKE_DIR=$1/cmake
mkdir -p ${CMAKE_DIR}
cd ${CMAKE_DIR}

# wget fails on travis-ci
if [[ "${CI}" == "true" && "${TRAVIS}" == "true" ]]; then
  WGET_CMD="travis_retry wget"
else
  WGET_CMD="wget"
fi

${WGET_CMD} --quiet ${CMAKE_URL} -O cmake.tgz
tar -f cmake.tgz --strip-components=1 -xz -C ${CMAKE_DIR}

echo "${CMAKE_DIR}/bin/cmake"
