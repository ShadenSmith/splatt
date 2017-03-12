#!/bin/bash

if [ "$#" -eq 0 ]; then
  echo "usage: <download directory>";
  exit 1;
fi

ARCH="$(uname -s)-$(uname -m)"
CMAKE_FILE="cmake-3.6.3-${ARCH}.tar.gz"

CMAKE_URL="https://cmake.org/files/v3.6/${CMAKE_FILE}"
CMAKE_DIR=$1/cmake
mkdir -p ${CMAKE_DIR}
cd ${CMAKE_DIR}

curl -O ${CMAKE_URL}
tar -f ${CMAKE_FILE} --strip-components=1 -xz -C ${CMAKE_DIR}

echo "${CMAKE_DIR}/bin/cmake"
