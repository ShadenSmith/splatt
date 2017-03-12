#!/bin/bash

if [ "$#" -eq 0 ]; then
  echo "usage: <download directory>";
  exit 1;
fi

# Download location
ARCH="$(uname -s)-$(uname -m)"
CMAKE_FILE="cmake-3.6.3-${ARCH}.tar.gz"
CMAKE_URL="https://cmake.org/files/v3.6/${CMAKE_FILE}"
CMAKE_DIR=$1/cmake
mkdir -p ${CMAKE_DIR}
cd ${CMAKE_DIR}

# Download binary
curl -O ${CMAKE_URL}
tar -f ${CMAKE_FILE} --strip-components=1 -xz -C ${CMAKE_DIR}

# Apple has a different app directory than Linux
APP_DIR=${CMAKE_DIR}
if [[ "$(uname -s)" == "Darwin" ]]; then
  APP_DIR="${CMAKE_DIR}/CMake.app/Contents"
fi

# Return CMake binary path in STDOUT.
echo "${APP_DIR}/bin/cmake"
