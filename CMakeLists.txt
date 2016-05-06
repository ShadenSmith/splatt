project(CSPLATT)

cmake_minimum_required(VERSION 2.8.0)

# Get version string from git repo if possible
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git/)
  execute_process(COMMAND git --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git rev-parse --short HEAD
                  OUTPUT_VARIABLE SPLATT_COMMIT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND git --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git rev-parse --abbrev-ref HEAD
                  OUTPUT_VARIABLE SPLATT_BRANCH
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(SPLATT_VERSION_STR "${SPLATT_COMMIT}-${SPLATT_BRANCH}")
  set(C_DEFINES "${C_DEFINES} -DSPLATT_VERSION_STR=\\\"${SPLATT_VERSION_STR}\\\"")
endif()


find_package(OpenMP)
if(${OPENMP_FOUND})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message("OpenMP runtime not found! SPLATT will not be multithreaded.")
endif()


# ALL LIBRARIES -- will concatenate to this
set(SPLATT_LIBS splatt m rt)

# setup Fortran
enable_language(Fortran)
if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
  set(SPLATT_LIBS ${SPLATT_LIBS} ifcore)
else()
  set(SPLATT_LIBS ${SPLATT_LIBS} gfortran)
endif()




if (DEFINED MPICC)
  message("Using user supplied mpicc=${MPICC}")
  set(${MPI_C_COMPILER} ${MPICC})
endif()

# check for MPI mode
if (DEFINED USE_MPI)
  message("Building with MPI support.")
  add_definitions(-DSPLATT_USE_MPI=${USE_MPI})
  find_package(MPI REQUIRED)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
  file(GLOB MPI_SOURCES src/mpi/*.c)
  set(SPLATT_LIBS ${SPLATT_LIBS} ${MPI_C_LIBRARIES})

  message("MPI_C_COMPILER: ${MPI_C_COMPILER}")
  message("MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
  message("MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")
  message("MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
  message("MPIEXEC: ${MPIEXEC}")
else()
  set(MPI_SOURCES "")
endif()


# check for debug mode
if (DEFINED DEBUG)
  message("Building in DEBUG mode. Assertions enabled.")
  add_definitions(-DDEBUG=${DEBUG})
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
else()
  message("Building in RELEASE mode.")
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
endif()


# type widths
if (DEFINED SPLATT_IDX_WIDTH)
  add_definitions(-DSPLATT_IDX_TYPEWIDTH=${SPLATT_IDX_WIDTH})
  message("Using ${SPLATT_IDX_WIDTH}-bit integers.")
endif()

if (DEFINED SPLATT_VAL_WIDTH)
  if (${SPLATT_VAL_WIDTH} STREQUAL "single")
    add_definitions(-DSPLATT_VAL_TYPEWIDTH=32)
  elseif (${SPLATT_VAL_WIDTH} STREQUAL "double")
    add_definitions(-DSPLATT_VAL_TYPEWIDTH=64)
  else()
    message(FATAL_ERROR "Precision '${SPLATT_VAL_WIDTH}' not recognized.\
      Choose between {single, double}.")
  endif()
  message("Using ${SPLATT_VAL_WIDTH} precision floating point numbers.")
endif()


# BLAS/LAPACK
if (DEFINED DOWNLOAD_BLAS_LAPACK)
  message(WARNING "Downloading generic BLAS/LAPACK libraries.")
  message(WARNING "  NOTE: performance may suffer.")
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/scripts/download-blas-lapack.sh ${CMAKE_BINARY_DIR})
  set(USER_LAPACK_LIB ${CMAKE_BINARY_DIR}/lapack/lib/liblapack.a)
  set(USER_BLAS_LIB ${CMAKE_BINARY_DIR}/lapack/lib/libblas.a)

  # avoid annoying warning
  set(SPLATT_NOWARN ${DOWNLOAD_BLAS_LAPACK})
endif()

if (DEFINED USER_LAPACK_LIB)
  message("Using user supplied LAPACK=${USER_LAPACK_LIB}")
  set(SPLATT_LIBS ${SPLATT_LIBS} ${USER_LAPACK_LIB})
# auto find LAPACK
else()
  find_package(LAPACK)
  set(SPLATT_LIBS ${SPLATT_LIBS} ${LAPACK_LIBRARIES})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
  if(${LAPACK_FOUND})
    message("FOUND LAPACK LIBS: " ${LAPACK_LIBRARIES})
  else()
    message(FATAL_ERROR "Could not find LAPACK library. Run `./configure --help`  for assistance.")
  endif()
endif()

if (DEFINED USER_BLAS_LIB)
  message("Using user supplied BLAS=${USER_BLAS_LIB}")
  set(SPLATT_LIBS ${SPLATT_LIBS} ${USER_BLAS_LIB})
# auto find BLAS
else()
  find_package(BLAS)
  include_directories(${BLAS_INCLUDE_DIR})
  set(SPLATT_LIBS ${SPLATT_LIBS} ${BLAS_LIBRARIES})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
  if(${BLAS_FOUND})
    message("FOUND BLAS LIBS: " ${BLAS_LIBRARIES})
  else()
    message(FATAL_ERROR "Could not find BLAS library. Run `./configure --help`  for assistance.")
  endif()
endif()


# check for MT-Metis support
if (DEFINED USE_MTMETIS)
  message("Building with MT-Metis support.")
  add_definitions(-DSPLATT_USE_MTMETIS=${USE_MTMETIS})
  set(SPLATT_LIBS ${SPLATT_LIBS} mtmetis)
endif()

# check for PatoH support
if (DEFINED USE_PATOH)
  message("Building with PaToH support.")
  add_definitions(-DSPLATT_USE_PATOH=${USE_PATOH})
  set(SPLATT_LIBS ${SPLATT_LIBS} patoh)
endif()

# check for Ashado support
if (DEFINED USE_ASHADO)
  message("Building with Ashado support.")
  add_definitions(-DSPLATT_USE_ASHADO=${USE_ASHADO})
  set(SPLATT_LIBS ${SPLATT_LIBS} ashado)
endif()

add_definitions(-D_GNU_SOURCE)

set(C_DEFINES "${C_DEFINES}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_DEFINES}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -funroll-loops")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftree-vectorize")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fgnu89-inline")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstrict-aliasing")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopt-info-vec")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopt-info-vec-missed")

# Development mode enables much more strict warnings
if(DEFINED DEV_MODE)
  add_definitions(-DDEV_MODE=${DEV_MODE})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-missing-field-initializers")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-variable")
  #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result")
  #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  # Unfortunately, these warnings pop up in GCC 5.1 despite our MAX_NMODES checks
  #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-array-bounds")
else()
  # turn off warning entirely for release builds
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB SPLATT_SOURCES src/*.c src/ccp/*.c ${MPI_SOURCES})

# Generate splatt library
add_subdirectory(lib)

add_subdirectory(include)

# Compile applications
add_subdirectory(src)

# UNIT TESTS
add_subdirectory(tests)


# uninstall
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

