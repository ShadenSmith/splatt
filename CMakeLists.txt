project(CSPLATT)

cmake_minimum_required(VERSION 2.8.0)

include(cmake/version.cmake)
include(cmake/openmp.cmake)

# ALL LIBRARIES -- will concatenate to this
set(SPLATT_LIBS splatt m rt)

# setup Fortran
enable_language(Fortran)
if (NOT DEFINED CMAKE_Fortran_COMPILER)
  message(FATAL_ERROR "Fortran compiler not found!")
endif()

if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
  set(SPLATT_LIBS ${SPLATT_LIBS} ifcore)
else()
  set(SPLATT_LIBS ${SPLATT_LIBS} gfortran)
endif()

if (DEFINED MAX_MODES)
  add_definitions(-DSPLATT_MAX_NMODES=${MAX_MODES})
  message("Setting SPLATT_MAX_NMODES=${MAX_MODES}")
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



add_definitions(-D_GNU_SOURCE)

# Intel toggles
if(DEFINED INTEL_OPT)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -xHost")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mkl")

  # no annoying warnings
  set(SPLATT_NOWARN ${INTEL_OPT})
# GCC/Clang
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftree-vectorize")
  include(cmake/lapack.cmake)
endif()

set(C_DEFINES "${C_DEFINES}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_DEFINES}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -funroll-loops")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fgnu89-inline")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstrict-aliasing")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")


set(SPLATT_WARNINGS false)

# check for debug mode
if (DEFINED DEBUG)
  message("Building in DEBUG mode. Assertions enabled.")
  add_definitions(-DDEBUG=${DEBUG})
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")

  set(SPLATT_WARNINGS true)
else()
  message("Building in RELEASE mode.")
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
endif()

# Development mode enables much more strict warnings
if(DEFINED DEV_MODE)
  add_definitions(-DDEV_MODE=${DEV_MODE})
  set(SPLATT_WARNINGS true)
else()
endif()


if(SPLATT_WARNINGS)
  include(cmake/warnings.cmake)
else()
  # turn off warning entirely for release builds
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB SPLATT_SOURCES src/*.c src/ccp/*.c ${MPI_SOURCES})

include(cmake/partition.cmake)
include(cmake/mpi.cmake)

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

