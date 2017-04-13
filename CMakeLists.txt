project(CSPLATT)
cmake_minimum_required(VERSION 3.1.0)

include(cmake/version.cmake)

# Libraries to link against -- will concatenate to this
set(SPLATT_LIBS splatt m)

# Append to these
set(SPLATT_FLAGS "")   # shared C and C++ flags

# Intel customization
if(DEFINED INTEL_OPT)
  set(INTEL_OPT true)
  set(SPLATT_FLAGS  "-restrict -D_XOPEN_SOURCE=600")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
# GCC/Clang
else()
  set(INTEL_OPT false)
endif()

# OS-specific stuff
include(cmake/system.cmake)

# Configuration
include(cmake/types.cmake)
include(cmake/constants.cmake)
include(cmake/debug.cmake)
include(cmake/flags.cmake)

# Libraries
include(cmake/lapack.cmake)
include(cmake/openmp.cmake)
include(cmake/mpi.cmake)
include(cmake/partition.cmake)
include(cmake/fortran.cmake)

# Set flags/defines
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${SPLATT_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPLATT_FLAGS}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(include)

# Source code files to compile for library
file(GLOB SPLATT_SOURCES
      src/*.c
      src/*.cc
      src/ccp/*.c
      src/cpd/*.c
      src/cpd/*.cc
      src/cpd/constraints/*.c
      src/cpd/constraints/*.cc
      ${MPI_SOURCES})

# Generate splatt library
add_subdirectory(lib)

# Compile applications
add_subdirectory(src)

# Unit tests
add_subdirectory(tests)

# uninstall
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)
add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)


