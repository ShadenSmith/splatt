project(CSPLATT)
cmake_minimum_required(VERSION 2.6.0)

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
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
endif()

# check for MPI mode
if (DEFINED USE_MPI)
  message("Building with MPI support.")
  add_definitions(-DSPLATT_USE_MPI=${USE_MPI})
  find_package(MPI)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
  file(GLOB MPI_SOURCES src/mpi/*.c)
else()
  set(MPI_SOURCES "")
endif()

# check for PatoH support
if (DEFINED USE_PATOH)
  message("Building with PaToH support.")
  add_definitions(-DSPLATT_USE_PATOH=${USE_PATOH})
  set(PATOH_LIB "patoh")
endif()

# check for Ashado support
if (DEFINED USE_ASHADO)
  message("Building with Ashado support.")
  add_definitions(-DSPLATT_USE_ASHADO=${USE_ASHADO})
  set(ASHADO_LIB "ashado")
endif()

add_definitions(-D_GNU_SOURCE)

set(C_DEFINES "${C_DEFINES}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_DEFINES}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
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

file(GLOB SPLATT_SOURCES src/*.c ${MPI_SOURCES})

# Generate splatt library
add_subdirectory(lib)

add_subdirectory(include)

# Compile applications
add_subdirectory(src)

# UNIT TESTS
#add_subdirectory(test)
