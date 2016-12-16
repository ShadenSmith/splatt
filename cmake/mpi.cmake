
if (DEFINED MPICC)
  message("Using user supplied mpicc=${MPICC}")
  set(MPI_C_COMPILER ${MPICC})
endif()

if (DEFINED MPICXX)
  message("Using user supplied mpicxx=${MPICXX}")
  set(MPI_CXX_COMPILER ${MPICC})
endif()

# check for MPI mode
if (DEFINED USE_MPI)
  message("Building with MPI support.")
  add_definitions(-DSPLATT_USE_MPI=${USE_MPI})
  find_package(MPI REQUIRED)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
  file(GLOB MPI_SOURCES
      src/mpi/*.c
      src/mpi/*.cc)
  set(SPLATT_LIBS ${SPLATT_LIBS} ${MPI_C_LIBRARIES})
else()
  set(MPI_SOURCES "")
endif()

