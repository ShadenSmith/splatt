
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
  file(GLOB MPI_SOURCES src/mpi/*.c)

  # set new MPI vars
  include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}")
  set(SPLATT_LIBS ${SPLATT_LIBS} ${MPI_C_LIBRARIES})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_C_LINK_FLAGS}")
else()
  set(MPI_SOURCES "")
endif()

