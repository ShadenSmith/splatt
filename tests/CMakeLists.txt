
project(CSPLATT_TEST)
cmake_minimum_required(VERSION 2.6.0)

add_definitions(-DSPLATT_TEST_DATASETS="${CMAKE_CURRENT_SOURCE_DIR}/tensors/")
add_definitions(-DSPLATT_TEST_GRAPHS="${CMAKE_CURRENT_SOURCE_DIR}/graphs/")

file(GLOB TEST_SOURCES *.c)

add_executable(splatt_test ${TEST_SOURCES})
set_target_properties(splatt_test PROPERTIES OUTPUT_NAME test)
target_link_libraries(splatt_test ${SPLATT_LIBS})

# MPI tests too
if (DEFINED USE_MPI)
  file(GLOB MPI_TEST_SOURCES mpi/*.c)
  add_executable(splatt_test_mpi ${MPI_TEST_SOURCES})
  set_target_properties(splatt_test_mpi PROPERTIES OUTPUT_NAME test_mpi)
  target_link_libraries(splatt_test_mpi ${SPLATT_LIBS})
endif()
