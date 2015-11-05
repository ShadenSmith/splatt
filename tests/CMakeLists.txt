
project(CSPLATT_TEST)
cmake_minimum_required(VERSION 2.6.0)

add_definitions(-DSPLATT_TEST_DATASETS="${CMAKE_CURRENT_SOURCE_DIR}/tensors/")
add_definitions(-DSPLATT_TEST_GRAPHS="${CMAKE_CURRENT_SOURCE_DIR}/graphs/")

file(GLOB TEST_SOURCES *.c)
add_executable(splatt_test ${TEST_SOURCES})
set_target_properties(splatt_test PROPERTIES OUTPUT_NAME test)

# lots of linking
target_link_libraries(splatt_test splatt)
target_link_libraries(splatt_test m)
target_link_libraries(splatt_test rt)
target_link_libraries(splatt_test ${MPI_C_LIBRARIES})
target_link_libraries(splatt_test ${PATOH_LIB})
target_link_libraries(splatt_test ${ASHADO_LIB})
target_link_libraries(splatt_test ${MTMETIS_LIB})

