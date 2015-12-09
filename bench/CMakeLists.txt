
project(CSPLATT_BENCH)
cmake_minimum_required(VERSION 2.6.0)

# tucker
add_executable(tucker_bench tucker_bench.c)
set_target_properties(tucker_bench PROPERTIES OUTPUT_NAME tucker_bench)
target_link_libraries(tucker_bench ${SPLATT_LIBS})

