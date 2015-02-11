
project(CSPLATT_APPS)
cmake_minimum_required(VERSION 2.6.0)

file(GLOB CMD_SOURCES cmds/*.c)
add_executable(splatt_bin ${CMD_SOURCES})
set_target_properties(splatt_bin PROPERTIES OUTPUT_NAME splatt)
target_link_libraries(splatt_bin splatt m rt ${MPI_C_LIBRARIES})
install(TARGETS splatt_bin RUNTIME DESTINATION bin)

