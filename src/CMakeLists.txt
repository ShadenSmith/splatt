
project(CSPLATT_APPS)
cmake_minimum_required(VERSION 2.6.0)

add_executable(splatt_bin splatt_bin.c)
set_target_properties(splatt_bin PROPERTIES OUTPUT_NAME splatt)
target_link_libraries(splatt_bin splatt m rt)
install(TARGETS splatt_bin RUNTIME DESTINATION bin)

