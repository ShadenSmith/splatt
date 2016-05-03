
project(CSPLATT_APPS)
cmake_minimum_required(VERSION 2.8.0)

file(GLOB CMD_SOURCES cmds/*.c)
add_executable(splatt_bin ${CMD_SOURCES})
set_target_properties(splatt_bin PROPERTIES OUTPUT_NAME splatt)

# lots of linking
target_link_libraries(splatt_bin ${SPLATT_LIBS})

install(TARGETS splatt_bin RUNTIME DESTINATION bin)

