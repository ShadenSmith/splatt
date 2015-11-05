
project(CSPLATT_APPS)
cmake_minimum_required(VERSION 2.6.0)

file(GLOB CMD_SOURCES cmds/*.c)
add_executable(splatt_bin ${CMD_SOURCES})
set_target_properties(splatt_bin PROPERTIES OUTPUT_NAME splatt)

# lots of linking
target_link_libraries(splatt_bin splatt)
target_link_libraries(splatt_bin splatt)
target_link_libraries(splatt_bin m)
target_link_libraries(splatt_bin rt)
target_link_libraries(splatt_bin ${MPI_C_LIBRARIES})
target_link_libraries(splatt_bin ${MTMETIS_LIB})
target_link_libraries(splatt_bin ${PATOH_LIB})
target_link_libraries(splatt_bin ${ASHADO_LIB})

install(TARGETS splatt_bin RUNTIME DESTINATION bin)

