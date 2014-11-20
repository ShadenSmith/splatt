
project(CSPLATT_APPS)
cmake_minimum_required(VERSION 2.6.0)

add_executable(splatt splatt.c)
target_link_libraries(splatt lsplatt m rt)

add_executable(ttkrao ttkrao.c)
target_link_libraries(ttkrao lsplatt m rt)

add_executable(ttstats ttstats.c)
target_link_libraries(ttstats lsplatt m rt)
