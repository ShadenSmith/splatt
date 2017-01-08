INSTALL(FILES splatt.h DESTINATION include)

FILE(GLOB install_headers splatt/*.h)
# Don't install config files.
LIST(REMOVE_ITEM install_headers "${CMAKE_SOURCE_DIR}/include/splatt/types_config.h")

INSTALL(FILES ${install_headers} DESTINATION include/splatt)
