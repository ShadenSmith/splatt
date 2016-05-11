INSTALL(FILES splatt.h DESTINATION include)

FILE(GLOB install_headers splatt/*.h)
INSTALL(FILES ${install_headers} DESTINATION include/splatt)
