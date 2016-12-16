
if(SPLATT_WARNINGS)
  set(SPLATT_FLAGS "${SPLATT_FLAGS} -pedantic")
  set(SPLATT_FLAGS "${SPLATT_FLAGS} -Wall")
  set(SPLATT_FLAGS "${SPLATT_FLAGS} -Wextra")
  set(SPLATT_FLAGS "${SPLATT_FLAGS} -Wno-missing-field-initializers")
  set(SPLATT_FLAGS "${SPLATT_FLAGS} -Wno-unused-parameter")
  set(SPLATT_FLAGS "${SPLATT_FLAGS} -Wno-unused-variable")
  #set(SPLATT_FLAGS "${SPLATT_FLAGS} -Wno-unused-result")
  #set(SPLATT_FLAGS "${SPLATT_FLAGS} -Wno-unused-function")
  # Unfortunately, these warnings pop up in GCC 5.1 despite our MAX_NMODES checks
  #set(SPLATT_FLAGS "${SPLATT_FLAGS} -Wno-array-bounds")
else()
  # turn off warning entirely for release builds
  set(SPLATT_FLAGS "${SPLATT_FLAGS} -w")
endif()


