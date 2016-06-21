
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-missing-field-initializers")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-variable")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
# Unfortunately, these warnings pop up in GCC 5.1 despite our MAX_NMODES checks
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-array-bounds")

