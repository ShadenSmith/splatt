
if(DEFINED DISABLE_OPENMP)
  message("Disabling OpenMP. SPLATT will not be multi-threaded.")

  set(SPLATT_NOWARN ${DISABLE_OPENMP})
else()
  find_package(OpenMP)
  if(${OPENMP_FOUND})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    message("OpenMP runtime not found! SPLATT will not be multi-threaded.")
  endif()
endif()

