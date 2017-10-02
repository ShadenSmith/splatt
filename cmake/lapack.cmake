
# just use MKL flag
if (INTEL_OPT)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mkl")
else()

  # BLAS/LAPACK
  if (DEFINED DOWNLOAD_BLAS_LAPACK)

    message(STATUS "Building BLIS & reference LAPACK.")
    execute_process(
        COMMAND
        ${CMAKE_SOURCE_DIR}/scripts/download-blas-lapack.sh
        ${CMAKE_BINARY_DIR}/blis
        ${CONFIG_BLAS_INT}
    )

    set(USER_LAPACK_LIB ${CMAKE_BINARY_DIR}/blis/lib/liblapack.a)
    set(USER_BLAS_LIB ${CMAKE_BINARY_DIR}/blis/lib/libblis.a)

    # Force Fortran linking (due to reference LAPACK)
    set(USE_FORTRAN 1)

    # avoid annoying warning
    set(SPLATT_NOWARN ${DOWNLOAD_BLAS_LAPACK})
  endif()

  if (DEFINED USER_LAPACK_LIB)
    message("Using user supplied LAPACK=${USER_LAPACK_LIB}")
    set(SPLATT_LIBS ${SPLATT_LIBS} ${USER_LAPACK_LIB})
  # auto find LAPACK
  else()
    find_package(LAPACK)
    set(SPLATT_LIBS ${SPLATT_LIBS} ${LAPACK_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
    if(${LAPACK_FOUND})
      message("FOUND LAPACK LIBS: " ${LAPACK_LIBRARIES})
    else()
      message(FATAL_ERROR "Could not find LAPACK library. Run `./configure --help`  for assistance.")
    endif()
  endif()

  if (DEFINED USER_BLAS_LIB)
    message("Using user supplied BLAS=${USER_BLAS_LIB}")
    set(SPLATT_LIBS ${SPLATT_LIBS} ${USER_BLAS_LIB})
  # auto find BLAS
  else()
    find_package(BLAS)
    include_directories(${BLAS_INCLUDE_DIR})
    set(SPLATT_LIBS ${SPLATT_LIBS} ${BLAS_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
    if(${BLAS_FOUND})
      message("FOUND BLAS LIBS: " ${BLAS_LIBRARIES})
    else()
      message(FATAL_ERROR "Could not find BLAS library. Run `./configure --help`  for assistance.")
    endif()
  endif()

endif() # not INTEL_OPT

