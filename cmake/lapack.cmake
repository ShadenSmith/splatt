
# just use MKL flag
if (INTEL_OPT)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mkl")
else()

  # BLAS/LAPACK
  if (DEFINED DOWNLOAD_BLAS_LAPACK)

    message(WARNING "Downloading generic BLAS/LAPACK libraries.")
    message(WARNING "  NOTE: performance may suffer.")
    execute_process(COMMAND ${CMAKE_SOURCE_DIR}/scripts/download-blas-lapack.sh ${CMAKE_BINARY_DIR})

    # Enable linking against Fortran
    enable_language(Fortran)

    set(USER_LAPACK_LIB ${CMAKE_BINARY_DIR}/lapack/lib/liblapack.a)

    # Link against generic BLAS and a Fortran library.
    # TODO: Is there a better way to do this? The Fortran library must be added
    # AFTER BLAS/LAPACK.
    if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
      set(USER_BLAS_LIB ${CMAKE_BINARY_DIR}/lapack/lib/libblas.a ifcore)
    else()
      set(USER_BLAS_LIB ${CMAKE_BINARY_DIR}/lapack/lib/libblas.a gfortran)
    endif()

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

