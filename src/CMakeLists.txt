# Install snapy library
#
#   library: snapy_debug.o
#   library: snapy_release.o
#
# Define the following symbols
#
#   SNAP_INCLUDE_DIR
#   snapy::snap
#   snapy::snap_cu
#
# Normal usage would be:
#
#   include_directories(${SNAP_INCLUDE_DIR})
#   target_link_libraries(snapy::snap snapy::snap_cu)

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${CMAKE_BINARY_DIR}
  ${NETCDF_INCLUDES}
  ${PNETCDF_INCLUDE_DIR}
  SYSTEM ${TORCH_API_INCLUDE_DIR}
  SYSTEM ${TORCH_INCLUDE_DIR}
  SYSTEM ${KINTERA_INCLUDE_DIR}
  SYSTEM ${HARP_INCLUDE_DIR}
  SYSTEM ${CMAKE_SOURCE_DIR}/external/gloo
  )

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

set(namel snap)
string(TOUPPER ${namel} nameu)

file(GLOB src_files
    #*.cpp
    bc/*.cpp
    coord/*.cpp
    eos/*.cpp
    eos/aneos/*.cpp
    hydro/*.cpp
    riemann/*.cpp
    recon/*.cpp
    mesh/*.cpp
    output/*.cpp
    scalar/*.cpp
    utils/*.cpp
    forcing/*.cpp
    implicit/*.cpp
    input/*.cpp
    sedimentation/*.cpp
    layout/*.cpp
    )

add_library(${namel}_${buildl}
    SHARED
    ${src_files}
    )

set_target_properties(${namel}_${buildl}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
    BUILD_RPATH_USE_ORIGIN TRUE
    )

target_link_libraries(${namel}_${buildl}
    PUBLIC
    ${DISORT_LIBRARY}
    ${HARP_LIBRARY}
    ${KINTERA_LIBRARY}
    ${NETCDF_LIBRARIES}
    ${PNETCDF_LIBRARIES}
    ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    fmt::fmt
    yaml-cpp::yaml-cpp
    )

if(PNETCDF_LIBRARIES)
  get_filename_component(PNETCDF_LIBRARY_DIR "${PNETCDF_LIBRARY}" DIRECTORY)
  set_property(TARGET ${namel}_${buildl} APPEND PROPERTY
    BUILD_RPATH "${PNETCDF_LIBRARY_DIR};$ORIGIN/../pinc/lib")
endif()

if(UCX_FOUND)
  target_link_libraries(${namel}_${buildl}
    PUBLIC commux::commux)

  set_property(TARGET ${namel}_${buildl} APPEND PROPERTY
    BUILD_RPATH "${COMMUX_LIBRARY_DIR};$ORIGIN/../../commux/lib")

  if(CUDA)
    if(NOT COMMUX_CUDA_FOUND)
      message(FATAL_ERROR
        "CUDA=ON requires commux's CUDA sidecar library "
        "libcommux_cuda.so, but it was not found")
    endif()

    target_link_libraries(${namel}_${buildl}
      PUBLIC commux::commux_cuda)

    target_compile_definitions(${namel}_${buildl}
      PRIVATE COMMUX_WITH_CUDA_RUNTIME=1)
  endif()
endif()

add_library(snapy::snap ALIAS ${namel}_${buildl})

# CUDA support and source codes
if (CUDA AND CUDAToolkit_FOUND)
  target_include_directories(${namel}_${buildl}
    SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

  target_link_libraries(${namel}_${buildl}
    PUBLIC
    CUDA::cudart
    ${TORCH_CUDA_LIBRARY}
    ${C10_CUDA_LIBRARY}
    )

  file(GLOB cu_src_files
      bc/*.cu
      eos/*.cu
      eos/aneos/*.cu
      coord/*.cu
      hydro/*.cu
      recon/*.cu
      riemann/*.cu
      scalar/*.cu
      sedimentation/*.cu
      implicit/*.cu
      utils/*.cu
      layout/*.cu
      )

  add_library(${namel}_cuda_${buildl}
      SHARED
      ${cu_src_files}
      )

  set_target_properties(${namel}_cuda_${buildl}
      PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      )

  target_link_libraries(${namel}_cuda_${buildl}
      PUBLIC
      CUDA::cudart
      CUDA::cudart_static
      ${KINTERA_CUDA_LIBRARY}
      ${TORCH_CUDA_LIBRARY}
      ${C10_CUDA_LIBRARY}
      )

  add_library(snapy::snap_cu ALIAS ${namel}_cuda_${buildl})
endif()

set(SNAP_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  CACHE INTERNAL "snap include directory")
