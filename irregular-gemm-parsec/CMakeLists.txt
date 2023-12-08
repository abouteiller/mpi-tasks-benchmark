cmake_minimum_required(VERSION 3.12)
project(irregular-gemm-parsec)

if(NOT TARGET PaRSEC::parsec OR NOT TARGET PaRSEC::parsec-ptgpp)
  find_package(PaRSEC CONFIG HINTS "${PaRSEC_ROOT}" REQUIRED)
  if(NOT TARGET PaRSEC::parsec)
    message(FATAL_ERROR "User requested PaRSEC not found with PaRSEC_ROOT=${PaRSEC_ROOT} (no PaRSEC::parsec target)")
  elseif(NOT TARGET PaRSEC::parsec-ptgpp)
    message(FATAL_ERROR "User requested PaRSEC not found with PaRSEC_ROOT=${PaRSEC_ROOT} (PaRSEC::parsec target exists, but not PaRSEC::parsec-ptgpp)")
  endif()
endif()

option(IGGOP_HAVE_CUDA "Enable building CUDA IGGOP" ${PARSEC_HAVE_DEV_CUDA_SUPPORT})
if(IGGOP_HAVE_CUDA AND PARSEC_HAVE_DEV_CUDA_SUPPORT)
  find_package(CUDA REQUIRED)
endif()

option(IGGOP_HAVE_HIP "Enable building HIP IGGOP" ${PARSEC_HAVE_DEV_HIP_SUPPORT})
if(IGGOP_HAVE_HIP AND PARSEC_HAVE_DEV_HIP_SUPPORT)
  find_package(hipblas HINTS ENV{ROCM_PATH} REQUIRED)
  find_package(rocsolver HINTS ENV{ROCM_PATH} REQUIRED)
endif()

include(ParsecCompilePTG)
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

set(BIN_SOURCES testing_dgemm_irr_sparse.c)
set(SOURCES irr_bs_tm.c gemm_irr_sparse.c dgemm_irr_sparse_wrapper.c)
set(JDF_SOURCES dgemm_irr_sparse_genB.jdf irr_bs_tm_init.jdf)

add_library(iggop OBJECT ${SOURCES})
target_ptg_sources(iggop PRIVATE "${JDF_SOURCES}")
target_compile_definitions(iggop PUBLIC $<$<BOOL:${IGGOP_HAVE_CUDA}>:IGGOP_HAVE_CUDA> $<$<BOOL:${IGGOP_HAVE_HIP}>:IGGOP_HAVE_HIP>)

target_link_libraries(iggop
                      PUBLIC PaRSEC::parsec ${EXTRA_LIBS}
                      $<$<BOOL:${IGGOP_HAVE_CUDA}>:${CUDA_cublas_LIBRARY}>
                      $<$<BOOL:${IGGOP_HAVE_HIP}>:roc::hipblas>)

foreach(test ${BIN_SOURCES})
  string(REPLACE "\.c" "" testbin "${test}")
  add_executable(${testbin} ${test})
  target_link_libraries(${testbin} PRIVATE iggop)
endforeach()
