cmake_minimum_required(VERSION 3.22)
project(bare_hopper LANGUAGES CXX CUDA)
find_package(CUDAToolkit 12.3 REQUIRED)
find_package(Git REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CUTLASS_ROOT "${CMAKE_SOURCE_DIR}/.deps/cutlass" CACHE PATH "CUTLASS v3.9.2 checkout")
set(CUTLASS_REVISION ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e)
execute_process(COMMAND ${GIT_EXECUTABLE} -C "${CUTLASS_ROOT}" rev-parse HEAD
  OUTPUT_VARIABLE cutlass_head OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE git_status)
if(NOT git_status EQUAL 0 OR NOT cutlass_head STREQUAL CUTLASS_REVISION)
  message(FATAL_ERROR "Run scripts/setup_cutlass.sh: expected CUTLASS ${CUTLASS_REVISION}")
endif()
execute_process(COMMAND ${GIT_EXECUTABLE} -C "${CUTLASS_ROOT}" status --porcelain --untracked-files=no
  OUTPUT_VARIABLE cutlass_dirty OUTPUT_STRIP_TRAILING_WHITESPACE)
if(cutlass_dirty)
  message(FATAL_ERROR "CUTLASS tracked files are modified; reference must be pristine")
endif()

set(BARE_CANDIDATE_SOURCE "${CMAKE_SOURCE_DIR}/src/candidate.cu" CACHE FILEPATH
  "CUDA-only implementation of candidate.hpp; no CUTLASS/CuTe include paths")
add_library(bare_candidate OBJECT "${BARE_CANDIDATE_SOURCE}")
target_include_directories(bare_candidate PRIVATE include)
add_executable(gemm_bench src/gemm_bench.cu src/reference.cu $<TARGET_OBJECTS:bare_candidate>)
target_include_directories(gemm_bench PRIVATE include "${CUTLASS_ROOT}/include" "${CUTLASS_ROOT}/tools/util/include")
target_link_libraries(gemm_bench PRIVATE CUDA::cublas CUDA::cuda_driver)
target_compile_definitions(gemm_bench PRIVATE BARE_CUTLASS_REVISION="${CUTLASS_REVISION}")
foreach(target bare_candidate gemm_bench)
  set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES OFF)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_90a,code=sm_90a>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endforeach()
install(TARGETS gemm_bench RUNTIME DESTINATION bin)
