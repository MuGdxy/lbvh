cmake_minimum_required(VERSION 3.20)
project(lbvh)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDA REQUIRED)
enable_language(CUDA)

option(LBVH_BUILD_EXAMPLES "Build tests" ON)

add_library(lbvh INTERFACE)
target_include_directories(lbvh INTERFACE "${PROJECT_SOURCE_DIR}/include")
file(GLOB_RECURSE LBVH_HEADERS "${PROJECT_SOURCE_DIR}/include/lbvh/*.cuh")
source_group(TREE "${PROJECT_SOURCE_DIR}/include/" FILES ${LBVH_HEADERS})

# lambda & constexpr support
target_compile_options(lbvh INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
if(LBVH_BUILD_EXAMPLES)
    add_executable(lbvh_example)
    target_link_libraries(lbvh_example PRIVATE lbvh)
    file(GLOB_RECURSE LBVH_EXAMPLE_CU_SOURCE "${PROJECT_SOURCE_DIR}/examples/*.cu")
    file(GLOB_RECURSE LBVH_EXAMPLE_CPP_SOURCE "${PROJECT_SOURCE_DIR}/examples/*.cpp")
    target_sources(lbvh_example PRIVATE "${LBVH_EXAMPLE_CU_SOURCE}" "${LBVH_EXAMPLE_CPP_SOURCE}" "${LBVH_HEADERS}")
    source_group(TREE "${PROJECT_SOURCE_DIR}/include/" FILES ${LBVH_HEADERS})
    source_group(TREE "${PROJECT_SOURCE_DIR}/" FILES ${LBVH_EXAMPLE_CU_SOURCE} ${LBVH_EXAMPLE_CPP_SOURCE})
endif()
