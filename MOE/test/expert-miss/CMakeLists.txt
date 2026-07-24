cmake_minimum_required(VERSION 3.22.2)

project(expert-miss-probe LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
    message(FATAL_ERROR "expert-miss-probe currently requires Linux or Android")
endif()

set(LLAMA_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_ROOT}/ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

add_library(expert-pack STATIC
    src/expert-pack.cpp)
target_compile_features(expert-pack PUBLIC cxx_std_17)
target_include_directories(expert-pack PUBLIC include)
target_link_libraries(expert-pack PUBLIC ggml-base)

function(expert_miss_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()
expert_miss_enable_warnings(expert-pack)

option(EXPERT_MISS_BUILD_GENERATOR "Build the host GGUF-to-pack generator" ON)
if(EXPERT_MISS_BUILD_GENERATOR)
    add_executable(expert-pack-gen
        src/expert-pack-gen.cpp)
    target_compile_features(expert-pack-gen PRIVATE cxx_std_17)
    target_link_libraries(expert-pack-gen PRIVATE expert-pack)
    expert_miss_enable_warnings(expert-pack-gen)
endif()

add_executable(expert-ufs-probe
    src/expert-ufs-probe.cpp)
target_compile_features(expert-ufs-probe PRIVATE cxx_std_17)
target_link_libraries(expert-ufs-probe PRIVATE expert-pack ggml)
expert_miss_enable_warnings(expert-ufs-probe)

include(CTest)
set(EXPERT_MISS_TEST_MODEL "" CACHE FILEPATH "Optional Phi-mini-MoE GGUF used by the integration test")
if(BUILD_TESTING AND NOT ANDROID AND EXPERT_MISS_BUILD_GENERATOR AND EXPERT_MISS_TEST_MODEL)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME expert-miss-cpu-pipeline
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_pipeline.py
            --generator $<TARGET_FILE:expert-pack-gen>
            --probe $<TARGET_FILE:expert-ufs-probe>
            --model ${EXPERT_MISS_TEST_MODEL})
endif()

