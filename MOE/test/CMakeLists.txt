cmake_minimum_required(VERSION 3.22.2)
project(buffer-capacity-probe C CXX)

if(NOT ANDROID)
    message(FATAL_ERROR "This project builds the Android HLOS probe; use the Android NDK toolchain")
endif()
if(NOT HEXAGON_SDK_ROOT)
    message(FATAL_ERROR "HEXAGON_SDK_ROOT is required for FastRPC IDL generation")
endif()

set(PREBUILT_LIB_DIR toolv19_v79 CACHE STRING "QAIRT/Hexagon prebuilt selector")
include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
include(ExternalProject)
find_package(OpenCL REQUIRED)

add_library(buffer_capacity_iface OBJECT
    ${CMAKE_CURRENT_BINARY_DIR}/buffer_capacity_iface_stub.c)
set_target_properties(buffer_capacity_iface PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(buffer_capacity_iface PRIVATE
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_BINARY_DIR})
build_idl(${CMAKE_CURRENT_SOURCE_DIR}/inc/buffer_capacity_iface.idl buffer_capacity_iface)

add_executable(buffer-capacity-probe
    src/buffer_capacity_probe.cpp
    $<TARGET_OBJECTS:buffer_capacity_iface>)
target_compile_features(buffer-capacity-probe PRIVATE cxx_std_17)
target_compile_definitions(buffer-capacity-probe PRIVATE CL_TARGET_OPENCL_VERSION=300)
target_include_directories(buffer-capacity-probe PRIVATE
    ${OpenCL_INCLUDE_DIRS}
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(buffer-capacity-probe PRIVATE ${OpenCL_LIBRARIES} dl log)

option(BUFFER_CAPACITY_BUILD_HTP "Build the v79 DSP skel with the HLOS probe" ON)
if(BUFFER_CAPACITY_BUILD_HTP)
    if(NOT HEXAGON_TOOLS_ROOT)
        message(FATAL_ERROR "HEXAGON_TOOLS_ROOT is required when BUFFER_CAPACITY_BUILD_HTP=ON")
    endif()
    ExternalProject_Add(buffer-capacity-htp-v79
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dsp
        BUILD_ALWAYS ON
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/libbuffer-capacity-htp-v79.so
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/dsp/cmake-toolchain.cmake
            -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
            -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}
            -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT}
            -DDSP_VERSION=v79)
endif()
