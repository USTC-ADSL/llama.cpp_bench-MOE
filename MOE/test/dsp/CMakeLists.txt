cmake_minimum_required(VERSION 3.22.2)
project(buffer-capacity-htp C ASM)

if(NOT HEXAGON_SDK_ROOT)
    message(FATAL_ERROR "HEXAGON_SDK_ROOT is required")
endif()

set(PREBUILT_LIB_DIR toolv19_v79 CACHE STRING "QAIRT/Hexagon prebuilt selector")
include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

set(DSP_VERSION v79 CACHE STRING "Hexagon architecture")
set(HTP_LIB buffer-capacity-htp-${DSP_VERSION})

add_library(${HTP_LIB} SHARED
    buffer_capacity_iface_imp.c
    ${CMAKE_CURRENT_BINARY_DIR}/buffer_capacity_iface_skel.c)

target_include_directories(${HTP_LIB} PRIVATE
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_BINARY_DIR})

build_idl(${CMAKE_CURRENT_SOURCE_DIR}/../inc/buffer_capacity_iface.idl ${HTP_LIB})
target_compile_definitions(${HTP_LIB} PRIVATE NDEBUG=1 FARF_HIGH=1)
install(TARGETS ${HTP_LIB} LIBRARY DESTINATION .)
