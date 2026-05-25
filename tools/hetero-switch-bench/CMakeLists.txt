set(TARGET hetero-switch-bench)

find_package(OpenCL REQUIRED)

add_executable(${TARGET} hetero-switch-bench.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE ${OpenCL_LIBRARIES})

if (UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} PRIVATE dl)
endif()

if (ANDROID)
    find_library(LOG_LIB log)
    if (LOG_LIB)
        target_link_libraries(${TARGET} PRIVATE ${LOG_LIB})
    endif()
endif()

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
