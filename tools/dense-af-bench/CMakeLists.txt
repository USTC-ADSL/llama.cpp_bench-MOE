add_executable(dense-af-bench dense-af-bench.cpp)
target_link_libraries(dense-af-bench PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(dense-af-bench PRIVATE cxx_std_17)
