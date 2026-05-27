# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# examples

if (EMSCRIPTEN)
else()
    add_subdirectory(gguf)
    add_subdirectory(gguf-hash)
    add_subdirectory(stage-profiler)
    add_subdirectory(backend-op-bench)
endif()
