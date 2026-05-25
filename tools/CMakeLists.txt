# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    if (GGML_OPENCL)
        add_subdirectory(hetero-switch-bench)
    endif()
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    add_subdirectory(qnn-kv-export)
    add_subdirectory(tokenize)
    add_subdirectory(parser)
    if (NOT GGML_BACKEND_DL)
        # these examples use the backends directly and cannot be built with dynamic loading
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    add_subdirectory(results)
endif()
