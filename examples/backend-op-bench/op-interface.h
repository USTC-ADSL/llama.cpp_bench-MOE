#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <cstdint>
#include <string>

// Base operator interface
struct OpInterface {
    virtual ~OpInterface() = default;

    // Get operator name
    virtual const char* name() const = 0;

    // Create tensors for this operator
    virtual void create_tensors(ggml_context* ctx) = 0;

    // Create computation graph
    virtual ggml_tensor* create_graph(ggml_context* ctx) = 0;

    // Fill input tensors with data
    virtual void fill_inputs() = 0;

    // Get output tensor
    virtual ggml_tensor* get_output() const = 0;

    // Get description (for logging)
    virtual std::string description() const = 0;
};
