#pragma once
#include "ggml-accel-budget.h"
#include <CL/cl.h>
#include <algorithm>
#ifndef _WIN32
#include <unistd.h>
#endif

// Included after CL declarations, before backend call sites. No release interception:
// the OpenCL implementation knows when children and queued commands stop retaining storage.
static uint64_t ggml_cl_budget_bytes(size_t size) {
    #ifdef _WIN32
    static const long page = 4096;
    #else
    static const long page = sysconf(_SC_PAGESIZE);
    #endif
    return ggml_budget_round(size, page > 0 ? uint64_t(page) : 4096);
}
static cl_mem ggml_cl_budget_finish(cl_mem mem, uint64_t ticket, cl_int * error) {
    if (!mem) { ggml_budget_cancel(ticket); return nullptr; }
    if (!ggml_budget_commit(ticket, reinterpret_cast<uintptr_t>(mem))) {
        clReleaseMemObject(mem); ggml_budget_cancel(ticket);
        if (error) *error = CL_OUT_OF_HOST_MEMORY;
        return nullptr;
    }
    cl_int err = clSetMemObjectDestructorCallback(mem, ggml_budget_cl_destroy,
                                                reinterpret_cast<void *>(uintptr_t(ticket)));
    if (err != CL_SUCCESS) {
        // This object has not escaped the allocation wrapper: no caller retain or
        // queued command can reference it. A successful release destroys this sole
        // explicit reference (driver caching remains a separately unknown residual).
        ggml_budget_gap();
        if (clReleaseMemObject(mem) == CL_SUCCESS) ggml_budget_release(ticket);
        if (error) *error = err;
        return nullptr;
    }
    return mem;
}
static uint64_t ggml_cl_budget_begin(size_t size, int coverage, const char * stage, cl_int * error) {
    if (!ggml_budget_pin_callback()) { ggml_budget_gap(); if (error) *error = CL_OUT_OF_HOST_MEMORY; return 0; }
    uint64_t ticket = ggml_budget_reserve(GGML_BUDGET_OPENCL, ggml_cl_budget_bytes(size), coverage, stage);
    if (!ticket && error) *error = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    return ticket;
}
static cl_mem ggml_budget_clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size, void * host, cl_int * error) {
    if (!ggml_budget_mode()) return clCreateBuffer(ctx, flags, size, host, error);
    int coverage = flags & CL_MEM_USE_HOST_PTR ? GGML_BUDGET_UNKNOWN : GGML_BUDGET_DEFAULT;
    auto ticket = ggml_cl_budget_begin(size, coverage, "buffer", error);
    if (!ticket) return nullptr;
    return ggml_cl_budget_finish(clCreateBuffer(ctx, flags, size, host, error), ticket, error);
}
#if CL_TARGET_OPENCL_VERSION >= 300
static cl_mem ggml_budget_clCreateBufferWithProperties(cl_context ctx, const cl_mem_properties * props,
        cl_mem_flags flags, size_t size, void * host, cl_int * error) {
    if (!ggml_budget_mode()) return clCreateBufferWithProperties(ctx, props, flags, size, host, error);
    auto ticket = ggml_cl_budget_begin(size, flags & CL_MEM_USE_HOST_PTR ? GGML_BUDGET_UNKNOWN : GGML_BUDGET_DEFAULT,
                                      "buffer_properties", error);
    if (!ticket) return nullptr;
    return ggml_cl_budget_finish(clCreateBufferWithProperties(ctx, props, flags, size, host, error), ticket, error);
}
#endif
static cl_mem ggml_cl_budget_alias_finish(cl_mem mem, cl_mem parent, cl_int * error) {
    if (!mem) return mem;
    uint64_t ticket = ggml_budget_alias(GGML_BUDGET_OPENCL, reinterpret_cast<uintptr_t>(parent), reinterpret_cast<uintptr_t>(mem));
    if (!ticket) { clReleaseMemObject(mem); if (error) *error = CL_INVALID_MEM_OBJECT; return nullptr; }
    return ggml_cl_budget_finish(mem, ticket, error);
}
static cl_mem ggml_budget_clCreateSubBuffer(cl_mem parent, cl_mem_flags flags, cl_buffer_create_type type,
        const void * region, cl_int * error) {
    cl_mem mem = clCreateSubBuffer(parent, flags, type, region, error);
    return ggml_budget_mode() ? ggml_cl_budget_alias_finish(mem, parent, error) : mem;
}
static uint64_t ggml_cl_image_nominal(const cl_image_format * fmt, const cl_image_desc * desc) {
    if (!fmt || !desc) return 0;
    uint64_t channels = 0, bytes = 0;
    switch (fmt->image_channel_order) {
        case CL_R: case CL_A: case CL_INTENSITY: case CL_LUMINANCE: channels = 1; break;
        case CL_RG: case CL_RA: case CL_Rx: channels = 2; break;
        case CL_RGB: case CL_RGx: channels = 3; break;
        case CL_RGBA: case CL_BGRA: case CL_ARGB: case CL_RGBx: channels = 4; break;
        default: return 0;
    }
    switch (fmt->image_channel_data_type) {
        case CL_SNORM_INT8: case CL_UNORM_INT8: case CL_SIGNED_INT8: case CL_UNSIGNED_INT8: bytes = 1; break;
        case CL_SNORM_INT16: case CL_UNORM_INT16: case CL_SIGNED_INT16: case CL_UNSIGNED_INT16: case CL_HALF_FLOAT: bytes = 2; break;
        case CL_SIGNED_INT32: case CL_UNSIGNED_INT32: case CL_FLOAT: bytes = 4; break;
        default: return 0;
    }
    uint64_t n = channels * bytes;
    for (size_t x : {desc->image_width, std::max(size_t(1), desc->image_height),
                     std::max(size_t(1), desc->image_depth), std::max(size_t(1), desc->image_array_size)}) {
        if (x && n > UINT64_MAX / x) return UINT64_MAX;
        n *= x;
    }
    return n;
}
static cl_mem ggml_budget_clCreateImage(cl_context ctx, cl_mem_flags flags, const cl_image_format * fmt,
        const cl_image_desc * desc, void * host, cl_int * error) {
    if (!ggml_budget_mode()) return clCreateImage(ctx, flags, fmt, desc, host, error);
    if (desc && desc->buffer && desc->image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
        return ggml_cl_budget_alias_finish(clCreateImage(ctx, flags, fmt, desc, host, error), desc->buffer, error);
    }
    // OpenCL exposes no pre-allocation physical pitch for independent tiled images.
    // Even CL_MEM_SIZE can exceed width*height*pixel_size (Adreno metadata/padding).
    // Enforce is explicitly unsupported here, rather than allocating over a cap
    // before discovering the extra capacity. Preserve the caller's error handling.
    ggml_budget_gap();
    if (ggml_budget_mode()==2) { if(error) *error=CL_INVALID_OPERATION; return nullptr; }
    uint64_t bytes = ggml_cl_image_nominal(fmt, desc);
    auto ticket = ggml_cl_budget_begin(bytes, GGML_BUDGET_UNKNOWN, "image_unknown_layout", error);
    if (!ticket) return nullptr;
    auto mem=clCreateImage(ctx, flags, fmt, desc, host, error);
    if(mem) {
        size_t actual=0;
        if(clGetMemObjectInfo(mem,CL_MEM_SIZE,sizeof(actual),&actual,nullptr)!=CL_SUCCESS ||
           !ggml_budget_observe_size(ticket,ggml_cl_budget_bytes(actual))) {
            clReleaseMemObject(mem);ggml_budget_cancel(ticket);
            if(error) *error=CL_OUT_OF_HOST_MEMORY;
            return nullptr;
        }
    }
    return ggml_cl_budget_finish(mem, ticket, error);
}
#define clCreateBuffer ggml_budget_clCreateBuffer
#define clCreateSubBuffer ggml_budget_clCreateSubBuffer
#define clCreateImage ggml_budget_clCreateImage
#if CL_TARGET_OPENCL_VERSION >= 300
#define clCreateBufferWithProperties ggml_budget_clCreateBufferWithProperties
#endif
