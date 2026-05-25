#include "../ggml/src/ggml-opencl/ggml-opencl-extra-metadata.hpp"
#include "testing.h"

int main() {
    testing t;

    t.test("quantized extras preserve their originating buffer metadata", [](testing & t) {
        ggml_opencl_extra_header base_header;
        ggml_opencl_extra_header_reset(base_header, GGML_OPENCL_EXTRA_MAGIC_BASE);

        const void * owner = reinterpret_cast<const void *>(0x1234);
        cl_mem base_buffer = reinterpret_cast<cl_mem>(0x5678);
        ggml_opencl_extra_header_bind_base(base_header, owner, base_buffer, 256);

        const auto base_view = ggml_opencl_extra_header_base_view(base_header, owner);
        t.assert_true("base metadata should be readable after initialization",
                      base_view.data_device == base_buffer && base_view.offset == 256);

        ggml_opencl_extra_header q4_0_header;
        ggml_opencl_extra_header_reset(q4_0_header, GGML_OPENCL_EXTRA_MAGIC_Q4_0);
        ggml_opencl_extra_header_bind_base(q4_0_header, owner, base_view.data_device, base_view.offset);

        const auto q4_0_view = ggml_opencl_extra_header_base_view(q4_0_header, owner);
        t.assert_true("q4_0 staging should retain the original base buffer and offset",
                      q4_0_view.data_device == base_buffer && q4_0_view.offset == 256);

        ggml_opencl_extra_header q6_k_header;
        ggml_opencl_extra_header_reset(q6_k_header, GGML_OPENCL_EXTRA_MAGIC_Q6_K);
        ggml_opencl_extra_header_bind_base(q6_k_header, owner, q4_0_view.data_device, q4_0_view.offset);

        const auto q6_k_view = ggml_opencl_extra_header_base_view(q6_k_header, owner);
        t.assert_true("repeated restaging should still resolve the original base buffer",
                      q6_k_view.data_device == base_buffer && q6_k_view.offset == 256);
    });

    t.test("owner mismatches invalidate stale extra metadata", [](testing & t) {
        ggml_opencl_extra_header header;
        ggml_opencl_extra_header_reset(header, GGML_OPENCL_EXTRA_MAGIC_Q4_1);
        ggml_opencl_extra_header_bind_base(
                header,
                reinterpret_cast<const void *>(0x1111),
                reinterpret_cast<cl_mem>(0x2222),
                512);

        const auto stale_view = ggml_opencl_extra_header_base_view(
                header,
                reinterpret_cast<const void *>(0x3333));
        t.assert_true("mismatched owners should invalidate the preserved base view",
                      stale_view.data_device == nullptr && stale_view.offset == 0);
    });

    t.test("base extras require owner match before reuse", [](testing & t) {
        ggml_opencl_extra_header header;
        ggml_opencl_extra_header_reset(header, GGML_OPENCL_EXTRA_MAGIC_BASE);

        const void * owner = reinterpret_cast<const void *>(0x4444);
        const void * other_owner = reinterpret_cast<const void *>(0x5555);
        const cl_mem base_buffer = reinterpret_cast<cl_mem>(0x6666);
        ggml_opencl_extra_header_bind_base(header, owner, base_buffer, 1024);

        t.assert_true("matching owner should allow base extra reuse",
                      ggml_opencl_extra_header_can_reuse_base_extra(header, owner));
        t.assert_true("mismatched owner should block base extra reuse",
                      !ggml_opencl_extra_header_can_reuse_base_extra(header, other_owner));
    });

    t.test("quantized extras require owner match before reuse", [](testing & t) {
        ggml_opencl_extra_header header;
        ggml_opencl_extra_header_reset(header, GGML_OPENCL_EXTRA_MAGIC_Q4_0);

        const void * owner = reinterpret_cast<const void *>(0x7777);
        const void * other_owner = reinterpret_cast<const void *>(0x8888);
        const cl_mem base_buffer = reinterpret_cast<cl_mem>(0x9999);
        ggml_opencl_extra_header_bind_base(header, owner, base_buffer, 2048);

        t.assert_true("matching owner should allow quantized extra reuse",
                      ggml_opencl_extra_header_can_reuse(header, owner));
        t.assert_true("mismatched owner should block quantized extra reuse",
                      !ggml_opencl_extra_header_can_reuse(header, other_owner));
    });

    t.test("extras require base-offset match before reuse", [](testing & t) {
        ggml_opencl_extra_header base_header;
        ggml_opencl_extra_header_reset(base_header, GGML_OPENCL_EXTRA_MAGIC_BASE);

        const void * owner = reinterpret_cast<const void *>(0xaaaa);
        const cl_mem base_buffer = reinterpret_cast<cl_mem>(0xbbbb);
        ggml_opencl_extra_header_bind_base(base_header, owner, base_buffer, 4096);

        t.assert_true("matching base offset should allow base extra reuse",
                      ggml_opencl_extra_header_can_reuse(base_header, owner, 4096));
        t.assert_true("different base offset should block base extra reuse",
                      !ggml_opencl_extra_header_can_reuse(base_header, owner, 8192));

        ggml_opencl_extra_header q4_0_header;
        ggml_opencl_extra_header_reset(q4_0_header, GGML_OPENCL_EXTRA_MAGIC_Q4_0);
        ggml_opencl_extra_header_bind_base(q4_0_header, owner, base_buffer, 12288);

        t.assert_true("matching base offset should allow quantized extra reuse",
                      ggml_opencl_extra_header_can_reuse(q4_0_header, owner, 12288));
        t.assert_true("different base offset should block quantized extra reuse",
                      !ggml_opencl_extra_header_can_reuse(q4_0_header, owner, 16384));
    });

    t.test("view offsets are applied on top of preserved base metadata", [](testing & t) {
        ggml_opencl_extra_header header;
        ggml_opencl_extra_header_reset(header, GGML_OPENCL_EXTRA_MAGIC_BASE);

        const void * owner = reinterpret_cast<const void *>(0x1357);
        const cl_mem base_buffer = reinterpret_cast<cl_mem>(0x2468);
        ggml_opencl_extra_header_bind_base(header, owner, base_buffer, 4096);

        const auto view = ggml_opencl_extra_header_view(header, owner, 1536);
        t.assert_true("view offsets must be added on top of the base offset",
                      view.data_device == base_buffer && view.offset == 5632);
    });

    return t.summary();
}
