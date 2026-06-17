#include "testing.h"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cstdint>

int main() {
    testing t;

    t.test("gallocr refreshes stale view buffer metadata from backing tensor", [](testing & t) {
        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead()*8 + ggml_graph_overhead_custom(8, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);

        ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_set_name(src, "src");
        ggml_tensor * view = ggml_view_1d(ctx, src, 4, sizeof(float));
        ggml_set_name(view, "view");
        ggml_tensor * out = ggml_cont(ctx, view);
        ggml_set_name(out, "out");

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
        ggml_build_forward_expand(graph, out);

        ggml_backend_buffer_t old_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 1024);
        ggml_backend_buffer_t new_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 1024);

        src->buffer = new_buf;
        src->data = ggml_backend_buffer_get_base(new_buf);
        view->buffer = old_buf;
        view->data = (uint8_t *) ggml_backend_buffer_get_base(old_buf) + view->view_offs;

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        const bool allocated = ggml_gallocr_alloc_graph(galloc, graph);

        t.assert_true("graph allocation should succeed", allocated);
        t.assert_true("view buffer should follow its backing tensor", view->buffer == src->buffer);
        t.assert_true("view data should follow backing tensor plus view offset",
                view->data == (uint8_t *) src->data + view->view_offs);

        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(old_buf);
        ggml_backend_buffer_free(new_buf);
        ggml_free(ctx);
    });

    return t.summary();
}
