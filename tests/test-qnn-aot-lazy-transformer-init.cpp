#include "../ggml/src/ggml-qnn/qnn/aot-init-policy.hpp"

#include <cstdlib>
#include <iostream>

int main() {
    unsetenv("GGML_QNN_AOT_LAZY_GRAPH_INIT");

    if (!qnn::qnn_aot_graph_family_uses_eager_init("transformers")) {
        std::cerr << "transformer family should use eager init by default\n";
        return 1;
    }

    if (!qnn::qnn_aot_graph_family_uses_eager_init("attention")) {
        std::cerr << "attention family unexpectedly lost eager init\n";
        return 1;
    }

    if (!qnn::qnn_aot_graph_family_uses_eager_init("lm_head")) {
        std::cerr << "lm_head family should use eager init by default\n";
        return 1;
    }

    setenv("GGML_QNN_AOT_LAZY_GRAPH_INIT", "1", 1);

    if (qnn::qnn_aot_graph_family_uses_eager_init("transformers")) {
        std::cerr << "transformer family should honor lazy init override\n";
        return 1;
    }

    if (qnn::qnn_aot_graph_family_uses_eager_init("lm_head")) {
        std::cerr << "lm_head family should honor lazy init override\n";
        return 1;
    }

    if (!qnn::qnn_aot_graph_family_uses_eager_init("attention")) {
        std::cerr << "attention family should stay eager under lazy override\n";
        return 1;
    }

    return 0;
}
