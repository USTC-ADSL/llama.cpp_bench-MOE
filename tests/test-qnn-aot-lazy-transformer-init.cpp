#include "../ggml/src/ggml-qnn/qnn/aot-init-policy.hpp"

#include <iostream>

int main() {
    if (qnn::qnn_aot_graph_family_uses_eager_init("transformers")) {
        std::cerr << "transformer family unexpectedly uses eager init\n";
        return 1;
    }

    if (!qnn::qnn_aot_graph_family_uses_eager_init("attention")) {
        std::cerr << "attention family unexpectedly lost eager init\n";
        return 1;
    }

    if (qnn::qnn_aot_graph_family_uses_eager_init("lm_head")) {
        std::cerr << "lm_head family unexpectedly uses eager init\n";
        return 1;
    }

    return 0;
}
