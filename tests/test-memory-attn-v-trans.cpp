#include "../src/llama-cparams.h"
#include "../src/llama-memory.h"

#include <cstdio>

static bool expect_eq(bool actual, bool expected, const char * label) {
    if (actual == expected) {
        return true;
    }

    std::fprintf(stderr,
            "%s: expected %d, got %d\n",
            label,
            expected ? 1 : 0,
            actual ? 1 : 0);
    return false;
}

int main(void) {
    llama_cparams cparams = {};
    llama_memory_params params = {};

    bool ok = true;

    cparams.flash_attn = true;
    ok &= expect_eq(
            llama_memory_resolve_attn_v_trans(params, cparams),
            false,
            "default flash-attn keeps V cache non-transposed");

    cparams.flash_attn = false;
    ok &= expect_eq(
            llama_memory_resolve_attn_v_trans(params, cparams),
            true,
            "default non-flash path transposes V cache");

    params.attn_v_trans_pinned = true;
    params.attn_v_trans = false;
    cparams.flash_attn = false;
    ok &= expect_eq(
            llama_memory_resolve_attn_v_trans(params, cparams),
            false,
            "pinned layout must ignore a later flash-attn disable");

    params.attn_v_trans = true;
    cparams.flash_attn = true;
    ok &= expect_eq(
            llama_memory_resolve_attn_v_trans(params, cparams),
            true,
            "pinned layout must ignore a later flash-attn enable");

    return ok ? 0 : 1;
}
