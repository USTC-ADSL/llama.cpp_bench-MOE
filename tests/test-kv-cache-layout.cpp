#include "../src/llama-kv-cache.h"
#include "testing.h"

#include <cstdint>
#include <vector>

static std::vector<uint8_t> layout_from_rows(
        const std::vector<uint8_t> & rows,
        bool transposed,
        uint32_t kv_size,
        uint32_t n_embd,
        size_t elem_size) {
    std::vector<uint8_t> layout(rows.size(), 0);
    const uint32_t n_cells = rows.size() / (n_embd * elem_size);

    for (uint32_t cell = 0; cell < n_cells; ++cell) {
        for (uint32_t elem = 0; elem < n_embd; ++elem) {
            const size_t src = (cell * n_embd + elem) * elem_size;
            const size_t dst = llama_kv_cache_v_offset(transposed, cell, elem, kv_size, n_embd, elem_size);
            for (size_t b = 0; b < elem_size; ++b) {
                layout[dst + b] = rows[src + b];
            }
        }
    }

    return layout;
}

static std::vector<uint8_t> rows_from_layout(
        const std::vector<uint8_t> & layout,
        bool transposed,
        uint32_t kv_size,
        uint32_t n_embd,
        size_t elem_size) {
    std::vector<uint8_t> rows(layout.size(), 0);
    const uint32_t n_cells = rows.size() / (n_embd * elem_size);

    for (uint32_t cell = 0; cell < n_cells; ++cell) {
        for (uint32_t elem = 0; elem < n_embd; ++elem) {
            const size_t src = llama_kv_cache_v_offset(transposed, cell, elem, kv_size, n_embd, elem_size);
            const size_t dst = (cell * n_embd + elem) * elem_size;
            for (size_t b = 0; b < elem_size; ++b) {
                rows[dst + b] = layout[src + b];
            }
        }
    }

    return rows;
}

int main() {
    testing t;

    t.test("V cache offset maps canonical rows and transposed columns", [](testing & t) {
        t.assert_equal("row-major V offset should address cell rows",
                (size_t) 14,
                llama_kv_cache_v_offset(false, 2, 1, 4, 3, 2));
        t.assert_equal("transposed V offset should address embedding columns",
                (size_t) 12,
                llama_kv_cache_v_offset(true, 2, 1, 4, 3, 2));
    });

    t.test("V cache layout conversion preserves canonical rows", [](testing & t) {
        const uint32_t kv_size = 4;
        const uint32_t n_embd = 3;
        const size_t elem_size = 2;

        std::vector<uint8_t> canonical = {
            0x00, 0x10, 0x01, 0x11, 0x02, 0x12,
            0x03, 0x13, 0x04, 0x14, 0x05, 0x15,
            0x06, 0x16, 0x07, 0x17, 0x08, 0x18,
            0x09, 0x19, 0x0a, 0x1a, 0x0b, 0x1b,
        };

        const std::vector<uint8_t> transposed =
            layout_from_rows(canonical, true, kv_size, n_embd, elem_size);
        const std::vector<uint8_t> restored =
            rows_from_layout(transposed, true, kv_size, n_embd, elem_size);

        t.assert_true("transposed V layout should round-trip through canonical rows",
                restored == canonical);
    });

    return t.summary();
}
