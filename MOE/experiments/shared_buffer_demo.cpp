#include "raw_buffer_view.h"
#include <iostream>

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "shared-buffer-demo [bytes]\nCreates one CPU/GPU/HTP allocation and imports it without copying.\n";
        return 0;
    }
    try {
        const size_t bytes = argc == 2 ? std::stoull(argv[1]) : 4096;
        auto buffer = std::make_shared<moe::SharedDmaBuffer>(bytes);
        auto gpu = std::make_shared<moe::OpenClDevice>();
        moe::OpenClDmaView gpu_view(gpu, buffer);
        moe::enable_unsigned_htp(*buffer->session());
        moe::HtpMapping htp_view(buffer);
        std::cout << "bytes=" << buffer->size() << " allocation=" << buffer->allocation_id()
                  << " gpu_import=" << gpu_view.method() << " htp_mapping=delayed\n";
        htp_view.close();
        gpu_view.close();
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
