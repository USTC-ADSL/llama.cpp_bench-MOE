#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cl_int code, const char * stage) {
    if (code != CL_SUCCESS) throw std::runtime_error(std::string(stage) + ": " + std::to_string(code));
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try {
        std::ifstream input(argv[1]);
        std::string source((std::istreambuf_iterator<char>(input)), {});
        const auto start = source.find("kernel void kernel_convert_block_q4_0_trans4_ns(");
        const auto end = source.find("kernel void kernel_restore_block_q4_0_trans4_ns(", start);
        if (start == std::string::npos || end == std::string::npos) throw std::runtime_error("kernel not found");
        const std::string original = source.substr(start, end-start);
        const std::string prefix = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n#define QK4_0 32\nstruct block_q4_0 { half d; uchar qs[16]; };\n";
        cl_platform_id platform;
        cl_device_id device;
        check(clGetPlatformIDs(1, &platform, nullptr), "platform");
        check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "device");
        cl_int err;
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        check(err, "context");
        cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        check(err, "queue");
        const cl_uint K = 2048, M = 64, E = 3;
        const size_t blocks = size_t(K/32)*M*E;
        std::vector<uint8_t> packed(blocks*18), expected_q(blocks*16);
        std::vector<uint16_t> expected_d(blocks);
        uint32_t random = 0x12345678;
        for (size_t b=0; b<blocks; ++b) {
            uint16_t scale = uint16_t(0x3000 + b%0x1000);
            std::memcpy(packed.data()+b*18, &scale, 2);
            for (int i=0; i<16; ++i) {
                random ^= random << 13; random ^= random >> 17; random ^= random << 5;
                packed[b*18+2+i] = uint8_t(random);
            }
            const size_t kblock = b % (K/32), row = (b/(K/32))%M, expert = b/((K/32)*M);
            expected_d[row+kblock*M+expert*(K/32)*M] = scale;
            uint8_t reordered[16];
            for (int i=0; i<8; ++i) {
                const auto x0 = packed[b*18+2+2*i], x1 = packed[b*18+3+2*i];
                reordered[i] = (x0&15) | ((x1&15)<<4);
                reordered[i+8] = (x0>>4) | (x1&240);
            }
            for (size_t plane=0; plane<4; ++plane) {
                const size_t word = expert*(K/32)*M*4 + kblock*M*4 + row + plane*M;
                std::memcpy(expected_q.data()+word*4, reordered+plane*4, 4);
            }
        }
        cl_mem in = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, packed.size(), packed.data(), &err);
        check(err, "input");
        cl_mem q = clCreateBuffer(context, CL_MEM_READ_WRITE, expected_q.size(), nullptr, &err);
        check(err, "q");
        cl_mem d = clCreateBuffer(context, CL_MEM_READ_WRITE, expected_d.size()*2, nullptr, &err);
        check(err, "d");
        for (const std::string variant : {"original", "vload8", "direct_read", "scalar_build", "scalar_private"}) {
            std::string code = original;
            const std::string unsafe = "((global ushort8 *)(&(b->qs[0])))[0]";
            if (variant == "vload8") code.replace(code.find(unsafe), unsafe.size(), "vload8(0, (global ushort *)(&(b->qs[0])))");
            if (variant == "direct_read") {
                for (const std::string index : {"2*i + 0", "2*i + 1"}) {
                    const auto from = "pre_block_ptr[" + index + "]";
                    code.replace(code.find(from), from.size(), "b->qs[" + index + "]");
                }
            }
            if (variant == "scalar_private" || variant == "scalar_build") {
                const auto first = code.find("ushort8 pre_block =");
                const auto last = code.find("uint offset =", first);
                std::string scalar =
                    "uint q0=0, q1=0, q2=0, q3=0;\n"
                    "for (uint i=0; i<8; ++i) {\n"
                    " uint x0=b->qs[2*i], x1=b->qs[2*i+1];\n"
                    " uint lo=(x0&15)|((x1&15)<<4), hi=(x0>>4)|(x1&240);\n"
                    " if (i<4) { q0 |= lo << (i*8); q2 |= hi << (i*8); }\n"
                    " else { q1 |= lo << ((i-4)*8); q3 |= hi << ((i-4)*8); }\n"
                    "}\nuint4 q_block=(uint4)(q0,q1,q2,q3);\n";
                if (variant == "scalar_build") {
                    scalar = "ushort8 pre_block = " + unsafe + ";\nuchar * pre_block_ptr=(uchar *)(&pre_block);\n" + scalar;
                    for (const std::string index : {"2*i", "2*i+1"}) {
                        const auto from = "b->qs[" + index + "]";
                        scalar.replace(scalar.find(from), from.size(), "pre_block_ptr[" + index + "]");
                    }
                }
                code.replace(first, last-first, scalar);
            }
            code = prefix+code;
            const char * ptr = code.c_str();
            cl_program program = clCreateProgramWithSource(context, 1, &ptr, nullptr, &err);
            check(err, "program");
            err = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", nullptr, nullptr);
            if (err) {
                size_t size=0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
                std::string log(size, '\0'); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
                std::cerr << log << '\n'; check(err, "build");
            }
            cl_kernel kernel = clCreateKernel(program, "kernel_convert_block_q4_0_trans4_ns", &err);
            check(err, "kernel");
            check(clSetKernelArg(kernel, 0, sizeof(in), &in), "arg0");
            check(clSetKernelArg(kernel, 1, sizeof(q), &q), "arg1");
            check(clSetKernelArg(kernel, 2, sizeof(d), &d), "arg2");
            check(clSetKernelArg(kernel, 3, sizeof(K), &K), "arg3");
            check(clSetKernelArg(kernel, 4, sizeof(M), &M), "arg4");
            const size_t global[] = {M,K/32,E}, local[] = {64,1,1};
            check(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr), "run");
            std::vector<uint8_t> actual_q(expected_q.size());
            std::vector<uint16_t> actual_d(expected_d.size());
            check(clEnqueueReadBuffer(queue, q, CL_TRUE, 0, actual_q.size(), actual_q.data(), 0, nullptr, nullptr), "read_q");
            check(clEnqueueReadBuffer(queue, d, CL_TRUE, 0, actual_d.size()*2, actual_d.data(), 0, nullptr, nullptr), "read_d");
            size_t q_errors=0, d_errors=0;
            for (size_t i=0; i<actual_q.size(); ++i) q_errors += actual_q[i]!=expected_q[i];
            for (size_t i=0; i<actual_d.size(); ++i) d_errors += actual_d[i]!=expected_d[i];
            std::cout << "{\"variant\":\"" << variant << "\",\"q_bytes\":" << actual_q.size()
                      << ",\"q_mismatches\":" << q_errors << ",\"scale_count\":" << actual_d.size()
                      << ",\"scale_mismatches\":" << d_errors << "}" << std::endl;
            clReleaseKernel(kernel); clReleaseProgram(program);
        }
        clReleaseMemObject(in); clReleaseMemObject(q); clReleaseMemObject(d);
        clReleaseCommandQueue(queue); clReleaseContext(context);
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
