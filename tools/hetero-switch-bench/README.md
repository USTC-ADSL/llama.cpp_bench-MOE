# hetero-switch-bench

Standalone Workflow2 microbenchmark for OpenCL shared-host-ptr overheads (QNN/RPCMEM-style host shared buffer vs memcpy path).

## What it measures

For each size (default: 1KB, 64KB, 1MB, 16MB), and for each mode:

- `shared_host_ptr`: rpcmem allocation + OpenCL buffer with `CL_MEM_USE_HOST_PTR`
- `memcpy`: regular device buffer with explicit `clEnqueueWriteBuffer` / `clEnqueueReadBuffer`

It benchmarks and validates:

1. `host_write_to_opencl_read`
2. `opencl_write_to_host_read`

CSV columns:

- `mode,flow,size_bytes,iter,latency_us,throughput_gbps,valid`

## Build

Built as target `hetero-switch-bench` when `GGML_OPENCL=ON`.

## Usage

```bash
./bin/hetero-switch-bench --warmup 5 --iters 50 --csv /data/local/tmp/wf2.csv
```

Optional:

```bash
--sizes 1024,65536,1048576,16777216
```

## Runtime prerequisites (Android)

- OpenCL runtime available
- ADSP rpcmem library resolvable (`libcdsprpc.so` or `libadsprpc.so`)
- Proper environment variables (e.g., `LD_LIBRARY_PATH`, `ADSP_LIBRARY_PATH`) depending on device image
