| route | workload | rc | devices | n_prompt | n_gen | avg_ms | tokens/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | pp128_tg0 | 0 | none | 128 | 0 | 16117.565 | 7.941646 |
| cpu | pp256_tg0 | 0 | none | 256 | 0 | 31774.438 | 8.056791 |
| cpu | pp512_tg0 | 0 | none | 512 | 0 | 60951.911 | 8.400065 |
| cpu | pp0_tg1 | 1 |  |  |  |  |  |
| cpu | pp0_tg32 | 1 |  |  |  |  |  |
| cpu | pp0_tg128 | 1 |  |  |  |  |  |
| cpu | pp0_tg256 | 1 |  |  |  |  |  |
| opencl | pp128_tg0 | 0 | GPUOpenCL | 128 | 0 | 443.398 | 288.679750 |
| opencl | pp256_tg0 | 0 | GPUOpenCL | 256 | 0 | 806.133 | 317.565394 |
| opencl | pp512_tg0 | 0 | GPUOpenCL | 512 | 0 | 1572.946 | 325.503920 |
| opencl | pp0_tg1 | 1 |  |  |  |  |  |
| opencl | pp0_tg32 | 1 |  |  |  |  |  |
| opencl | pp0_tg128 | 1 |  |  |  |  |  |
| opencl | pp0_tg256 | 1 |  |  |  |  |  |
| qnn | pp128_tg0 | 0 | qnn-npu | 128 | 0 | 1235.502 | 103.601597 |
| qnn | pp256_tg0 | 0 | qnn-npu | 256 | 0 | 842.627 | 303.811796 |
| qnn | pp512_tg0 | 0 | qnn-npu | 512 | 0 | 1094.167 | 467.936025 |
| qnn | pp0_tg1 | 1 |  |  |  |  |  |
| qnn | pp0_tg32 | 1 |  |  |  |  |  |
| qnn | pp0_tg128 | 1 |  |  |  |  |  |
| qnn | pp0_tg256 | 1 |  |  |  |  |  |
| cpu->opencl | pp256_tg1 | 0 | GPUOpenCL | 256 | 1 | 73281.636 | 3.507018 |
| cpu->qnn-npu | pp256_tg1 | 0 | qnn-npu | 256 | 1 | 71496.021 | 3.594606 |
| opencl->cpu | pp256_tg1 | 0 | GPUOpenCL | 256 | 1 | 4941.226 | 52.011380 |
| opencl->qnn-npu | pp256_tg1 | 0 | qnn-npu | 256 | 1 | 1423.511 | 180.539540 |
| opencl->qnn-npu | pp256_tg1 | 0 | GPUOpenCL | 256 | 1 | 4089.619 | 62.842047 |
| qnn-npu->cpu | pp256_tg1 | 0 | qnn-npu | 256 | 1 | 1916.028 | 134.131672 |
| qnn-npu->opencl | pp256_tg1 | 0 | qnn-npu | 256 | 1 | 1188.606 | 216.219741 |
| qnn-npu->opencl | pp256_tg1 | 0 | GPUOpenCL | 256 | 1 | 3131.239 | 82.076140 |
