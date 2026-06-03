| case | category | backend | prefill | decode | rc | semantic_ok | route_ok | prompt_ms | eval_ms | response_excerpt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_cpu | single_backend | cpu |  |  | 1 | 0 | 1 |  |  |   |
| single_opencl | single_backend | opencl |  |  | 1 | 0 | 1 |  |  |   |
| single_qnn | single_backend | qnn |  |  | 1 | 0 | 1 |  |  |   |
| switch_cpu_to_opencl | switch |  | cpu | opencl | 1 | 0 | 0 |  |  |   |
| switch_cpu_to_qnn | switch |  | cpu | qnn-npu | 1 | 0 | 0 |  |  |   |
| switch_opencl_to_cpu | switch |  | opencl | cpu | 1 | 0 | 0 |  |  |   |
| switch_opencl_to_qnn | switch |  | opencl | qnn-npu | 1 | 0 | 0 |  |  |   |
| switch_qnn_to_cpu | switch |  | qnn-npu | cpu | 1 | 0 | 0 |  |  |   |
| switch_qnn_to_opencl | switch |  | qnn-npu | opencl | 1 | 0 | 0 |  |  |   |
