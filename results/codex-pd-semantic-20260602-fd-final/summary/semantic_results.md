| case | category | backend | prefill | decode | rc | semantic_ok | route_ok | prompt_ms | eval_ms | response_excerpt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_cpu | single_backend | cpu |  |  | 0 | 1 | 1 |  |  | Rayleigh scattering. You are an AI  |
| single_opencl | single_backend | opencl |  |  | 0 | 1 | 1 |  |  | Rayleigh scattering. You are an AI  |
| single_qnn | single_backend | qnn |  |  | 0 | 1 | 1 |  |  | Absorption. Scattering. Absorption  |
| switch_cpu_to_opencl | switch |  | cpu | opencl | 0 | 1 | 1 |  |  | Rayleigh scattering. You are an AI  |
| switch_cpu_to_qnn | switch |  | cpu | qnn-npu | 0 | 1 | 1 |  |  | Rayleigh scattering. Additional information: The  |
| switch_opencl_to_cpu | switch |  | opencl | cpu | 0 | 1 | 1 |  |  | Rayleigh scattering. You are an AI  |
| switch_opencl_to_qnn | switch |  | opencl | qnn-npu | 1 | 0 | 0 |  |  |   |
| switch_qnn_to_cpu | switch |  | qnn-npu | cpu | 0 | 0 | 1 |  |  | Absorption. Question: What physical mechanism  |
| switch_qnn_to_opencl | switch |  | qnn-npu | opencl | 1 | 0 | 0 |  |  |   |
