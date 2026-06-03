| case | category | backend | prefill | decode | rc | semantic_ok | route_ok | prompt_ms | eval_ms | response_excerpt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_cpu | single_backend | cpu |  |  | 255 | 0 | 1 |  |  | During the day, the blue color of the sky  |
| single_opencl | single_backend | opencl |  |  | 1 | 0 | 1 |  |  |   |
| single_qnn | single_backend | qnn |  |  | 0 | 1 | 1 |  |  | The sky appears blue during the day due to the scattering of sunlight by Earth's atmosphere. This scattering makes the sky predominantly  |
| switch_cpu_to_opencl | switch |  | cpu | opencl | 1 | 0 | 0 |  |  |   |
| switch_cpu_to_qnn | switch |  | cpu | qnn-npu | 0 | 1 | 0 |  |  | During the day, the blue color of the sky is due to the scattering of sunlight by Earth's atmosphere. The blue  |
| switch_opencl_to_cpu | switch |  | opencl | cpu | 1 | 0 | 0 |  |  |   |
| switch_opencl_to_qnn | switch |  | opencl | qnn-npu | 1 | 0 | 0 |  |  |   |
