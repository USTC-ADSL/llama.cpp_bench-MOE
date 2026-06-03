| case | category | backend | prefill | decode | rc | semantic_ok | route_ok | prompt_ms | eval_ms | response_excerpt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_cpu | single_backend | cpu |  |  | 0 | 1 | 1 |  |  | During the day, the blue color of the sky is due to the scattering of  |
| single_opencl | single_backend | opencl |  |  | 0 | 1 | 1 |  |  | During the day, the blue color of the sky is due to the scattering of  |
| single_qnn | single_backend | qnn |  |  | 0 | 1 | 1 |  |  | The sky appears blue during the day due to the scattering of sunlight by Earth's  |
| switch_cpu_to_opencl | switch |  | cpu | opencl | 0 | 1 | 1 |  |  | During the day, the blue color of the sky is due to the scattering of  |
| switch_cpu_to_qnn | switch |  | cpu | qnn-npu | 0 | 1 | 1 |  |  | During the day, the blue color of the sky is due to the scattering of  |
| switch_opencl_to_cpu | switch |  | opencl | cpu | 0 | 1 | 1 |  |  | During the day, the blue color of the sky is due to the scattering of  |
| switch_opencl_to_qnn | switch |  | opencl | qnn-npu | 0 | 1 | 1 |  |  | During the day, the blue color of the sky is due to the scattering of  |
| switch_qnn_to_cpu | switch |  | qnn-npu | cpu | 0 | 1 | 1 |  |  | The blue color of the sky appears due to the scattering of sunlight by atmospheric gases  |
| switch_qnn_to_opencl | switch |  | qnn-npu | opencl | 0 | 1 | 0 |  |  | The sky appears blue during the day due to the scattering of sunlight by Earth's  |
