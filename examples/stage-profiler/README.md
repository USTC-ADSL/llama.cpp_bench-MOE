# Stage Profiler

A tool for profiling the execution time of different stages in the LLaMA inference pipeline.

## Overview

This tool divides each transformer layer into three stages and measures the time spent in each:

### Stage 1: Attention Projection (`Attn_Proj`)
- `attn_norm` (RMSNorm)
- Q/K/V MatMul projections
- **Ends at**: `cb("Vcur", il)`

### Stage 2: Attention Core (`Attn_Core`)
- RoPE (Rotary Position Embedding)
- KV Cache operations (store/load)
- Attention Score computation
- Output Projection (`wo`)
- Residual Add
- **Ends at**: `cb("ffn_inp", il)`

### Stage 3: FFN Block (`FFN_Block`)
- `ffn_norm` (RMSNorm)
- Gate/Up MatMul projections
- SiLU activation
- Down MatMul projection
- Residual Add
- **Ends at**: `cb("l_out", il)`

## Inference Phases

The tool distinguishes between two inference phases:

- **Prefill**: Processing the input prompt (executed once)
- **Decode**: Generating new tokens one by one (executed multiple times, statistics averaged)

## Usage

```bash
# Basic usage
llama-stage-profiler -m model.gguf -p "Hello world" -n 10

# Output to JSON file
llama-stage-profiler -m model.gguf -p "Hello world" -n 10 --json -o timing.json

# Use specific backend (e.g., OpenCL GPU)
llama-stage-profiler -m model.gguf -dev OpenCL0 -ngl 28

# List available devices
llama-stage-profiler --list-devices
```

## Options

| Option | Description |
|--------|-------------|
| `-m, --model PATH` | Model file path (required) |
| `-p, --prompt TEXT` | Test prompt (default: "Hello, how are you today?") |
| `-n, --n-predict N` | Number of tokens to generate in decode phase (default: 10) |
| `--json` | Output in JSON format |
| `-o, --output PATH` | Output file path (default: stdout) |
| `-dev, --device NAME` | Use specific device (e.g., 'OpenCL0', 'CUDA0', 'HTP0') |
| `--list-devices` | List available devices and exit |
| `-ngl, --n-gpu-layers N` | Number of layers to offload to device |
| `-h, --help` | Show help message |

## Output Format

### Table Format (default)

```
========================================
Stage Profiler Results
========================================
Model: llama-7b
Layers: 32
Device: OpenCL0

=== prefill Phase ===

Per-Layer Timing (microseconds):
Layer    | Stage1(Proj)   | Stage2(Attn)   | Stage3(FFN)    | Total
---------|----------------|----------------|----------------|---------------
0        | 1234.56        | 2345.67        | 3456.78        | 7036.01
1        | 1234.56        | 2345.67        | 3456.78        | 7036.01
...

Global Stage Summary:
Stage        | Total(us)    | Mean(us)     | Min(us)      | Max(us)      | StdDev       | Count    | Percent
-------------|--------------|--------------|--------------|--------------|--------------|----------|--------
Attn_Proj    | 39506.88     | 1234.59      | 1200.00      | 1300.00      | 25.00        | 32       | 17.5%
Attn_Core    | 75061.44     | 2345.67      | 2300.00      | 2400.00      | 30.00        | 32       | 33.3%
FFN_Block    | 110616.96    | 3456.78      | 3400.00      | 3500.00      | 35.00        | 32       | 49.1%

Total prefill time: 225185.28 us (225.19 ms)
```

### JSON Format

```json
{
  "model": "llama-7b",
  "n_layers": 32,
  "device": "OpenCL0",
  "decode_iterations": 10,
  "prefill": {
    "total_time_us": 225185.28,
    "total_time_ms": 225.19,
    "stages": {
      "Attn_Proj": {
        "total_us": 39506.88,
        "mean_us": 1234.59,
        "min_us": 1200.00,
        "max_us": 1300.00,
        "stddev_us": 25.00,
        "count": 32
      },
      "Attn_Core": { ... },
      "FFN_Block": { ... }
    },
    "layers": [
      {
        "layer_id": 0,
        "stage1_us": 1234.56,
        "stage2_us": 2345.67,
        "stage3_us": 3456.78,
        "total_us": 7036.01
      },
      ...
    ]
  },
  "decode": { ... }
}
```

## Use Cases

1. **Performance Analysis**: Identify which stage is the bottleneck in your inference pipeline
2. **Backend Comparison**: Compare timing across different backends (CPU, CUDA, OpenCL, etc.)
3. **Optimization Guidance**: Focus optimization efforts on the most time-consuming stages
4. **Hardware Selection**: Evaluate which hardware accelerator provides the best performance for each stage

## Notes

- The timing is measured using `std::chrono::high_resolution_clock`
- For decode phase, statistics are averaged across all generated tokens
- Stage boundaries are detected based on tensor names in the computation graph
- The tool uses the same callback mechanism as `llama-op-profiler`