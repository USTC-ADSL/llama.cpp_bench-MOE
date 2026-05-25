# GGUF 转换与新模型正确性验证

本文说明从 Hugging Face 下载新模型后，如何在当前项目中转换 GGUF、量化、做基础正确性验证，并把模型用于 `llama-completion`、`llama-bench` 和 MoE 相关运行。

注意：这里的“保证能运行”是工程验证流程，不是承诺任意 HF 模型都一定支持。前提是当前 `src/models/`、`convert_hf_to_gguf.py`、`gguf-py/gguf/tensor_mapping.py` 已支持该模型架构、tensor 命名、tokenizer 和 chat template。新架构仍可能需要补 runtime loader 或 converter 映射。

## 1. 准备 Python 环境

建议在仓库根目录创建独立 venv：

```sh
python3 -m venv .venv-gguf
source .venv-gguf/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements/requirements-convert_hf_to_gguf.txt
```

如果要更新 converter 的 tokenizer pre-tokenizer hash：

```sh
python3 -m pip install -r requirements/requirements-convert_hf_to_gguf_update.txt
```

如果要跑随机 tokenizer 测试：

```sh
python3 -m pip install -r requirements/requirements-test-tokenizer-random.txt
```

## 2. 下载 Hugging Face 模型

方式一：使用 `huggingface-cli`：

```sh
export HF_REPO=<org-or-user>/<model-name>
export HF_MODEL_DIR=$PWD/models-hf/<model-name>

huggingface-cli download "$HF_REPO" \
  --local-dir "$HF_MODEL_DIR" \
  --local-dir-use-symlinks False
```

方式二：模型仓库已经是单个 GGUF 文件时，可以用轻量下载脚本：

```sh
mkdir -p models-gguf
scripts/hf.sh \
  --repo <org-or-user>/<gguf-repo> \
  --file <model-file.gguf> \
  --outdir models-gguf
```

`scripts/hf.sh` 只适合下载 HF 上已有的 GGUF 文件；如果是 PyTorch/Safetensors checkpoint，需要走下一步转换。

## 3. 转换为 GGUF

把 HF checkpoint 转为 F16/BF16/F32 GGUF：

```sh
python3 convert_hf_to_gguf.py "$HF_MODEL_DIR" \
  --outfile "$HF_MODEL_DIR/model-f16.gguf" \
  --outtype f16
```

常用 `--outtype`：

```text
f32
  最高精度，文件最大，通常只用于基准或排查。

f16
  常用中间格式，后续再用 llama-quantize 量化。

bf16
  某些 BF16 原始模型可用；是否合适取决于模型和后端支持。

q8_0
  可以直接生成 Q8_0，但通常仍建议先保留一个 f16/bf16 中间件用于对比。
```

如果是 LoRA：

```sh
python3 convert_lora_to_gguf.py <lora-dir> \
  --outfile <lora-output.gguf>
```

如果是旧格式：

```sh
python3 convert_llama_ggml_to_gguf.py <legacy-model.bin> \
  --outfile <legacy-model.gguf>
```

## 4. 检查 GGUF 结构和 metadata

使用 Python GGUF dump：

```sh
python3 gguf-py/gguf/scripts/gguf_dump.py "$HF_MODEL_DIR/model-f16.gguf"
```

使用 C++ GGUF 示例：

```sh
cmake --build build-clean-min -j --target llama-gguf
build-clean-min/bin/llama-gguf "$HF_MODEL_DIR/model-f16.gguf" r n
```

如果模型很大，需要拆分：

```sh
cmake --build build-clean-min -j --target llama-gguf-split
build-clean-min/bin/llama-gguf-split \
  --split-max-size 4G \
  "$HF_MODEL_DIR/model-f16.gguf" \
  "$HF_MODEL_DIR/model-f16-split.gguf"
```

## 5. 量化

先构建量化工具：

```sh
cmake --build build-clean-min -j --target llama-quantize llama-imatrix llama-perplexity
```

基础量化：

```sh
build-clean-min/bin/llama-quantize \
  "$HF_MODEL_DIR/model-f16.gguf" \
  "$HF_MODEL_DIR/model-Q4_K_M.gguf" \
  Q4_K_M
```

MoE 模型常见注意点：

- 大 MoE 模型更建议先保留 `Q8_0` 或 `f16` 作为对照，再生成低比特量化。
- 如果要改 expert 使用数量，可通过 `llama-quantize --override-kv` 显式覆盖 metadata，但必须记录命令，不能把它当成默认行为。
- `--cpu-moe` / `--n-cpu-moe` 是运行时 expert 放置策略，不等同于模型转换成功。

示例：

```sh
build-clean-min/bin/llama-quantize \
  --override-kv qwen3moe.expert_used_count=int:16 \
  "$HF_MODEL_DIR/model-f16.gguf" \
  "$HF_MODEL_DIR/model-qwen3moe-exp16-Q8_0.gguf" \
  Q8_0
```

## 6. Tokenizer 和 chat template 验证

对仓库内 vocab fixtures 跑回归：

```sh
cmake --build build-clean-min -j --target \
  test-tokenizer-0 \
  test-tokenizer-1-spm \
  test-jinja \
  test-chat-template \
  test-chat-auto-parser \
  test-chat

ctest --test-dir build-clean-min --output-on-failure \
  -R 'test-tokenizer|test-jinja|test-chat'
```

对新模型做 tokenize smoke：

```sh
cmake --build build-clean-min -j --target llama-tokenize

build-clean-min/bin/llama-tokenize \
  -m "$HF_MODEL_DIR/model-f16.gguf" \
  -p "Hello, world" \
  --show-count
```

如果模型需要特定 chat template，优先检查 GGUF metadata 是否带模板；否则用仓库模板显式传入：

```sh
build-clean-min/bin/llama-completion \
  -m "$HF_MODEL_DIR/model-f16.gguf" \
  --chat-template-file models/templates/<template>.jinja \
  -p "Hello" \
  -n 16
```

## 7. 端侧运行 smoke

本节所有运行都要先按第 9 节把二进制和模型推送到 Android 设备，再在设备上执行；不提供本机直接跑 smoke 的入口。

CPU 生成：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/model-Q4_K_M.gguf \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  llama-completion \
  --device CPU \
  -m "$MODEL_PATH" \
  -ngl 0 \
  -p "Write one sentence about heterogeneous inference." \
  -n 32 \
  -s 123 \
  --no-warmup
```

`llama-bench -pg` combined workload：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/model-Q4_K_M.gguf \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  llama-bench \
  --device CPU \
  -m "$MODEL_PATH" \
  -ngl 0 \
  -r 1 \
  -pg 128,16
```

MoE smoke：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/model-Q4_K_M.gguf \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  llama-completion \
  --device CPU \
  -m "$MODEL_PATH" \
  -ngl 0 \
  --cpu-moe \
  -p "Explain MoE routing in one sentence." \
  -n 32 \
  -s 123
```

MoE benchmark sweep：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/model-Q4_K_M.gguf \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  llama-bench \
  --device CPU \
  -m "$MODEL_PATH" \
  -ngl 0 \
  -ncmoe 0,4,8,999 \
  -r 1 \
  -pg 128,16
```

## 8. Perplexity / 质量回归

用同一份文本比较 F16/BF16 和量化模型：

评估语料也要先 push 到设备，`EVAL_TEXT` 写成设备侧路径。

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/model-f16.gguf \
EVAL_TEXT=/data/local/tmp/eval/plain-text-corpus.txt \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  llama-perplexity \
  --device CPU \
  -m "$MODEL_PATH" \
  -f "$EVAL_TEXT" \
  -ngl 0

DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/model-Q4_K_M.gguf \
EVAL_TEXT=/data/local/tmp/eval/plain-text-corpus.txt \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  llama-perplexity \
  --device CPU \
  -m "$MODEL_PATH" \
  -f "$EVAL_TEXT" \
  -ngl 0
```

数据质量要求：

- 同一模型家族、同一 tokenizer、同一文本才能比较 PPL。
- PPL 只能作为量化质量回归辅助，不代表人类偏好。
- 失败或异常输出必须记录，不能只保留成功结果。

## 9. 推送到 Android 设备运行

推送二进制：

```sh
DEVICE=<adb-serial> \
BUILD_DIR=build-qnn-opencl/bin \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/push_to_device_simple.sh
```

推送模型：

```sh
adb -s "$DEVICE" shell "mkdir -p /data/local/tmp/models"
adb -s "$DEVICE" push "$HF_MODEL_DIR/model-Q4_K_M.gguf" /data/local/tmp/models/
```

设备侧 smoke：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/model-Q4_K_M.gguf \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
BACKEND_DEVICE=GPUOpenCL \
bash scripts/snapdragon/adb/run-completion.sh \
  -p "Write one sentence about mobile LLM inference." \
  -n 32 \
  -s 123
```

设备侧 `llama-bench -pg`：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/model-Q4_K_M.gguf \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
BACKEND_DEVICE=qnn-npu \
bash scripts/snapdragon/adb/run-bench.sh \
  -v \
  -r 1 \
  -pg 512,32
```

## 10. 最小通过标准

一个新模型至少通过以下检查后，再进入后端切换实验：

```text
1. convert_hf_to_gguf.py 成功生成 GGUF，且 gguf_dump 能读 metadata。
2. llama-tokenize 能用该 GGUF 分词。
3. 端侧 llama-completion CPU smoke 能稳定输出，不崩溃。
4. 端侧 llama-bench -pg CPU smoke 能完成。
5. 如果要量化，量化模型和 f16/bf16 对照模型都能运行。
6. 如果是 MoE，至少跑一次 --cpu-moe 或 -ncmoe sweep，并记录命令。
7. 上设备前，本机路径只用于转换和打包；真正的 smoke 都在设备侧运行，脚本只使用 DEVICE 和 MODEL_PATH。
```
