# 精简范围与恢复边界

本文说明当前工作区保留什么、删除什么，以及为什么 GGUF 转换、模型正确性验证和 MoE 运行辅助工具需要恢复。

当前目标仍然不是维护完整上游 `llama.cpp` 产品面，而是保留两条必要链路：

- 后端实验链路：CPU、GPUOpenCL、QNN/NPU、Snapdragon 后端驱动，Prefill/Decode phase switch，KV handoff/migration/alias/reserve/graph rebuild，可观测切换 overhead。
- 模型导入链路：从 Hugging Face 下载新模型后，可以转换成 GGUF、量化、检查 tokenizer/chat template、运行 smoke/perplexity，并验证 MoE 模型的基本运行路径。

当前仍不维护功耗实验。不要新增 power sampling、电流/电压采样、能耗表或基于功耗数据的论文结论，除非任务明确转向功耗。

## 保留的核心边界

核心运行时和后端：

```text
src/
include/
common/
ggml/
vendor/cpp-httplib/
vendor/nlohmann/
vendor/sheredom/
```

后端实验工具：

```text
tools/llama-bench/
tools/completion/
tools/hetero-switch-bench/      # 仅 GGML_OPENCL=ON 时构建
tools/qnn-kv-export/
examples/backend-op-bench/
examples/stage-profiler/
scripts/snapdragon/adb/
```

模型导入、GGUF 和正确性辅助工具：

```text
convert_hf_to_gguf.py
convert_hf_to_gguf_update.py
convert_llama_ggml_to_gguf.py
convert_lora_to_gguf.py
examples/convert_legacy_llama.py
gguf-py/
models/
models/templates/
examples/model-conversion/
tools/quantize/
tools/perplexity/
tools/imatrix/
tools/gguf-split/
tools/batched-bench/
tools/tokenize/
examples/gguf/
examples/gguf-hash/
scripts/hf.sh
scripts/jinja/
requirements/requirements-convert_*.txt
requirements/requirements-gguf_editor_gui.txt
requirements/requirements-test-tokenizer-random.txt
```

保留/恢复的主要 CMake target：

```text
llama-bench
llama-completion
llama-batched-bench
llama-gguf-split
llama-imatrix
llama-perplexity
llama-quantize
llama-tokenize
hetero-switch-bench       # 仅 GGML_OPENCL=ON 时构建
llama-qnn-kv-export
backend-op-bench
llama-stage-profiler
test-backend-ops
test-tokenizer-0
test-tokenizer-1-bpe
test-tokenizer-1-spm
test-jinja
test-chat-template
test-chat-auto-parser
test-chat-peg-parser
test-chat
test-arg-parser
test-gguf
test-quantize-fns
QNN/OpenCL phase switch tests
```

## 为什么恢复 GGUF 转换工具链

如果要从 Hugging Face 引入新模型，只保留 runtime loader 不够。实际链路至少需要：

- `convert_hf_to_gguf.py`：把 HF checkpoint/tokenizer/config 转为 GGUF。
- `gguf-py/`：转换脚本依赖的 GGUF reader/writer、tensor mapping、metadata、vocab 和 quant helper。
- `requirements/requirements-convert_hf_to_gguf.txt`：固定转换所需 Python 依赖入口。
- `models/templates/`：保留常用 chat template；新模型或 MoE 模型的 prompt/chat 格式经常依赖它。
- `models/ggml-vocab-*.gguf` 与 `.inp/.out`：tokenizer 回归测试需要这些 vocab fixtures。
- `tools/llama-tokenize` 和 tokenizer tests：验证新 GGUF 的 tokenizer 行为不是只看能不能加载。

这些文件和“后端切换”不是同一层，但它们是运行新模型前的必要入口。没有它们，就只能运行已经准备好的 GGUF，无法可靠引入新 HF 模型。

## 为什么恢复正确性辅助工具

新模型成功运行不是只看程序没崩溃，还要检查几个层面：

- GGUF 结构和 metadata 能被 runtime 正确读取：`test-gguf`、`examples/gguf`、`gguf-py/gguf/scripts/gguf_dump.py`。
- tokenizer 行为正确：`test-tokenizer-0`、`test-tokenizer-1-spm`、`tools/llama-tokenize`。
- chat template / jinja 路径正确：`models/templates/`、`test-jinja`、`test-chat-template`、`test-chat-auto-parser`、`test-chat`。
- 量化后没有明显质量退化：`llama-quantize`、`llama-perplexity`、`llama-imatrix`。
- 大模型或 MoE 模型文件较大时可切分：`llama-gguf-split`。
- batch/route 运行形态可验证：`llama-batched-bench`、`llama-bench -pg`。

因此这些工具应保留；它们不是功耗实验，也不是上游产品噪音。

## MoE 相关保留点

MoE runtime 支持仍在 `src/models/`、`src/llama-model.cpp`、`src/llama-graph.*`、`common/arg.cpp` 和 backend op tests 中。当前额外保留的 MoE 运行辅助包括：

- `llama-bench --n-cpu-moe` / `-ncmoe`，用于 MoE expert 放置策略 benchmark。
- `llama-completion --cpu-moe` / `--n-cpu-moe`，用于实际生成 smoke。
- `test-backend-ops` 中的 `MUL_MAT_ID`、`TOPK_MOE` 等 MoE 相关 op 覆盖。
- `models/templates/` 中的 MoE/large instruct 模板，例如 Kimi、MiniMax、GigaChat、OpenAI MoE、Qwen 系列等。
- `models/ggml-vocab-nomic-bert-moe.gguf` 作为 MoE vocab fixture。

这部分应保留，因为 MoE 模型是否能加载、分词、套用模板、执行 expert routing，是“能跑新模型”的组成部分。

## 仍然删除的内容

下面这些仍然不属于当前目标：

```text
.github/
.devops/
ci/
flake.nix
flake.lock
Makefile
build-xcframework.sh
tools/server/
tools/rpc/
tools/tts/
tools/mtmd/
examples/llama.android/
examples/llama.swiftui/
examples/training/
examples/retrieval/
examples/speculative*
docs/multimodal/
docs/ops/
media/
vendor/miniaudio/
vendor/stb/
```

删除理由：

- CI、DevOps、Docker/Nix/release 不是当前本地或 Snapdragon 实验的必要依赖。
- server/webui/RPC/TTS/MTMD/移动 App 示例会拉入额外依赖和维护面。
- training/retrieval/speculative/multimodal 不参与当前 Prefill/Decode 后端切换与新文本模型导入验证。
- `vendor/miniaudio`、`vendor/stb` 主要服务已删除的音频/多模态路径。

## 当前不能默认做什么

当前工作区仍不默认支持：

- `llama-server` / webui / OpenAI-compatible API。
- 多模态、TTS、training、retrieval、Swift/Android app examples。
- 上游完整 CI/release/docker/nix 打包。
- 功耗采样、能耗表、battery current/voltage 或 power-aware planner。

如果后续要恢复其中某块，应作为独立任务说明它和当前后端切换或模型导入验证的关系。

## 查看删除清单

```sh
git diff --cached --name-status --diff-filter=D
git diff --name-status --diff-filter=D
git status --short --untracked-files=all
git diff --stat
git diff --cached --stat
```
