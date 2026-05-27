## backend-op-bench

用于在不同后端（HTP0 / GPUOpenCL / CPU 等）上快速测量单个算子的执行时间。支持多种算子，每个算子独立配置维度和数据类型。

### 特性

- ✅ **多算子支持**：MUL_MAT、RMS_NORM、SILU、FFN
- ✅ **灵活的维度配置**：每个算子支持不同的维度参数
- ✅ **多后端对比**：同时测试 HTP0、GPUOpenCL、CPU 等
- ✅ **精确的时间测量**：分离冷启动和稳定状态，稳定态统计使用真实均值
- ✅ **结果验证**：自动检测 NaN/Inf
- ✅ **模块化架构**：易于添加新算子

### 构建

```bash
cmake -B build -DGGML_OPENCL=ON -DGGML_HEXAGON=ON .
cmake --build build --target backend-op-bench
```

### 运行示例

#### 测试 MUL_MAT（矩阵乘法）
```bash
# 默认配置：2048×2048 权重 × 2048×1 输入
./backend-op-bench --op mul_mat --backend HTP0 --backend GPUOpenCL --runs 20

# 自定义维度
./backend-op-bench --op mul_mat --m 4096 --k 4096 --n 4 --runs 10
```

#### 测试 RMS_NORM（RMS 归一化）
```bash
# 默认配置：2048×1 输入
./backend-op-bench --op rms_norm --backend HTP0 --runs 20

# 自定义维度
./backend-op-bench --op rms_norm --m 4096 --n 4 --runs 50
```

#### 测试 SILU（纯 SiLU 激活函数）
```bash
# 默认配置：2048×1 输入
./backend-op-bench --op swiglu --backend HTP0 --backend GPUOpenCL --runs 20

# 自定义维度
./backend-op-bench --op swiglu --m 4096 --n 4 --runs 50

# Llama-3.2-1B 中间层维度
./backend-op-bench --op swiglu --m 6144 --n 1 --runs 30
```

**SILU 计算流程**：
```
输入 x [d, n]

output = x * sigmoid(x)  [d, n]  (SiLU 激活函数)
```

**说明**：SiLU (Sigmoid Linear Unit) 也称为 Swish 激活函数，公式为 `f(x) = x * σ(x)`，其中 `σ(x)` 是 sigmoid 函数。

#### 测试 FFN（完整 FFN 层）
```bash
# 默认配置：双 2048×2048 权重 × 2048×1 输入
./backend-op-bench --op ffn --backend HTP0 --backend GPUOpenCL --runs 20

# 自定义维度
./backend-op-bench --op ffn --m 2048 --k 2048 --n 1 --runs 30
```

**FFN 计算流程**：
```
输入 x [k, n]

gate = W1 × x            [m, n]  (上投影-门控)
value = W2 × x           [m, n]  (上投影-值)
swiglu = SiLU(gate) ⊗ value [m, n]  (SWiGLU 激活)
output = W_down × swiglu [down_m, n]  (下投影)
```

### 输出示例

```
mul_mat: 2048x2048 * 2048x1, dtype=Q8_0xF32
backend       cold us      avg us       min us       max us       note
HTP0          1234.56      1100.23      1095.12      1105.34
GPUOpenCL     980.44       950.12       945.67       955.89
CPU           2345.67      2200.34      2195.12      2205.56
```

**输出字段说明**：
- `cold us`：首次运行时间（冷启动）
- `avg us`：稳定状态运行时间的算术平均值
- `min us`：稳定状态最小时间
- `max us`：稳定状态最大时间
- `note`：错误信息（如果有）

### 命令行参数

```
Usage: backend-op-bench [--op NAME] [--backend NAME] [--m M] [--k K] [--n N] [--runs R]

Operators:
  mul_mat   - Matrix multiplication (Q8_0 weight × F32 input)
  rms_norm  - RMS normalization (F32 input)
  swiglu    - SiLU activation: y = x * sigmoid(x)
  ffn       - FFN fused: MUL_MAT(Gate/Up) → SWIGLU → MUL_MAT(Down)

Options:
  --op NAME       算子名称（默认：mul_mat）
  --backend NAME  后端名称，可指定多个（默认：HTP0 GPUOpenCL CPU）
  --m M           输出维度/权重行数（默认：2048）
  --k K           输入维度/权重列数（默认：2048）
  --n N           批大小/输入列数（默认：1）
  --runs R        测试运行次数（默认：50）
  --help, -h      显示帮助信息
```

### 架构

```
op-interface.h          ← 算子虚基类接口
├── op-mul-mat.h        ← MUL_MAT 算子实现
├── op-rms-norm.h       ← RMS_NORM 算子实现
├── op-swiglu.h         ← SILU 算子实现（纯激活函数）
├── op-ffn.h            ← FFN 算子实现（完整 FFN 层）
├── bench-common.h      ← 通用基础设施（RAII、验证）
└── backend-op-bench.cpp ← 主程序（工厂、参数解析、测试循环）
```

### 添加新算子

#### 步骤 1：创建算子头文件

创建 `op-your-op.h`，继承 `OpInterface`：

```cpp
#pragma once
#include "ggml.h"
#include "op-interface.h"

struct OpYourOp : public OpInterface {
    int64_t d, n;
    ggml_tensor* x = nullptr;
    ggml_tensor* y = nullptr;

    OpYourOp(int64_t d_, int64_t n_) : d(d_), n(n_) {}

    const char* name() const override { return "your_op"; }

    void create_tensors(ggml_context* ctx) override {
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);
    }

    ggml_tensor* create_graph(ggml_context* ctx) override {
        y = ggml_your_op(ctx, x);  // 调用 GGML 算子
        ggml_set_name(y, "your_op_out");
        ggml_set_output(y);
        return y;
    }

    void fill_inputs() override {
        // 填充输入张量
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        const int64_t elems = ggml_nelements(x);
        std::vector<float> buf(elems);
        for (auto & v : buf) v = dist(rng);
        ggml_backend_tensor_set(x, buf.data(), 0, buf.size() * sizeof(float));
    }

    ggml_tensor* get_output() const override { return y; }

    std::string description() const override {
        return std::string("your_op: ") + std::to_string(d) + "x" + std::to_string(n);
    }
};
```

#### 步骤 2：在主程序中注册

编辑 `backend-op-bench.cpp`：

```cpp
#include "op-your-op.h"

static std::unique_ptr<OpInterface> create_operator(const options& opt) {
    // ... 现有代码 ...
    } else if (opt.op == "your_op") {
        return std::make_unique<OpYourOp>(opt.m, opt.n);
    }
    // ...
}
```

#### 步骤 3：更新帮助信息

```cpp
static void print_usage(const char * argv0) {
    // ...
    std::printf("  your_op   - Your operator description\n");
    // ...
}
```

#### 完成！

现在可以使用：
```bash
./backend-op-bench --op your_op --backend HTP0 --runs 20
```

### 性能优化建议

1. **冷启动 vs 稳定状态**：
   - `first_us` 包含初始化开销（缓存预热、内存分配等）
   - `avg_us` 反映稳定状态性能
   - 对比时应关注稳定状态指标

2. **批大小影响**：
   - 增加 `--n` 可能改善吞吐量
   - 但延迟可能增加

3. **后端选择**：
   - HTP0：Hexagon NPU，适合移动推理
   - OpenCL0：GPU 加速
   - CPU：基准参考

### 扩展建议

- 支持多量化格式（Q4_0、Q5_0、Q6_K 等）
- 配置文件驱动（JSON/YAML）
- 结果持久化（CSV 输出）
- 硬件计数器集成（Hexagon HAP_perf）
- 多算子组合测试

### 相关文档

- [GGML 后端文档](../../docs/)
- [Hexagon NPU 使用指南](../../docs/HEXAGON_NPU_USAGE.md)
- [性能分析指南](../../docs/LEARNING_GUIDE.md)
