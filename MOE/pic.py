import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# 您的原始数据
data_csv = """Model,Backend,Phase,Workload,Avg ms,Tok/s
Q4_0,CPU,prefill,pp128_tg0,1347.732,94.986
Q4_0,CPU,decode,pp0_tg32,1505.461,21.842
Q4_0,CPU,combined,pp128_tg16,2036.890,70.768
Q4_0,GPUOpenCL,prefill,pp128_tg0,1178.441,108.654
Q4_0,GPUOpenCL,decode,pp0_tg32,1388.461,23.057
Q4_0,GPUOpenCL,combined,pp128_tg16,1884.386,76.424
Q4_0,FastRPC/HTP0,prefill,pp128_tg0,3168.767,40.394
Q4_0,FastRPC/HTP0,decode,pp0_tg32,43303.917,0.739
Q4_0,FastRPC/HTP0,combined,pp128_tg16,25035.158,5.752
Q8_0,CPU,prefill,pp128_tg0,1803.907,71.962
Q8_0,CPU,decode,pp0_tg32,2209.120,14.976
Q8_0,CPU,combined,pp128_tg16,3347.653,47.194
Q8_0,GPUOpenCL,prefill,pp128_tg0,2466.680,51.892
Q8_0,GPUOpenCL,decode,pp0_tg32,1910.505,16.750
Q8_0,GPUOpenCL,combined,pp128_tg16,3277.074,43.942
Q8_0,FastRPC/HTP0,prefill,pp128_tg0,2829.180,45.249
Q8_0,FastRPC/HTP0,decode,pp0_tg32,41801.236,0.766
Q8_0,FastRPC/HTP0,combined,pp128_tg16,23742.825,6.065"""

# 读取数据
df = pd.read_csv(StringIO(data_csv))

# 组合量化模型与后端作为图例项
df['Config'] = df['Model'] + " - " + df['Backend']

# 设置绘图风格
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# 绘制折线图，横轴为 Phase，纵轴为 Tok/s
# 通过 hue (颜色) 区分后端组合，style (线型/标记) 区分量化级别
sns.lineplot(
    data=df, 
    x='Phase', 
    y='Tok/s', 
    hue='Backend', 
    style='Model', 
    markers=['o', 's'], 
    linewidth=2.5,
    markersize=10
)

# 调整图表标签和标题
plt.title('Performance Trend: Tok/s Across Inference Phases', fontsize=16, pad=15)
plt.xlabel('Inference Phase', fontsize=14)
plt.ylabel('Speed (Tokens / second)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 优化图例
plt.legend(title='Backend & Quantization', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 保存图片到本地
plt.savefig('./MOE/performance_line_chart.png', dpi=300, bbox_inches='tight')
print("图表已保存为 performance_line_chart.png")