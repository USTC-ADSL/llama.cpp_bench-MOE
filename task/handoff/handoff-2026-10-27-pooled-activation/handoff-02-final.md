# 总预算池最终交接

用户要求压缩2K异构总预算，已完成实现和设备验证。实现提交 a625c391e；通过 runner `--shared-arena pooled` 启用，原private路径保留。没有更改模型、chunk、调度、精度或三端并发策略。

最终容量963.09375→443.094116 MiB，减少约520 MiB（53.99%）。两个pooled输入中位2.420/2.403秒；同版本wetland private对照16.391秒，约6.77倍差异。完整结果在results.md、summary.json；保留不利数据：pooled wetland仍有缺页，两个窗口仍有47/2条LMKD Kill，没有宣称全部OOM消失。

完成原layout测试、新360例设备探针、三个完整模型进程共24次请求/18次计时。logits逐位一致、真实counts/owner一致、分区非重叠和容量界限全部验证。启动顺序pooled wetland/foundry、private wetland；不是ABBA性能统计，没有额外跑foundry private重复已确认现象。

构建/运行命令在build-commands.json、runs.json、handoff-01.md，日志在evidence/。build-record.json记录源码/运行库归档hash，runtime.tar.gz逐member校验通过。原始证据只留主机本地并有evidence-manifest.json；cleanup.json记录设备目录清理、模型stat和boot ID核验。用户TODO.md、paper/、results/未改动。

下一步如继续优化，应先定位剩余主机换出/加载峰值，或对当前保守partition上界继续做生命周期精确压缩；不应修改所有cached tensor指针绕过OpenCL offset缓存。新pool是按单请求、每层所有分支join后复用的协议，不可直接接入early-shared decode或多请求并发。当前对early-shared组合会明确报错，图metadata长期增长未做压力验证。

不要重跑本目录run.py覆盖已有tag/frozen文件；后续新实验建立新目录。当前任务没有未完成设备测试。
