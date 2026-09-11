# 工作区整理结果

## 目标与边界

用户要求整理当前工作区。保留已有文件内容和删除状态，将 E1 相关改动按用途保存为本地 Git 检查点；不推送、不运行设备实验、不清理模型、有效构建或实验日志。

整理起点：`moe-mobile-opt` 分支，`f964edbd00cd4c94e801eb7e3253b64a4e2907f8`。最初有 14 个修改文件、1,087 个已删除文件、8 个未跟踪文件，暂存区为空。

## 已完成

- `73f78e3d1`：单独记录整理前已经删除的 1,081 个 `docs/Motivation_Exp/E1-B/evidence/` 文件。本次未额外删除实验文件。
- `a3aee4279`：保存已有 6 个报告/CSV 迁移、E1 新报告、6 个 Python 脚本路径/标签同步，以及 Bug 简报和 6 份历史交接的名称同步，共 20 个变更条目。
- 本目录保存核验结果和任务清单，随最终整理记录提交。
- `docs/Motivation_Exp/E1-B/report.md` 和 `results.csv` 仍在原位，内容与起点相同。没有把它们当作重复文件删除。
- 根据历史交接的明确排除规则，`AGENTS.md` 和 `docs/Motivation_Exp/Exp1/Exp_require.md` 继续保持未提交，内容未改动。

## 核验结果

旧 E1-B 在起点共有 1,089 个受版本控制的文件。新 E1 只有 7 个根目录文件；逐文件 SHA-256 比较，2 个 CSV 完全一致，4 份协议/历史报告只作少量名称替换，主报告为不同内容。所有 1,081 个 evidence 文件均不在新目录，`results.csv` 也没有迁入新目录。因此此次整理不能描述为完整无损迁移。

报告中的 `evidence/...` 链接目前不可访问，依赖原始证据的分析器也无法完成验证。历史报告的 PASS 是过去实验结论，不是本次重新验证的结果。历史交接内容按已有修改原样提交，不追加新的实验结论。

主线程执行：

```sh
git diff --check
env PYTHONPYCACHEPREFIX=/tmp/llama-workspace-check-20260911 python3 -m py_compile tests/analyze-e1b-heavy.py tests/analyze-static-app.py tests/run-e1b-completion.py tests/run-e1b-heavy.py tests/run-static-coexist.py tests/static-app-web.py
rg -n 'docs/Motivation_Exp/E1-B' tests
git diff --cached --check
git status --short
```

语法检查及每次暂存差异检查通过；测试脚本中旧目录路径搜索无匹配（退出码 1）。证据删除提交前验证暂存内容恰为该目录下 1,081 个删除，无其他文件。语法检查缓存写入上述专用临时目录，检查后清理。未执行构建、部署或 ADB 测试，本任务仅涉及已有改动的静态整理。

## 证据入口与限制

完整的受版本控制的旧证据仍位于整理起点，可只读查看或导出，无需回滚当前分支。例如：

```sh
git show f964edbd00cd4c94e801eb7e3253b64a4e2907f8:docs/Motivation_Exp/E1-B/evidence/completion/placement-check.txt
git archive --format=tar --output=/tmp/e1-evidence-f964edbd0.tar f964edbd00cd4c94e801eb7e3253b64a4e2907f8 docs/Motivation_Exp/E1-B/evidence
```

上面的归档命令仅供后续需要时执行，本次未额外生成证据副本。Git 无法恢复从未提交的原始素材。若后续要求重跑分析，先将所需证据提取到隔离目录，核对路径和哈希，再决定如何接入 E1；不要直接启动设备实验或覆盖已有报告。

## 保留资产与最终状态

- 模型目录约 7.7 GiB、Android/OpenCL 构建目录约 655 MiB、编译缓存约 161 MiB，全部保留；有效构建支持后续增量编译。
- 盘点时未发现本仓库正在执行的构建或测试任务；其他目录的 ADB 任务未触碰。
- `.gitignore` 的 `/build*` 也匹配未跟踪的 `build-npu-opencl.sh`。本次未改忽略策略，该脚本仍在本地，后续若需版本化应单独处理。
- 上一轮 `pelican-bike.html` 在本次开始时已不在仓库及直接上一级，未擅自恢复或重建。
- 整理完成后预期仅剩 ` M AGENTS.md` 和 `?? docs/Motivation_Exp/Exp1/Exp_require.md`；这两项是明确保留的用户工作。
- 无待执行的整理步骤。下一步由用户决定是否将上述两项另行纳入版本，或是否恢复原始证据以重跑分析。
