# E1 较重任务补验进行中

用户同意增加两个App，验证原轻量场景可能内存需求太低。仓库/mnt/sda1/pzw/moe/llama.cpp_bench-MOE；唯一设备fd8657d6在线。原E1已完成，起点1e79e8bd0；用户AGENTS.md/Exp1文件不动。新evidence/heavy、protocol-heavy.md，原report/summary/completion不覆盖。

主线程完整读取AGENTS.md、E1报告/最终handoff以及相关App/static/E0 runner；新代码仅tests/run-e1b-heavy.py与tests/E1BTaskFront.java。沿用当前固定OpenCL库70bac8...6620f及模型，无后端重建；唯一新编译是javac+d8的task-front.dex，hash1c26a5...dc35。远端/data/local/tmp/llama-e1b-heavy已部署，/sdcard/Download/E1B-heavy只有生成素材。helper用于从最近任务恢复指定taskId，不发新Intent、不改优先级。

实际使用多看com.duokan.reader8.4.19 UID10285、编辑器com.miui.mediaeditor2.4.0.2.1 UID10305，后者进程名带:photo_editor。WPS首次联网/隐私协议选择退出，未同意；编辑器“不同意”仍能本地标记保存。多看只确认复制本测试外部文档，不加书架、不登录。app-environment.txt记录元数据。

素材由assets阶段生成：160页1536x2048独立JPEG栅格PDF121621574bytes；48MP8000x6000 JPEG14838990bytes，manifest保存SHA；raw资产被heavy/.gitignore排除，脚本确定性生成。仅两文件MediaStore登记，动态查询精确_data；最初image59/pdf60。初版.nomedia使图片media_type变0，已删除仅本任务空标记并改正部署。原始记录/素材保留。

任务：PDF访问进度x350/x700/x540固定到81/160，回前台验证同页并继续进度到x700后的页；图片标记固定红色斜线（两次同路径），保护未保存标记，回前台截图检测红色后保存，导出日志识别精确新文件，拉回验证8000x6000与红色。编辑器保存到DCIM/Creative/IMG_时间.jpg，不覆盖输入，必须清理仅日志所指本实验输出。

失败准备保留：直接file URI未登记导致编辑器拒绝；普通am start无URI返回多看会触发主界面隐私弹窗（未接受），不是内存失败，改TaskFront后通过；diagnostics/image-taskfront因.nomedia媒体分类失败；preflight-image-noLLM-1的状态/导出正常但gfxinfo用错主进程名导致无帧；改:photo_editor后preflight-image-noLLM-2完整通过。原轻量App内存比较diagnostic-light准备失败，未形成可用PSS基线，不将其当成功或共存结果。

已测内存（独立noLLM诊断，不放入正式性能轮）：PDF ready329006KiB/background262836KiB；image ready176519/background112684KiB；48MP导出15个样本峰值1096104KiB≈1.045GiB。ordinary preview并不重，导出有明显短暂峰值。diagnostics/image-export-check.json验证真正输出/contentURI63，路径/storage/emulated/0/DCIM/Creative/IMG_20260909_010159.jpg，28.79MB、hash49adf5...fbd2、8000x6000、red_pixels5788。早先diagnostics/E1B-exported.jpg只是未修改源文件，失败检查不代表真正输出损坏，actual文件才是输出。

完整noLLM预演：preflight-image-noLLM-2 state/continue true、first_frame34.031ms；preflight-pdf-noLLM-1 state/continue true、first_frame46.509ms。框架无新Intent返回task已验证。first_frame从已有gfxinfo framestats提取IntendedVsync>=TaskFront.START_US且Flags0的首FrameCompleted，是App帧完成非屏幕呈现。dispatch只是Binder耗时不作为体验指标。图片用截图判断已保存前标记，避免UI dump耗时使导出晚于推理；PDF仍UI页码核对，recovery_upper含工具开销不能等同实际恢复耗时。

**当前运行**：`python3 tests/run-e1b-heavy.py --stage pairs --task image --execute`，PTY **45075**，正式image三对AB/BA/AB。不要启动其它设备/UI工作；每条件准备+46s对照等待或B1模型加载+请求，约90s。六条件完成后主线程检查结果/trace/保存时刻重叠，再按预先固定全局顺序运行`--stage pairs --task pdf --execute`。B1仍-ngl99，原512/128 prompt已复制heavy/prompt.json；不重新初筛/不加配额/匿名压力/清缓存。

尚需：实现/执行独立分析（当前新runner只有生成/部署/诊断/预演/配对，没有analyzer）；21旧初筛不重跑，本次12正式条件。检查LLM3+3成功、token匹配、first_frame正且合理、状态和继续功能、图像save_action_us在request_received..request_end内，日志导出start/end可量化与推理重叠；PDFtaskrestore在请求中。stage正常result都有task/condition/state/continue/first_frame等，raw app-result/graphics/export-check与E0 traces全部保留。若真实状态fail或degraded，保留结果不筛掉。

然后写独立短report-heavy.md，并在原E1/report.md追加链接及限定结论；不要覆盖原轻量结果。收尾逐一验证远端测试产物/模型hash，pull应用输出后清理精确MediaStore IDs/输出路径，删除本任务两个目录；别清App数据或DCIM目录，注意多看可能复制了本测试PDF需定位精确复制路径，不触碰私人文件。raw导出JPG被ignore，可用hash/尺寸/标记检查及截图作为版本证据，不能谎称raw图全在Git。

每分钟中文进度；子代理只读单轮，已完成entrypoints探子，不复用。主线程最终验证/修改。按阶段Git封板，尚未提交本轮新文件。新的任务清单前两项已完成。
