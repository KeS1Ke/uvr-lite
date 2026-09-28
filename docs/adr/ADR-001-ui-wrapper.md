# ADR-001：uvr-lite UI 外包装

- 状态：**已接受**（2026-08-03，grill 24 问对齐）；2026-08-03 修订：第 8/14 条打包形态改为 Inno Setup 全量安装包（PyInstaller 向导弃用，见修订注记）；2026-09 修订：第 8/14 条改为**单包半在线**（CUDA 引擎不再内置，详见表内注记），并补记「日志」默认项的实际落地方式
- 背景：CLI 面向非专业用户（自用 + 亲友），需要图形界面与一键安装；安装位置可选、无需预装 Python、桌面 ♪ 图标点击进入。

## 决策

| # | 决策 | 结论 |
|---|---|---|
| 1 | UI 形态 | PySide6 桌面 GUI |
| 2 | 调用方式 | 进程内调 `engine.separate_file()`，加可选进度回调参数 |
| 3 | MVP 范围 | 核心集：**输入两种添加方式（选择文件/拖拽 + 选择输入文件夹）** + 模型下拉 + 参数表单（device/format/pcm/bigshifts/tta/batch-size）+ 输出文件夹选择 + 开始/取消 + 实时进度条 + 打开输出目录 |
| 4 | 启动入口 | `uvr-lite ui` 子命令 + 桌面快捷方式（`pythonw.exe -m uvr_lite.ui`，无黑窗） |
| 5 | 进度/取消 | `progress_callback(phase, done, total) -> bool`，返回 False 抛 `CancelledError`，UI 清理半成品；CLI 不传回调行为不变 |
| 6 | 分发 | 快捷方式指 venv，UI 本体不打包 exe |
| 7 | 图标 | ♪ 音乐符号（PIL 生成 ico+png，快捷方式/窗口/任务栏共用） |
| 8 | 安装器形态 | Inno Setup 7 安装包（内置绿色 Python + 全部依赖 + CPU torch + fp16 模型），安装 = 纯文件复制，用户免联网下载。**(2026-09 修订：CUDA torch 不再内置)**——它是默认不勾选的附加任务「下载 CUDA 推理引擎（约 3.3 GB，需联网）」，勾选时安装中联网下载（`ExternalSize` 3.27GB，进度页 + SHA256 校验）；未勾选可在应用内或 CLI `uvr-lite install-cuda` 事后补装（原「双 torch 全内置」表述已废止） |
| 9 | Python 前置 | 自动检测系统 Python 3.10+ 复用；没有则下载 python-build-standalone（绿色解压到安装目录） |
| 10 | 目录结构 | 单安装目录（默认 用户目录\uvr-lite）：代码+绿色 Python+venv+models |
| 11 | 向导技术栈 | PySide6 同栈 |
| 12 | 升级 | 覆盖升级：更新代码/依赖，保留 venv 与模型 |
| 13 | 卸载 | 带卸载入口（--uninstall）：删目录+删快捷方式+确认弹窗 |
| 14 | 打包形态 | Inno Setup 7 安装包（`uvr-lite-setup-{cpu|full}_v{version}.exe`，lzma2/ultra64；cpu 省 CUDA torch 3.3GB）。**(2026-09 修订：单包化，上述 cpu/full 双变体已取消)**——因为 CUDA torch 已移出包体，只出**单个** `uvr-lite-setup_v{#MyAppVersion}.exe`；压缩改为 `Compression=lzma2/max`（8MB 字典）而非 ultra64（64MB）：解压快 2-3 倍、体积仅 +3%（install.iss:41/45-47） |
| 15 | 发布 | GitHub Releases 挂 setup exe，README 双语下载徽章 |
| 16 | 模型缺失 | UI 内一键下载：提示条 + 下载按钮（复用 ensure_model，带进度与重试） |
| 17 | 平台范围 | 仅 Windows 做 GUI 向导；Linux/macOS 维持 install.sh |
| 18 | 下载源 | 主源+镜像回退：绿色 Python 备 ghproxy/国内镜像，torch 备清华 PyPI |
| 19 | 快捷方式 | 桌面 + 开始菜单（卸载同步删除） |
| 20 | 界面语言 | 中文 |
| 21 | 失败处理 | 队列单文件失败：跳过继续 + 结束汇总（成功 N/失败 M+清单） |
| 22 | 进度粒度 | 阶段 + 百分比 + 当前文件 i/N + ETA（线性估算） |
| 23 | 测试策略 | tdd 引擎 + UI 手工冒烟 |
| 24 | 目标用户 | 自用 + 亲友：文案口语化、流程简单、基础体验到位，不追求公开分发级打磨 |

## 默认项（同批确认）

- 参数/输出目录 QSettings 记忆（模型/参数/输出目录）
- 安装完成页"立即启动"按钮（不强制自动弹）
- 安装目录默认 `用户目录\uvr-lite`
- 文件列表提供"移除所选/清空"
- 逐票交付：每票完成演示验收后再下一票
- 实现默认：绿色 Python/torch 固定版本+SHA256+断点续传；日志写 `安装目录/logs/uvr-lite.log`（RotatingFileHandler，单文件 2 MB、保留 3 个回滚，UTF-8，只记录 WARNING 及以上；UI/CLI 的错误弹窗带上日志路径，安装器创建空的 logs 目录、卸载不影响——2026-09 落地，仅本地排错用，非遥测/非结构化日志方案）；错误友好中文弹窗+日志路径；窗口标题/About 用 `uvr-lite`；安装器中途关闭可续装

## 理由摘要

- **PySide6 而非 Web/Tkinter**：桌面体验最贴近非专业用户习惯；与安装向导同栈、组件复用；Tkinter 界面朴素且两套代码。
- **进程内调 engine 而非子进程**：进度可控（回调）、取消可控、无 stdout 解析脆弱性；engine 改动约 15 行且 CLI 行为不变。
- **回调返回 False 即取消**：长任务（CPU ~6× 实时）中途可停，是长任务 UI 基本体验；为后续批量队列复用。
- **GUI 安装向导 + 绿色 Python**：非专业用户零前置（不装 Python、不 clone、不碰黑窗），单安装目录概念最简，卸载 = 删目录。
- **全量安装包而非运行时下载**：非专业用户零前置、零网络依赖；torch CPU/CUDA 双内置，应用内切换（torch.ini，重启生效）。原 PyInstaller onefile 向导方案因收集 bug 与 >2GB 体积不可行弃用；NSIS 有 ~2GB 硬上限（full 变体无法打包）→ 最终用 Inno Setup 7。（2026-09 修订：CUDA torch 实测**不再内置**——内置它会让所有用户为一个多数人用不上的引擎多下 3.3 GB；改为默认不勾选的附加任务按需下载，不勾选的用户仍是零联网，CUDA 则随时可事后补装；随之 cpu/full 双变体取消，只出单个 setup exe）
- **覆盖升级**：保留 venv 与 320MB fp16 模型，升级免重下。（原写 640MB，2026-09 随 fp16 分发改正）
- **主源+镜像回退**：与 download.py 现有多源设计一致，国内网络安装成功率优先。

## 后果

- 正面：CLI 行为零变化（separate/download/models 回归保证）；新增路径（ui/installer）互不干扰主干。
- 负面：PySide6 依赖 ~150MB（仅 `[ui]` extras 安装）；安装器打包需维护构建脚本；中文/空格安装路径需显式测试。
- 后续可选项（**待办，均未实现**，不在本次 MVP 范围）：模型管理页、音频试听、频谱图预览、批量并行、UI 本体 exe 打包、代码签名、自动更新、遥测/判决式可观测性、Linux/macOS 向导、pytest-qt。日志方面本次只落地上述 `安装目录/logs/uvr-lite.log` 本地文件日志。
