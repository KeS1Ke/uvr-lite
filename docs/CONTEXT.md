# uvr-lite 领域术语表（CONTEXT）

> 本项目术语表，随实现演进维护。UI 外包装相关术语见 ADR-001。

## 核心术语

| 术语 | 定义 |
|---|---|
| **分离（separate）** | 将一段混合音频拆成两条音轨：人声（vocals）与伴奏（instrumental）。伴奏 = 原曲 − 人声（数学无损）。 |
| **模型（model）** | 训练好的分离网络权重（`.ckpt`）。`bs_roformer_ep317` 为主力，`mel_band_karaoke` 备选。SHA256 校验，不入 git。 |
| **引擎（engine）** | `uvr_lite/engine.py` 的分离执行层，提供 `separate_file()` 纯函数接口（输入/输出/参数）。 |
| **进度回调（progress_callback）** | 引擎向调用方上报进度的钩子：`callback(phase, done, total) -> bool`。返回 `False` 表示请求取消。 |
| **取消（cancel）** | 用户中止当前任务。引擎收到回调返回 `False` 后抛 `CancelledError`；UI 清理半成品输出。 |
| **半成品清理** | 取消后删除未写完的输出文件（`*-vocals.*` / `*-instrumental.*`），不留残缺文件。 |
| **任务队列（queue）** | UI 中待处理音频的列表。**两种添加方式**：选择文件（多选/拖拽）或选择输入文件夹（扫描其中常见音频格式 mp3/flac/wav/ogg/m4a，默认不递归、去重追加）。按添加顺序逐个处理；单文件失败跳过继续，结束汇总"成功 N / 失败 M"。 |
| **ETA** | 预计剩余时间，按已完成文件的平均速度线性估算。 |
| **UI 外包装** | 面向非专业用户的 PySide6 桌面界面，进程内调用引擎，不改变 CLI 行为。 |
| **全量安装包** | Inno Setup 7 制作的标准安装程序（`installer/install.iss`）：代码快照 + 内置绿色 Python（含全部依赖）+ CPU/CUDA 双 torch + 模型权重全部内置，用户安装即用、无需联网。 |
| **推理引擎切换** | 安装包内 torch_cpu/ 与 torch_cuda/ 两套独立 torch，应用内选择（自动/CPU/CUDA）写 `torch.ini`，启动时 `uvr_lite/__init__.py` 把对应目录插入 sys.path（重启生效）。 |
| **绿色 Python** | python-build-standalone 发行版，打包机下载后与依赖一起打进安装包；运行时直接使用（无 venv）。 |
| **单安装目录** | 全部组件（代码 + Python + 双 torch + 模型）位于用户选择的一个目录内；卸载 = 控制面板卸载（删目录 + 快捷方式 + 注册表）。 |
| **快捷方式** | 安装后生成桌面 + 开始菜单两处入口（♪ 图标），指向 `python\pythonw.exe -m uvr_lite.ui`（WorkingDir=app）。 |
| **参数记忆** | QSettings 记住上次的模型/参数/输出目录，下次启动直接可用。 |

## 文件角色

| 路径 | 角色 |
|---|---|
| `uvr_lite/engine.py` | 分离引擎（进度回调 + 取消） |
| `uvr_lite/__init__.py` | 版本 + torch 二进制切换（torch.ini / UVR_TORCH） |
| `uvr_lite/ui/` | UI 包（主窗口、推理线程、设备切换） |
| `installer/install.iss` | Inno Setup 全量安装脚本（文件复制 + 快捷方式 + 卸载清理） |
| `installer/copy_app.py` | 打包用代码快照复制（排除规则） |
| `scripts/build_installer.py` | 打包脚本：组装 bundle（Python+依赖+双 torch+模型）→ ISCC 编译 |

## 决策索引

- ADR-001：UI 外包装（UI/引擎接口/分发全套决策，24 条 + 5 项默认）；2026-08-03 更新：分发改为 **Inno Setup 全量安装包**（内置绿色 Python + 双 torch + fp16 模型，用户免联网下载；此前 PyInstaller 向导因收集 bug 弃用、NSIS 因 ~2GB 上限弃用）
