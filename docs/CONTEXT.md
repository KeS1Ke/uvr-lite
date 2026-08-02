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
| **安装向导（installer）** | 独立的 Windows GUI 安装程序（PyInstaller onefile 打包，无 torch），引导非专业用户完成安装。 |
| **绿色 Python** | python-build-standalone 发行版（自带 pip、绿色解压、不污染系统注册表），安装器在系统无 Python 时自动下载到安装目录。 |
| **单安装目录** | 安装模型：代码 + 绿色 Python + `.venv` + `models/` 全部位于用户选择的一个目录内；卸载 = 删目录 + 删快捷方式。 |
| **覆盖升级** | 检测到已安装时更新代码/依赖，保留 `.venv` 与 `models/`（免重下 640MB 模型）。 |
| **镜像回退** | 下载失败时自动切换备用源（绿色 Python → ghproxy/国内镜像；torch → 清华 PyPI；模型 → HuggingFace 镜像），与 `download.py` 现有多源设计一致。 |
| **快捷方式** | 安装后生成桌面 + 开始菜单两处入口（♪ 图标），指向 `.venv\Scripts\pythonw.exe -m uvr_lite.ui`（无黑窗）。 |
| **参数记忆** | QSettings 记住上次的模型/参数/输出目录，下次启动直接可用。 |

## 文件角色

| 路径 | 角色 |
|---|---|
| `uvr_lite/engine.py` | 分离引擎（将被加进度回调，票 1） |
| `uvr_lite/ui/` | UI 包（主窗口、推理线程、模型下载按钮） |
| `installer/` | 安装向导源码（独立于 `uvr_lite` 包，PyInstaller 打包） |
| `scripts/make_icon.py` | ♪ 图标生成器（PIL，输出 ico+png） |
| `scripts/build_installer.py` | 安装器打包脚本（onefile，携带代码快照） |

## 决策索引

- ADR-001：UI 外包装（UI/引擎接口/安装器/分发全套决策，24 条 + 5 项默认）
