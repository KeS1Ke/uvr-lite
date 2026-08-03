# uvr-lite

[English](README.md) | **简体中文**

![version](https://img.shields.io/badge/version-0.1.0-8A2BE2)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![license](https://img.shields.io/badge/license-MIT-green)
![platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0078D6)
![inference](https://img.shields.io/badge/inference-PyTorch%20CPU%20%2F%20CUDA-orange)
![downloads](https://img.shields.io/github/downloads/KeS1Ke/uvr-lite/total)

**轻量级人声 / 伴奏分离工具** —— 一个模型文件（约 640 MB），一键安装，输出两个无损音轨。同时提供 **中文桌面界面（Windows）** 与 **命令行**。

```bash
uvr-lite separate 歌曲.flac -o output
# → output/歌曲-vocals.flac        （人声轨）
# → output/歌曲-instrumental.flac  （伴奏轨，= 原曲 − 人声，数学无损）
```

## 演示

下图为一段 **MiMo TTS 合成人声 + 合成器伴奏** 混合音频的分离效果（对数频谱，冷色低能量 → 暖色高能量）：

![demo](docs/images/demo-spectrograms.png)

- **中间（人声轨）**：平滑的谐波横纹随旋律起伏 —— 歌声被完整提取
- **右侧（伴奏轨）**：低频能量带 + 打击乐瞬态 —— 节奏部分被完整保留

## 特性

- **一键部署**：`install.bat`（Windows）/ `install.sh`（Linux/macOS）自动完成 venv + 依赖 + torch（CPU/CUDA 自动分流）+ 模型下载（SHA256 校验）+ 冒烟测试
- **主力模型**：BS-RoFormer ep317（viperx 训练，SDR ≈ 10.9–12.9 dB），RTX 4060 上整曲（约 3 分钟）约 **51 秒**
- **双格式输出**：FLAC（16/24 bit）或 WAV，保持 44.1 kHz 原采样率
- **无 GUI、无训练代码**：仅推理，仓库代码 < 1 MB
- 可选多模型：`mel_band_karaoke`（Mel-Band RoFormer Karaoke，aufr33 & viperx 训练）

## 桌面界面（Windows，推荐非专业用户）

**下载**（按你的硬件选一个变体）：
- [uvr-lite-setup-cpu.exe](https://github.com/KeS1Ke/uvr-lite/releases/latest/download/uvr-lite-setup-cpu.exe) —— **约 325 MB**，仅 CPU 版 torch（无需独立显卡；大多数用户）
- [uvr-lite-setup-full.exe](https://github.com/KeS1Ke/uvr-lite/releases/latest/download/uvr-lite-setup-full.exe) —— **约 3.5 GB**，CPU + CUDA 双版 torch（NVIDIA 显卡用户）

两者都是**全量安装包**——Python、PyTorch、fp16 瘦身模型（320 MB，比原版 639 MB 小一半，输出差异不可闻）全部内置，安装过程无需联网下载任何组件

1. **双击安装**，选择安装位置（默认：你的用户目录）——所有文件装进一个文件夹，不会散落
2. 安装程序把全部组件复制到本机（**磁盘占用约 2 GB**（cpu 变体）/ **约 5.5 GB**（full））；**全程无需联网**——下载到什么就用什么
3. **完成**——桌面与开始菜单出现 **♪ 快捷方式**，双击即可打开界面
4. 把歌曲拖进窗口（或选择文件夹），选好模型，点「**开始分离**」——实时进度 + 预计剩余时间；处理完的文件打 **✓**，无法识别的格式在开始前就被标 **✗** 并跳过

小贴士：

- **推理引擎**：界面里可选 **自动 / CPU / CUDA**（自动模式：有独立显卡用 CUDA 版，否则 CPU 版）；切换后重启 uvr-lite 生效
- **升级**：重新运行安装程序即可——原地覆盖更新，保留你的设置
- **卸载**：控制面板 → 程序和功能 → uvr-lite（或运行安装目录下的 `Uninstall.exe`）——删除快捷方式、注册表与安装目录
- 界面为中文（面向亲友设计的）；命令行用法见下文，供高级用户使用

## 快速开始

### 一键安装

**Windows**

```bat
install.bat
```

**Linux / macOS**

```bash
bash install.sh
```

脚本自动完成：创建虚拟环境 `.venv` → 检测 NVIDIA 显卡（有卡装 CUDA 版 torch / 无卡装 CPU 版）→ 安装依赖 → 下载模型权重（约 320 MB，fp16 瘦身版，SHA256 校验）→ （GPU 环境）冒烟测试。

### 手动安装

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate.bat（PowerShell: .venv\Scripts\Activate.ps1）
pip install -e .
uvr-lite download                                    # 下载模型（约 320 MB）
```

## 用法

> **每次使用前**先激活虚拟环境（安装脚本里的激活不跨终端生效）。请按你的 shell 选择命令：
>
> **pwsh（PowerShell 7+）**（Windows）— 打开：Win+R 输入 `pwsh` 或开始菜单「PowerShell 7」；未安装先 `winget install Microsoft.PowerShell`
> ```powershell
> .venv\Scripts\Activate.ps1
> # 若被执行策略拦截，在该终端里运行一次：Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
>
> **powershell（Windows PowerShell 5.1，系统自带）**（Windows）— 打开：Win+R 输入 `powershell` 或开始菜单「Windows PowerShell」
> ```powershell
> .venv\Scripts\Activate.ps1
> # 若被执行策略拦截，在该终端里运行一次：Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
>
> 两种终端**共用同一个 `Activate.ps1` 与同一个 `.venv`**，仅入口命令不同（`pwsh` / `powershell`），可随意混用；执行策略按终端**分开记忆**，被哪个拦截就在哪个里设置一次。激活后先验证：`uvr-lite --version`。
>
> **cmd**（Windows）
> ```bat
> .venv\Scripts\activate.bat
> ```
>
> **bash**（Linux / macOS）
> ```bash
> source .venv/bin/activate
> ```
>
> 或直接调用完整路径：`.venv/bin/uvr-lite`（Windows: `.venv\Scripts\uvr-lite.exe`）。

```bash
# 单文件
uvr-lite separate song.mp3 -o output

# 多文件 / 指定格式与位深
uvr-lite separate a.flac b.wav -o out --format flac --pcm 24

# 其他模型（备选）
uvr-lite separate song.flac -m mel_band_karaoke

# 低显存 GPU：调小批大小防 OOM
uvr-lite separate song.flac --batch-size 1

# 质量/速度开关：1 = 无重叠（约 2 倍提速），2 = 默认，更大更稳
uvr-lite separate song.flac --num-overlap 1

# 查看模型状态 / 强制重下
uvr-lite models
uvr-lite download --force
```

| 参数 | 说明 |
|---|---|
| `-m, --model` | `bs_roformer_ep317`（默认）/ `mel_band_karaoke` |
| `--format` | `auto`（按峰值自动选 flac/wav，默认）/ `flac` / `wav` |
| `--pcm` | FLAC 位深 `16` / `24`（默认） |
| `--device` | `auto`（默认）/ `cpu` / `cuda` / `mps` |
| `--bigshifts N` | 圆形时移平均次数，>1 提升质量、线性增耗时（默认 1） |
| `--batch-size N` | 推理批大小（默认取模型配置）；低显存 GPU 可设 `1` 防 OOM |
| `--num-overlap N` | 重叠窗口数（质量/速度开关）：`1` 最快约 2 倍，`2` 默认，更大更稳 |
| `--tta` | 测试时增强（极性/声道反转平均，3 倍耗时，默认关） |

**注意事项**

- **mp3 输入**需 libsndfile ≥ 1.1（Windows 自带；Linux 装 `libsndfile1` 或升级 `soundfile` 包）
- **CPU 推理**约 6 倍实时（3 分钟歌曲 ≈ 17 分钟）——建议使用 GPU
- **磁盘占用**：cpu 变体约 2 GB，full 变体约 5.5 GB（双 torch + fp16 模型 320MB + Python）
- **模型校验缓存**：SHA256 校验一次后写入 `*.verified` 标记，后续运行跳过整文件哈希

## 工作原理

```
输入音频 → soundfile+soxr 解码 (44.1kHz，m4a 兜底 audioread) → （可选归一化）
       → BigShifts 圆形时移平均 → BS-RoFormer 前向（vocals 掩码）
       → instrumental = 原混合 − vocals（数学无损）
       → soundfile 写 FLAC/WAV
```

- **引擎**：`msst/` 为 [ZFTurbo Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) 的**推理最小子集**（裁剪训练/验证/集成/GUI，仅保留 RoFormer 家族推理路径）
- **模型**：默认模型托管于本仓库 [GitHub Releases](https://github.com/KeS1Ke/uvr-lite/releases/tag/models)，为 **fp16 瘦身版**（320 MB；`scripts/strip_model.py` 由原版转换，加载时透明转回 fp32 推理，输出差异约 -80 dB 不可闻）；原版 [TRvlvr/model_repo](https://github.com/TRvlvr/model_repo) 权重保留为回退镜像。SHA256 完整性校验，不入 git
- **批量处理复用会话**：多文件队列共用一个已加载模型（`Separator` 会话），不再逐文件重载 640MB 权重
- **代码结构**

```
uvr-lite/
├── uvr_lite/          # CLI 包：separate / download / models 命令
│   ├── engine.py      #   分离引擎（bigshifts 平均、instrumental 数学无损）
│   ├── models.py      #   模型注册表（URL + SHA256）
│   ├── download.py    #   流式下载 + 完整性校验
│   └── configs/       #   模型配置 yaml
├── msst/              # vendored 推理引擎（ZFTurbo MSST 裁剪子集，MIT）
├── install.bat|sh     # 一键安装脚本
└── scripts/           # 可选配套：analyze（DSP 分析）/ compose（算法作曲）/ render_spectro（频谱图）
```

### 可选配套脚本

`scripts/` 下提供三个纯 numpy 工具（独立于分离核心，仅 numpy/Pillow 依赖）：

- `scripts/analyze.py`：DSP 原曲分析（BPM/调性/和弦/结构/音色统计），输出 Markdown 报告
- `scripts/compose.py`：算法作曲引擎（可复现的旋律 + 编曲合成）
- `scripts/render_spectro.py`：对数频谱图渲染（PNG，本仓库演示图即由此生成）

## 致谢（Reference）

本项目是独立仓库（非 fork），代码主体为自研 CLI 封装 + ZFTurbo MSST 推理子集，模型与 **Ultimate Vocal Remover** 生态同源。感谢：

- **[Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui)**（Anjok07/aufr33，MIT）—— 人声分离领域标杆项目，本项目的模型生态与设计理念均受其启发
- **[ZFTurbo / Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)**（MIT）—— 推理引擎与模型训练框架来源，`msst/` 目录为其裁剪子集
- **viperx / aufr33** —— BS-RoFormer 与 Mel-Band RoFormer 模型训练者
- **[TRvlvr/model_repo](https://github.com/TRvlvr/model_repo)** —— 模型权重托管仓库

按照 MIT 许可要求：使用上述模型的第三方项目须保留对 UVR 及其开发者的署名。

## 许可

MIT License。详见 [LICENSE](LICENSE)。`msst/` 子目录保留 ZFTurbo MSST 的原始版权声明。
