# uvr-lite

[English](README.md) | **简体中文**

![version](https://img.shields.io/badge/version-0.1.0-8A2BE2)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![license](https://img.shields.io/badge/license-MIT-green)
![platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0078D6)
![inference](https://img.shields.io/badge/inference-PyTorch%20CPU%20%2F%20CUDA-orange)

**轻量级人声 / 伴奏分离命令行工具** —— 一个模型文件（约 640 MB），一条命令，输出两个无损音轨。

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

脚本自动完成：创建虚拟环境 `.venv` → 检测 NVIDIA 显卡（有卡装 CUDA 版 torch / 无卡装 CPU 版）→ 安装依赖 → 下载模型权重（约 640 MB，SHA256 校验）→ （GPU 环境）冒烟测试。

### 手动安装

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
uvr-lite download                                    # 下载模型（约 640 MB）
```

## 用法

> **每次使用前**先激活虚拟环境（安装脚本里的激活不跨终端生效）。请按你的 shell 选择命令：
>
> **PowerShell**（Windows）
> ```powershell
> .venv\Scripts\Activate.ps1
> # 若被执行策略拦截，先运行一次：Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
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
| `--tta` | 测试时增强（极性/声道反转平均，3 倍耗时，默认关） |

**注意事项**

- **mp3 输入**需 libsndfile ≥ 1.1（Windows 自带；Linux 装 `libsndfile1` 或升级 `soundfile` 包）
- **CPU 推理**约 6 倍实时（3 分钟歌曲 ≈ 17 分钟）——建议使用 GPU
- **磁盘占用**约 4 GB（CUDA torch 2.5GB + 模型 640MB + 依赖）

## 工作原理

```
输入音频 → librosa 解码 (44.1kHz) → （可选归一化）
       → BigShifts 圆形时移平均 → BS-RoFormer 前向（vocals 掩码）
       → instrumental = 原混合 − vocals（数学无损）
       → soundfile 写 FLAC/WAV
```

- **引擎**：`msst/` 为 [ZFTurbo Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) 的**推理最小子集**（裁剪训练/验证/集成/GUI，仅保留 RoFormer 家族推理路径）
- **模型**：`.ckpt` 权重托管于 [TRvlvr/model_repo](https://github.com/TRvlvr/model_repo)（UVR 官方模型仓库），SHA256 完整性校验，不入 git
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
