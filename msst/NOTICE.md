# Notice

`msst/` 下的 Python 文件 vendored 自 [ZFTurbo / Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)（MIT），只保留推理所需的子集。部分文件没有留下上游版权头。这里只记录本副本里实际能看到的署名，不转载许可全文。

The Python files under `msst/` are vendored from [ZFTurbo / Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) (MIT), trimmed to the inference subset. Some files in this copy do not keep an upstream copyright header. This note records attribution that is actually present in the tree; it does not reproduce a license text.

- `msst/utils/model_utils.py` 保留了作者行：`Roman Solovyev (ZFTurbo)`（https://github.com/ZFTurbo/）。
- `msst/models/bs_roformer/mel_filters.py` 的模块 docstring 写明：这是 librosa 梅尔滤波器的迷你实现，作者为 Brian McFee 等，MIT。具体说明以该文件 docstring 为准。
- `msst/models/bs_roformer/attend.py` 的实现风格来自 lucidrains（einops / `Attend`）。本副本没有保留原文件头。
