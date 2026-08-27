"""渲染原曲与新作的对数频谱对比图 (Pillow) -> output/spectrogram-*.png"""
import os

import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 22050
N_FFT = 2048
HOP = 512

def load_mono(path):
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr

def spectrogram_image(path, out_path, max_s=None, title=""):
    x, sr = load_mono(path)
    if max_s:
        x = x[:int(max_s * sr)]
    win = np.hanning(N_FFT).astype(np.float32)
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    S = np.fft.rfft(xp[idx] * win[None, :], axis=1)
    mag = np.abs(S)
    # 对数压缩 + 归一化 (0-255)
    logmag = 20 * np.log10(mag + 1e-6)
    logmag -= logmag.max()
    logmag /= 60.0  # 60dB 动态范围映射
    img = np.clip(logmag, -1, 0)
    img = ((img + 1) * 255).astype(np.uint8)
    # 频率轴对数映射 (20Hz-10kHz)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    fmax = 10000
    sel = freqs <= fmax
    img = img[:, sel]
    freqs = freqs[sel]
    # 对数频率重采样到 400 行
    n_rows = 400
    log_f = np.linspace(np.log10(30), np.log10(fmax), n_rows)
    row_idx = np.clip(np.searchsorted(freqs, 10 ** log_f), 0, len(freqs) - 1)
    img_rows = img[:, row_idx].T  # (rows, frames)
    # 时间压缩到 900 列
    n_cols = min(900, img_rows.shape[1])
    step = img_rows.shape[1] // n_cols
    img_small = img_rows[:, ::max(1, step)][:, :n_cols]
    im = Image.fromarray(img_small)
    im = im.resize((900, 400), Image.LANCZOS)
    # 染色: 冷色低能量 -> 暖色高能量（a 保持 0~255 整数域，浮点直接赋 uint8 会被截断为 0）
    rgb = np.zeros((400, 900, 3), dtype=np.uint8)
    a = np.asarray(im, dtype=float)
    rgb[..., 0] = np.clip(a * 1.6, 0, 255)          # 红
    rgb[..., 1] = np.clip(a * 0.9 - 0.15, 0, 255)   # 绿
    rgb[..., 2] = np.clip(a * 0.5 - 0.25, 0, 255)   # 蓝
    out = Image.fromarray(rgb)
    out.save(out_path)
    print(f"{out_path}  ({img_small.shape[1]}x{img_small.shape[0]})")

if __name__ == "__main__":
    os.makedirs(os.path.join(BASE, "output"), exist_ok=True)
    spectrogram_image(os.path.join(BASE, "tmp", "orig_mono.wav"),
                      os.path.join(BASE, "output", "spectrogram-original.png"),
                      title="GO GHOST original")
    spectrogram_image(os.path.join(BASE, "tmp", "new.wav"),
                      os.path.join(BASE, "output", "spectrogram-new.png"),
                      title="inspired piece")
