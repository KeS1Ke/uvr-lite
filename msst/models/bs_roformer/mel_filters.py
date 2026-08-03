# coding: utf-8
"""librosa.filters.mel 迷你实现（vendored，MIT 许可保留）。

替代完整 librosa 依赖（连带 scipy/numba/llvmlite/scikit-learn 等约 320MB），
仅提供 mel band 模型所需的三角梅尔滤波器组。逻辑与 librosa 0.10 的
mel / mel_frequencies / hz_to_mel / mel_to_hz / fft_frequencies 一致
（librosa 作者 Brian McFee 等，MIT 许可）。

上游参考: https://github.com/librosa/librosa/blob/main/librosa/filters.py
"""

import numpy as np


def fft_frequencies(sr=22050, n_fft=2048):
    """FFT 各 bin 的中心频率（Hz）。"""
    return np.fft.rfftfreq(n_fft, d=1.0 / sr)


def hz_to_mel(frequencies, htk=False):
    """Hz → mel 刻度；htk=False 用 Slaney 公式（与 librosa 默认一致）。"""
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if np.ndim(frequencies):
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def mel_to_hz(mels, htk=False):
    """mel 刻度 → Hz；htk=False 用 Slaney 公式。"""
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if np.ndim(mels):
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """mel 等距中心频率（Hz）。"""
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    mels = np.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels, htk=htk)


def mel(sr=22050, n_fft=2048, n_mels=128, fmin=0.0, fmax=None,
        htk=False, norm="slaney", dtype=np.float32):
    """三角梅尔滤波器组，shape (n_mels, 1 + n_fft // 2)。"""
    if fmax is None:
        fmax = float(sr) / 2
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # 'Center freqs' of FFT bins 与 mel bands
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        # Slaney 归一化：三角面积归一，与 librosa 默认一致
        enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    return weights
