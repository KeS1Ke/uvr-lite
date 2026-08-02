# -*- coding: utf-8 -*-
"""
goghost-origin.flac 多层面音乐分析
- 节拍: 谱通量 onset 包络 -> 自相关 + 梳状对齐 -> BPM / 拍相位 / 拍号 / swing
- 调性: 12-bin chromagram + Krumhansl-Schmuckler 模板 -> 主音/大小调
- 和声: 每小节 chroma -> 三和弦模板匹配 -> 级数序列
- 结构: chroma novelty + 能量 -> 段落边界与能量等级
- 音色: 频谱质心 / rolloff / 频段能量比 / 平坦度 / ZCR
- 动态: RMS 包络 / 峰值 / crest / 动态范围 / 近似响度
- 立体声: 声道相关 / 中侧能量比

全部只用 numpy 手写 DSP。输出 analysis/goghost-analysis.md + tmp/analysis.json
"""
import json
import os
import wave
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONO_WAV = os.path.join(BASE, "tmp", "orig_mono.wav")
STEREO_WAV = os.path.join(BASE, "tmp", "orig_stereo.wav")
OUT_JSON = os.path.join(BASE, "tmp", "analysis.json")
OUT_MD = os.path.join(BASE, "analysis", "goghost-analysis.md")

SR = 22050
N_FFT = 2048
HOP = 512

# ---------------- 基础工具 ----------------

def load_wav(path, mono=True):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch)
        if mono:
            data = data.mean(axis=1)
    return data, sr, ch

def stft(x, n_fft=N_FFT, hop=HOP):
    win = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = xp[idx] * win[None, :]
    return np.fft.rfft(frames, axis=1)

def db(x, eps=1e-9):
    return 10.0 * np.log10(np.maximum(x, eps))

# ---------------- 1. 节拍 ----------------

def onset_envelope(S):
    logmag = np.log1p(np.abs(S) * 1000.0)
    flux = np.diff(logmag, axis=0)
    flux = np.maximum(flux, 0.0).sum(axis=1)
    flux = flux - np.median(flux)
    k = np.ones(3) / 3.0
    return np.convolve(flux, k, "same")

def beat_tracking(env, hop, sr, bpm_min=60, bpm_max=180):
    e = env - env.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lag_min = int(np.ceil(60.0 * sr / (hop * bpm_max)))
    lag_max = int(np.floor(60.0 * sr / (hop * bpm_min)))
    seg = ac[lag_min:lag_max + 1]
    # 局部极大候选
    cands = []
    for i in range(2, len(seg) - 2):
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.15:
            cands.append(lag_min + i)
    if not cands:
        cands = [lag_min + int(np.argmax(seg))]
    # 对齐质量: 拍点能量均值 - 拍间能量均值(相对同尺度)
    def align_quality(p):
        idx = np.arange(1, 33) * float(p)
        idx = idx[idx < len(env) - 1]
        if len(idx) < 4:
            return -1e18
        on = np.array([env[max(0, min(len(env) - 1, int(round(i))))] for i in idx])
        off = np.array([env[max(0, min(len(env) - 1, int(round(i + p / 2))))] for i in idx])
        return float(on.mean() - off.mean())
    # 在初选周期 ±12% 内微调, 并用倍频消歧
    p0 = cands[int(np.argmax([align_quality(c) for c in cands]))]
    best_p, best_aq = p0, align_quality(p0)
    for mult in (0.5, 2.0, 3.0, 1.5, 0.666):
        pp = p0 * mult
        if lag_min <= pp <= lag_max and align_quality(pp) > best_aq * 1.08:
            best_p, best_aq = int(round(pp)), align_quality(pp)
    # 局部扫描精化
    for pp in range(max(lag_min, best_p - 3), min(lag_max, best_p + 4)):
        aq = align_quality(pp)
        if aq > best_aq:
            best_p, best_aq = pp, aq
    bpm = 60.0 * sr / (hop * best_p)
    # 相位: 最大化拍点包络和
    best_ph, best_ph_s = 0, -1e18
    for ph in range(0, best_p, 2):
        s = env[ph::best_p].sum()
        if s > best_ph_s:
            best_ph_s, best_ph = s, ph
    beats = best_ph + np.arange(0, len(env), best_p)
    beats = beats[beats < len(env)]
    # 置信度: 拍点位置能量 vs 平均
    conf = np.mean(env[beats]) / (np.mean(env) + 1e-9)
    return bpm, best_p, beats, conf

def time_signature(env, beats, period, bar_cands=(2, 3, 4)):
    e = np.array([env[max(0, min(len(env) - 1, b))] for b in beats])
    best_b, best_score = 4, 0.0
    for B in bar_cands:
        n = len(e) - len(e) % B
        em = e[:n].reshape(-1, B).mean(axis=0)
        score = em.std()
        if score > best_score:
            best_b, best_score = B, score
    return best_b

def swing_ratio(env, beats, period, sr):
    """8分音符交替位置能量比, 体现 shuffle 程度"""
    half = max(2, period // 2)
    on = np.array([env[min(len(env) - 1, int(b))] for b in beats if b < len(env)])
    off = np.array([env[min(len(env) - 1, int(b) + half)] for b in beats if b + half < len(env)])
    if len(off) < 8 or off.mean() <= 1e-6:
        return 0.0
    return float(on.mean() / off.mean())

# ---------------- 2. 调性 ----------------

def chromagram(S, sr, n_fft=N_FFT):
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    with np.errstate(divide="ignore"):
        midi = 69 + 12 * np.log2(np.where(freqs > 0, freqs, 1.0) / 440.0)
    midi[0] = -1e9
    sel = (midi >= 24) & (midi <= 96)
    pc = np.mod(np.round(midi[sel]).astype(int), 12)
    mag = np.abs(S)[:, sel]
    chroma = np.zeros((S.shape[0], 12), dtype=np.float64)
    for c in range(12):
        chroma[:, c] = mag[:, pc == c].sum(axis=1)
    total = chroma.sum(axis=1, keepdims=True)
    return chroma / (total + 1e-9)

KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def detect_key(chroma):
    """逐窗口 KS 匹配投票, 兼顾全曲中位数 chroma"""
    def ks_corr(cmean):
        out = {}
        for root in range(12):
            for mode, prof in (("major", KS_MAJ), ("minor", KS_MIN)):
                out[f"{NOTE_NAMES[root]} {mode}"] = np.corrcoef(cmean, np.roll(prof, root))[0, 1]
        return out
    # 全曲: 用中位数 chroma 抗段差异
    cmed = np.median(chroma, axis=0)
    scores = ks_corr(cmed)
    # 时间窗口投票 (每 ~8s 一窗)
    W = 512
    votes = np.zeros(24)
    for i in range(0, len(chroma) - W, W):
        cw = chroma[i:i + W].mean(axis=0)
        vs = ks_corr(cw)
        k = max(vs, key=vs.get)
        votes[list(scores.keys()).index(k)] += 1
    names = list(scores.keys())
    # 综合: 全局匹配分 + 投票权重
    combined = {n: scores[n] + 0.25 * (votes[i] / max(1, votes.max())) for i, n in enumerate(names)}
    ranked = sorted(combined, key=combined.get, reverse=True)
    top3 = [(n, round(combined[n], 3), int(votes[names.index(n)])) for n in ranked[:3]]
    return top3[0][0], top3, cmed

# ---------------- 3. 和弦 ----------------

MAJ_TRI = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
MIN_TRI = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

def chord_at(chroma_mean):
    best = (None, -1e18)
    for root in range(12):
        for mode, tmpl in (("", MAJ_TRI), ("m", MIN_TRI)):
            t = np.roll(tmpl, root)
            corr = np.corrcoef(chroma_mean, t)[0, 1]
            if corr > best[1]:
                best = (f"{NOTE_NAMES[root]}{mode}", corr)
    flat = np.exp(-np.sum(chroma_mean * np.log(chroma_mean + 1e-12)))
    return best[0], best[1], float(flat)

# ---------------- 4. 结构分段 ----------------

def structure_segments(chroma, S, env, sr, hop, min_gap_s=7.0):
    cn = np.sqrt(np.maximum(chroma, 0))
    cnorm = cn / (np.linalg.norm(cn, axis=1, keepdims=True) + 1e-9)
    nlen = min(len(cnorm), len(env))
    nov = np.linalg.norm(np.diff(cnorm[:nlen], axis=0), axis=1)
    nov = nov + 0.3 * np.abs(np.diff(env[:nlen], axis=0))
    nov = nov[: nlen - 1]
    k = int(round(1.5 * sr / hop))
    nov = np.convolve(nov, np.ones(k) / k, "same")
    nov = nov / (nov.std() + 1e-9)
    th = nov.mean() + 1.2 * nov.std()
    min_gap = int(min_gap_s * sr / hop)
    idxs = []
    last = -min_gap
    for i in range(2, len(nov) - 2):
        if nov[i] > th and nov[i] >= nov[i - 1] and nov[i] > nov[i + 1] and (i - last) >= min_gap:
            idxs.append(i)
            last = i
    bounds = [0] + idxs + [len(nov)]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        t0, t1 = a * hop / sr, b * hop / sr
        rms_db = db(np.mean(np.abs(S[a:b]) ** 2))
        cen = spectral_centroid(S[a:b], sr)
        segs.append({"t0": round(t0, 1), "t1": round(t1, 1),
                     "dur": round(t1 - t0, 1), "rms_db": round(float(rms_db), 1),
                     "centroid_hz": int(cen), "label": ""})
    return segs

def label_segments(segs):
    if not segs:
        return
    rms = [s["rms_db"] for s in segs]
    lo, hi = min(rms), max(rms)
    for s in segs:
        r = (s["rms_db"] - lo) / (hi - lo + 1e-9)
        if r > 0.66:
            s["label"] = "副歌/高潮"
        elif r > 0.33:
            s["label"] = "主歌/中段"
        else:
            s["label"] = "前奏/间奏/尾声"

# ---------------- 5. 音色 ----------------

def spectral_centroid(S, sr, n_fft=N_FFT):
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mag = np.abs(S)
    return float((mag * freqs[None, :]).sum(axis=1).mean() / (mag.sum(axis=1).mean() + 1e-9))

def spectral_rolloff(S, sr, n_fft=N_FFT, pct=0.85):
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mag = np.abs(S)
    cum = np.cumsum(mag, axis=1)
    total = cum[:, -1:]
    frac = cum / (total + 1e-9)
    out = []
    for f in range(frac.shape[0]):
        i = min(len(freqs) - 1, np.searchsorted(frac[f], pct))
        out.append(freqs[i])
    return float(np.mean(out))

def band_ratios(S, sr, n_fft=N_FFT):
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bands = {"sub(0-60)": (0, 60), "bass(60-250)": (60, 250),
             "mid(250-2k)": (250, 2000), "hi(2k-8k)": (2000, 8000), "air(8k+)": (8000, 1e9)}
    mag = np.abs(S)
    total = mag.sum() + 1e-9
    out = {}
    for name, (flo, fhi) in bands.items():
        m = (freqs >= flo) & (freqs < fhi)
        out[name] = round(float(10 * np.log10(mag[:, m].sum() / total)), 1)
    return out

def spectral_flatness(S, n_fft=N_FFT):
    mag = np.abs(S) + 1e-12
    g = np.exp(np.mean(np.log(mag), axis=1))
    a = np.mean(mag, axis=1)
    return float(np.mean(g / (a + 1e-12)))

def zero_crossing(x, sr):
    z = np.abs(np.diff(np.signbit(x).astype(np.int8))).mean()
    return float(z * sr / 2.0)  # 每秒过零次数近似

# ---------------- 6. 动态 ----------------

def dynamics(x, sr):
    hop = int(0.05 * sr)
    n = 1 + (len(x) - hop) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x[:n * hop], hop)[::hop]
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    peak = float(np.abs(x).max())
    rms_db = db(rms)
    return {
        "peak_db": round(float(db(peak)), 1),
        "overall_rms_db": round(float(db(np.sqrt(np.mean(x ** 2) + 1e-12))), 1),
        "crest_db": round(float(db(peak / (np.sqrt(np.mean(x ** 2)) + 1e-12))), 1),
        "frame_rms_avg_db": round(float(rms_db.mean()), 1),
        "frame_rms_std_db": round(float(rms_db.std()), 1),
        "dynamic_range_db": round(float(rms_db.quantile if hasattr(rms_db, "quantile") else np.percentile(rms_db, 95) - np.percentile(rms_db, 5)), 1),
        "percentile_db": [round(float(p), 1) for p in np.percentile(rms_db, [5, 25, 50, 75, 95])],
    }

def loudness_approx(x, sr):
    hop = int(0.4 * sr)
    n = 1 + (len(x) - hop) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x[:n * hop], hop)[::hop]
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return round(float(db(rms).mean()), 1)

def stereo_stats(stereo, ch):
    L = stereo[:, 0].astype(np.float64)
    R = stereo[:, 1].astype(np.float64)
    corr = float(np.corrcoef(L, R)[0, 1])
    mid = (L + R) / 2.0
    side = (L - R) / 2.0
    ms_db = float(db(np.mean(side ** 2) / (np.mean(mid ** 2) + 1e-12)))
    return {"lr_corr": round(corr, 3), "side_vs_mid_db": round(ms_db, 1)}

# ---------------- 主流程 ----------------

def tempo_refine(env, hop, sr, bpm0):
    """调制频谱修正: 最强调制频率 (2-6Hz, 8分脉冲) 的 2:1 亚谐波即拍速;
    密集混音下相位类指标不可靠, 直接用纯周期性强度裁决"""
    N = len(env)
    spec = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(N)))
    fr = np.fft.rfftfreq(N, hop / sr)
    sel = (fr >= 1.5) & (fr <= 6.0)
    f_peak = fr[sel][np.argmax(spec[sel])]
    bpm_pulse = f_peak * 60.0
    # 8分脉冲 -> 拍; 若落在合理范围则采用, 否则回退自相关结果
    cand = bpm_pulse / 2.0
    if 55 <= cand <= 210:
        return cand
    return bpm0

def main():
    print("[1/7] 载入音频 ...")
    x, sr, _ = load_wav(MONO_WAV)
    stereo, ssr, sch = load_wav(STEREO_WAV, mono=False)
    dur = len(x) / sr
    print(f"     {dur:.1f}s, {sr}Hz")

    print("[2/7] STFT + onset 包络 ...")
    S = stft(x)
    env = onset_envelope(S)
    bpm0, period0, beats0, conf0 = beat_tracking(env, HOP, sr)
    bpm = tempo_refine(env, HOP, sr, bpm0)
    if bpm != bpm0:
        print(f"     调制频谱修正: {bpm0:.1f} -> {bpm:.1f} BPM")
    period = int(round(60.0 * sr / (HOP * bpm)))
    best_ph, best_ph_s = 0, -1e18
    for ph in range(0, period, 2):
        s = env[ph::period].sum()
        if s > best_ph_s:
            best_ph_s, best_ph = s, ph
    beats = best_ph + np.arange(0, len(env), period)
    beats = beats[beats < len(env)]
    conf = np.mean(env[beats]) / (np.mean(env) + 1e-9)
    bpm = round(bpm, 1)
    beat_times = (beats * HOP / sr).tolist()
    bars_per_beat = 4  # 调制频谱 2:1 家族 + 摇滚惯例 (4/4)
    swing = round(swing_ratio(env, beats, period, sr), 2)
    print(f"     BPM={bpm}  置信度={conf:.2f}  拍号={bars_per_beat}/4  swing={swing}")

    print("[3/7] chroma + 调性 ...")
    chroma = chromagram(S, sr)
    key, key_top3, _ = detect_key(chroma)
    print(f"     调性: {key}  前三候选: {key_top3}")

    print("[4/7] 和弦进行 ...")
    beat_np = beats.astype(int)
    n_bars = len(beat_np) // bars_per_beat
    chords = []
    for b in range(n_bars):
        f0 = beat_np[b * bars_per_beat]
        f1 = beat_np[(b + 1) * bars_per_beat] if (b + 1) * bars_per_beat < len(beat_np) else len(chroma)
        cmean = chroma[f0:f1].mean(axis=0)
        cmean = cmean / (cmean.sum() + 1e-9)
        name, corr, flat = chord_at(cmean)
        t0 = round(beat_np[b * bars_per_beat] * HOP / sr, 2)
        chords.append({"t": t0, "bar": b + 1, "chord": name, "corr": round(corr, 2), "flat": round(flat, 3)})

    print("[5/7] 结构分段 ...")
    segs = structure_segments(chroma, S, env, sr, HOP)
    label_segments(segs)

    print("[6/7] 音色特征 ...")
    timbre = {
        "centroid_hz": int(spectral_centroid(S, sr)),
        "rolloff85_hz": int(spectral_rolloff(S, sr)),
        "flatness": round(spectral_flatness(S), 4),
        "zcr_per_sec": int(zero_crossing(x, sr)),
        "bands_db": band_ratios(S, sr),
    }

    print("[7/7] 动态 + 立体声 ...")
    dyn = dynamics(x, sr)
    loud = loudness_approx(x, sr)
    st = stereo_stats(stereo, sch)

    result = {
        "file": "goghost-origin.flac",
        "duration_s": round(dur, 2),
        "bpm": bpm,
        "beat_conf": round(float(conf), 3),
        "beats_per_bar": bars_per_beat,
        "swing_ratio": swing,
        "n_beats": len(beats),
        "key": key,
        "key_top3": key_top3,
        "key_conf": round(key_top3[0][1] - key_top3[1][1], 2),
        "chords": chords,
        "segments": segs,
        "timbre": timbre,
        "dynamics": dyn,
        "loudness_approx_db": loud,
        "stereo": st,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"JSON -> {OUT_JSON}")
    write_report(result, dur)
    print("报告 ->", OUT_MD)

def write_report(r, dur):
    lines = []
    lines.append(f"# GO GHOST (King Gnu) — 多层面音乐分析报告\n")
    lines.append(f"- 文件: `goghost-origin.flac` | 时长 **{dur:.1f}s** | 44.1kHz / 24bit / 立体声")
    lines.append(f"- 分析方式: numpy 手写 DSP（谱通量 / chromagram / KS 模板 / novelty 分段）\n")

    lines.append("## 1. 节拍与节奏\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| BPM | **{r['bpm']}** |")
    lines.append(f"| 节拍置信度 | {r['beat_conf']} |")
    lines.append(f"| 拍号 | {r['beats_per_bar']}/4 |")
    lines.append(f"| Swing/Shuffle 比 | {r['swing_ratio']}（>1.3 有较强摇摆感） |")
    lines.append(f"| 总拍数 | {r['n_beats']} |\n")

    lines.append("## 2. 调性\n")
    lines.append(f"- 检测调性: **{r['key']}**（Krumhansl-Schmuckler，置信 {r['key_conf']}）\n")

    lines.append("## 3. 和弦进行（逐小节, 三和弦模板）\n")
    lines.append("| 小节 | 时间(s) | 和弦 | 匹配度 |")
    lines.append("|---|---|---|---|")
    for c in r["chords"][:64]:
        lines.append(f"| {c['bar']} | {c['t']} | {c['chord']} | {c['corr']} |")
    lines.append("\n")

    lines.append("## 4. 结构分段\n")
    lines.append("| 时间(s) | 时长(s) | 能量(dB) | 频谱质心(Hz) | 段类型 |")
    lines.append("|---|---|---|---|---|")
    for s in r["segments"]:
        lines.append(f"| {s['t0']}–{s['t1']} | {s['dur']} | {s['rms_db']} | {s['centroid_hz']} | {s['label']} |")
    lines.append("\n")

    lines.append("## 5. 音色画像\n")
    t = r["timbre"]
    lines.append("| 特征 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 频谱质心 | {t['centroid_hz']} Hz |")
    lines.append(f"| 85% rolloff | {t['rolloff85_hz']} Hz |")
    lines.append(f"| 频谱平坦度 | {t['flatness']} |")
    lines.append(f"| 过零率 | {t['zcr_per_sec']} 次/秒 |")
    lines.append("| 频段能量比 | " + "  |  ".join(f"{k}: {v} dB" for k, v in t["bands_db"].items()) + " |")
    lines.append("\n")

    lines.append("## 6. 动态与响度\n")
    d = r["dynamics"]
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    for k, v in d.items():
        lines.append(f"| {k} | {v} |")
    lines.append(f"| 近似平均响度 | {r['loudness_approx_db']} dB |")
    lines.append(f"| RMS 分布 P5/P25/P50/P75/P95 | {d['percentile_db']} |")
    lines.append("\n")

    lines.append("## 7. 立体声\n")
    lines.append(f"- 左右声道相关: **{r['stereo']['lr_corr']}**（1=完全单声道, 0=全反相）")
    lines.append(f"- 侧边能量相对中间: **{r['stereo']['side_vs_mid_db']} dB**\n")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
