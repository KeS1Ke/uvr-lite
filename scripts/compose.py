# -*- coding: utf-8 -*-
"""
合成引擎: 基于 goghost-origin.flac 分析画像, 生成 ~3 分钟 J-rock 风格纯器乐新作
- 150 BPM / 4/4 / 八分直拍驱动 (对齐原曲画像)
- D 小调 (F 大调关系小调, 和声小调 C# 属和弦色彩) —— 全新建材
- 段落能量架构模仿原曲: intro -> build -> verse -> pre -> chorus -> bridge -> final -> outro
- 合成: 正弦扫频底鼓 / 噪声军鼓 / hi-hat / 失真锯齿贝斯 / 失真锯齿"吉他" / pad / lead
- 混音: 分轨增益 + 平移 + FFT 卷积混响 + 附点八分延迟 + 软削波母带 + 结尾淡出
输出 tmp/new.wav + tmp/composition.json
"""
import json
import os
import wave
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 44100
BPM = 150.0
BEAT = 60.0 / BPM          # 0.4 s
BAR = BEAT * 4             # 1.6 s
EIGHTH = BEAT / 2
DOTTED8 = EIGHTH * 1.5     # 300ms 附点八分延迟
SEED = 2026
rng = np.random.default_rng(SEED)

N_BARS = 112
N_SAMPLES = int(N_BARS * BAR * SR)  # 179.2s

# ---------------- 音高与调性 ----------------
def nf(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)

# D 自然小调 (D4=62); A7 段用和声小调 C#
D_MINOR = [62, 64, 65, 67, 69, 70, 72]
D_HARM = [62, 64, 65, 67, 69, 70, 73]
CHORD_VOICE = {  # 根音 midi + 和弦音程(相对根音)
    "Dm": (50, [0, 3, 7]), "Bb": (46, [0, 4, 7]), "F": (41, [0, 4, 7]),
    "C": (48, [0, 4, 7]), "Gm": (43, [0, 3, 7]), "A": (45, [0, 4, 7]),
    "A7": (45, [0, 4, 7, 10]),
}
def chord_midis(name, oct=1):
    root, iv = CHORD_VOICE[name]
    return [root + 12 * oct + i for i in iv]

# ---------------- 基础合成 ----------------
def t_axis(dur):
    return np.arange(int(dur * SR), dtype=np.float64) / SR

def env_adsr(n, a, d, s, r):
    e = np.ones(n)
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    if na > 0:
        e[:na] = np.linspace(0, 1, na)
    if nd > 0:
        e[na:na + nd] = np.linspace(1, s, nd)
    if nr > 0:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e

def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))

def sine(f, dur):
    return np.sin(2 * np.pi * f * t_axis(dur))

def saw(f, dur, harm=10):
    tt = t_axis(dur)
    y = np.zeros(len(tt))
    for k in range(1, harm + 1):
        y += np.sin(2 * np.pi * f * k * tt) / k
    return y / (1 + np.log(harm))

def fft_filter(x, lo, hi):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / SR)
    m = (fr >= lo) & (fr <= hi)
    X[~m] *= 0.01
    return np.fft.irfft(X, len(x))

def soft_clip(x, drive=1.0):
    return np.tanh(drive * x) / np.tanh(drive)

def fft_conv(x, ir):
    n = len(x) + len(ir) - 1
    X = np.fft.rfft(x, n)
    H = np.fft.rfft(ir, n)
    return np.fft.irfft(X * H, n)[:len(x)]

# ---------------- 鼓组合成 ----------------
def kick():
    dur = 0.22
    tt = t_axis(dur)
    f = 150 * np.exp(-tt / 0.05) + 40
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(len(tt), 0.09)
    click = fft_filter(rng.standard_normal(len(tt)), 800, 6000) * env_exp(len(tt), 0.004) * 0.8
    return body + click

def snare():
    dur = 0.25
    tt = t_axis(dur)
    nz = rng.standard_normal(len(tt))
    body = fft_filter(nz, 400, 8000) * env_exp(len(tt), 0.06)
    tone = sine(190, dur) * env_exp(len(tt), 0.09) * 0.6
    return body + tone

def hat(open_=False):
    dur = 0.28 if open_ else 0.06
    tt = t_axis(dur)
    nz = fft_filter(rng.standard_normal(len(tt)), 6500, 18000)
    y = nz * env_exp(len(tt), 0.035 if open_ else 0.012)
    if open_:
        y *= 0.5 + 0.5 * np.sin(np.pi * np.linspace(0, 1, len(tt)))
    return y * 0.6

# ---------------- 旋律/和声乐器 ----------------
def bass_note(f, dur, drive=2.2):
    n = int(dur * SR)
    y = saw(f, dur, harm=8) * env_adsr(n, 0.004, 0.08, 0.85, 0.06)
    return fft_filter(soft_clip(y, drive) * 0.8, 40, 900)

def guitar_chord(freqs, dur, drive=3.5):
    n = int(dur * SR)
    y = np.zeros(n)
    for f in freqs:
        y += saw(f, dur, harm=6)
    y /= len(freqs)
    y = fft_filter(soft_clip(y, drive), 120, 7000)
    return y * env_adsr(n, 0.003, 0.05, 0.8, 0.08)

def pad_chord(freqs, dur):
    n = int(dur * SR)
    y = np.zeros(n)
    for f in freqs:
        for det in (-0.15, 0.15):
            y += saw(f * 2 ** (det / 1200), dur, harm=4)
    y /= (len(freqs) * 2)
    y = fft_filter(y, 150, 4000)
    return y * env_adsr(n, 0.25, 0.3, 0.9, 0.5)

def lead_note(f, dur):
    n = int(dur * SR)
    y = 0.5 * saw(f, dur, harm=6) + 0.5 * saw(f * 2 ** (0.03 / 12), dur, harm=6)
    y = soft_clip(y, 1.6) * 0.9
    y = fft_filter(y, 250, 9500)
    return y * env_adsr(n, 0.012, 0.1, 0.85, 0.15)

# ---------------- 轨道 ----------------
class Track:
    def __init__(self, stereo=False):
        self.stereo = stereo
        self.events = []
        self.buf = np.zeros(N_SAMPLES, dtype=np.float64)
        if stereo:
            self.L = np.zeros(N_SAMPLES, dtype=np.float64)
            self.R = np.zeros(N_SAMPLES, dtype=np.float64)

    def add(self, start, synth, vel=1.0, pan=0.0):
        self.events.append((float(start), synth, float(vel), float(pan)))

    def render(self):
        for start, synth, vel, pan in self.events:
            s = synth() * vel
            i0 = int(start * SR)
            i1 = min(N_SAMPLES, i0 + len(s))
            if i1 <= i0:
                continue
            seg = s[:i1 - i0]
            if self.stereo:
                gl = np.sqrt((1 - pan) / 2 + 0.5) * 1.414
                gr = np.sqrt((1 + pan) / 2 + 0.5) * 1.414
                self.L[i0:i1] += seg * gl
                self.R[i0:i1] += seg * gr
            else:
                self.buf[i0:i1] += seg

# ---------------- 乐句生成 ----------------
def snap_scale(m, scale):
    """把音高吸附到最近的音阶音 (支持跨八度)"""
    pcs = {s % 12 for s in scale}
    for d in range(13):
        if (m + d) % 12 in pcs:
            return m + d
        if (m - d) % 12 in pcs:
            return m - d
    return m

def fit_to_range(m, lo, hi, mid):
    """把目标音移到 [lo, hi] 内最靠近 mid 的八度位置"""
    while m < lo:
        m += 12
    while m > hi:
        m -= 12
    for cand in (m - 12, m + 12):
        if lo <= cand <= hi and abs(cand - mid) < abs(m - mid):
            m = cand
    return m

def gen_melody(bar_start, n_bars, scale, density, phrase_len=4, rng_local=None,
               target_roots=None, base_midi=62, range_span=14, rest_p=0.18):
    """在 [bar_start, bar_start+n_bars] 内生成 lead 音符: (start, midi, dur, vel)"""
    if rng_local is None:
        rng_local = rng
    notes = []
    lo, hi = base_midi, base_midi + range_span
    mid = (lo + hi) // 2
    n_ph = n_bars // phrase_len
    for p in range(n_ph):
        raw = target_roots[p % len(target_roots)] if target_roots else scale[1]
        tgt = fit_to_range(int(raw), lo, hi, mid)
        npos = int(phrase_len * BAR / EIGHTH)  # 8分音符槽位
        occupied = set()
        for slot in range(npos):
            if rng_local.random() < rest_p:
                continue
            if rng_local.random() > density:
                continue
            if slot in occupied or ((slot - 1) in occupied and rng_local.random() < 0.7):
                continue
            occupied.add(slot)
            t0 = bar_start + p * phrase_len * BAR + slot * EIGHTH + rng_local.normal(0, 0.004)
            ph_pos = slot / max(1, npos - 1)
            arch = 1 - 2 * abs(ph_pos - 0.45)
            m = mid + int(round(arch * (tgt - mid) * 0.5)) + int(rng_local.choice([-2, -1, -1, 0, 0, 1, 1, 2]))
            if ph_pos > 0.7 and rng_local.random() < 0.5:
                m = tgt + int(rng_local.choice([-2, -1, 0, 0, 1, 2]))
            m = snap_scale(min(max(m, lo - 1), hi + 1), scale)
            dur = EIGHTH if (slot % 2 == 1 and rng_local.random() < 0.3) else BEAT
            vel = 0.7 + 0.3 * rng_local.random()
            notes.append((t0, m, dur, vel))
    notes.sort(key=lambda x: x[0])
    return notes

# ---------------- 段落谱 ----------------
S = []
def add(name, bars, prog, energy, mel=None):
    if len(prog) >= bars:
        prog = prog[:bars]
    else:
        prog = prog * (bars // len(prog))
    S.append({"name": name, "bars": bars, "prog": prog,
              "energy": energy, "mel": mel})

add("intro",   8, ["Dm", "Dm", "Bb", "Bb", "F", "F", "A", "A"], 0)
add("build1",  8, ["Dm", "Bb", "F", "A"] * 2, 1)
add("verse1", 12, ["Dm", "Bb", "Gm", "A"] * 2 + ["F", "C", "Dm", "A7"], 2,
    dict(density=0.5, base=62, span=14))
add("pre1",    8, ["F", "C", "Dm", "A7"] * 2, 2)
add("chorus1",16, (["Dm", "Bb", "F", "C"] + ["Dm", "Bb", "Gm", "A7"]) * 2, 3,
    dict(density=0.75, base=69, span=16))
add("bridge",  8, ["Bb", "Gm", "Dm", "A"] * 2, 1)
add("verse2", 12, ["Dm", "Bb", "Gm", "A"] * 2 + ["F", "C", "Dm", "A7"], 2,
    dict(density=0.55, base=62, span=14))
add("build2",  4, ["F", "C", "Dm", "A7"], 2)
add("final",  24, (["Dm", "Bb", "F", "C"] + ["Dm", "Bb", "Gm", "A7"]) * 3 +
                  ["Dm", "Bb", "F", "C"] + ["F", "C", "Dm", "A7"] + ["Gm", "A7", "Dm", "Dm"], 3,
    dict(density=0.85, base=69, span=16))
add("outro",  12, ["Dm", "Bb", "F", "A"] * 2 + ["Dm", "Dm", "Dm", "Dm"], 1,
    dict(density=0.4, base=62, span=14))

# ---------------- 主流程 ----------------
def main():
    print(f"生成 {N_SAMPLES / SR:.1f}s @ {BPM} BPM, D 小调, {N_BARS} 小节 ...")

    drums = Track(stereo=True)
    bass = Track()
    guitar = Track(stereo=True)
    pad = Track(stereo=True)
    lead = Track()

    # 各段落编配 (energy 分级增益, 还原结构动态)
    t0 = 0.0
    SEC_GAIN = {0: 1.0, 1: 0.5, 2: 0.74, 3: 1.12}
    for sec in S:
        nb = sec["bars"]
        prog = sec["prog"][:nb]
        e = sec["energy"]
        g = SEC_GAIN[e]
        print(f"  [{sec['name']}] {nb} bars @ {t0:.1f}s  energy={e}  gain={g}")
        for b in range(nb):
            tbar = t0 + b * BAR
            ch = prog[b]
            root, _ = CHORD_VOICE[ch]
            # --- 鼓 ---
            if e >= 1:
                if e == 3:
                    kp = [0, 3, 6, 7]        # 高潮: 1, 2&, 4, 4& 切分驱动
                elif e == 2:
                    kp = [0, 3, 6]           # 主歌: 1, 2&, 4
                else:
                    kp = [0, 4]              # build: 1, 3
                for e8 in kp:
                    drums.add(tbar + e8 * EIGHTH, kick, g * (0.95 + 0.05 * rng.random()), 0)
            if e >= 2:
                for e8 in (2, 6):            # 军鼓 2 & 4
                    vel = 0.98 if (b % 4 == 3) else 0.9
                    drums.add(tbar + e8 * EIGHTH, snare, g * vel * (0.9 + 0.1 * rng.random()), 0)
                if b % 4 == 3 and rng.random() < 0.7:   # 段尾 16分军鼓滚奏
                    for k in range(4):
                        drums.add(tbar + 6.5 * EIGHTH + k * EIGHTH / 2, snare,
                                  g * (0.5 + 0.45 * k / 4), 0)
            if e >= 1:
                for e8 in range(8):          # 8分直拍 hi-hat
                    open_ = (e8 == 6 and e >= 3 and rng.random() < 0.5)
                    vel = 0.55 + (0.25 if e8 % 2 == 1 else 0.0)
                    drums.add(tbar + e8 * EIGHTH, lambda o=open_: hat(o),
                              g * vel * (0.85 + 0.15 * rng.random()), 0.12)
            # --- 贝斯 ---
            if e >= 1:
                br = root - 12
                if e == 3:
                    pat = [(0, 0), (1, 12), (2, 0), (3, 0), (4, 0), (5, 12), (6, 0), (7, 7)]
                elif e == 2:
                    pat = [(0, 0), (2, 0), (3, 7), (4, 0), (6, 7), (7, 12)]
                else:
                    pat = [(0, 0)]
                for e8, iv in pat:
                    bass.add(tbar + e8 * EIGHTH, lambda f=nf(br + iv): bass_note(f, 0.24),
                             g * (0.75 if e >= 2 else 0.6), 0)
            # --- 失真"吉他" ---
            if e >= 2:
                ch_m = chord_midis(ch, oct=1)
                if e == 3:
                    for e8 in range(8):      # 8分连续 chug
                        guitar.add(tbar + e8 * EIGHTH, lambda fs=ch_m: guitar_chord(fs, EIGHTH * 0.9),
                                   g * 0.6, 0.3 if e8 % 2 else -0.3)
                else:
                    for e8 in (1, 3, 5, 7):  # 反拍切分 stab
                        guitar.add(tbar + e8 * EIGHTH, lambda fs=ch_m: guitar_chord(fs, EIGHTH * 0.85),
                                   g * 0.5, 0.25 if e8 % 4 == 1 else -0.25)
            # --- pad ---
            if e == 0:
                pad.add(tbar, lambda fs=chord_midis(ch, oct=1): pad_chord(fs, BAR * 0.95), 0.5, 0)
            elif e == 1:
                if b % 2 == 0:
                    pad.add(tbar, lambda fs=chord_midis(ch, oct=1): pad_chord(fs, BAR * 1.9), 0.45, 0.2)
            elif e == 2 and b % 2 == 0:
                pad.add(tbar, lambda fs=chord_midis(ch, oct=1): pad_chord(fs, BAR * 1.9), 0.35, 0.2)
            elif e >= 3:
                pad.add(tbar, lambda fs=chord_midis(ch, oct=1): pad_chord(fs, BAR * 0.95), 0.38, 0.25)
        t0 += nb * BAR

    # --- lead 旋律 (主歌/副歌/桥段/尾声) ---
    lead_t0 = 0.0
    LEAD_GAIN = {1: 0.55, 2: 0.7, 3: 1.0}
    for sec in S:
        if sec["mel"]:
            m = sec["mel"]
            scale = D_HARM if any(c == "A7" for c in sec["prog"]) else D_MINOR
            roots = [CHORD_VOICE[c][0] for c in sec["prog"]]  # 原始根音, 生成时八度适配
            notes = gen_melody(lead_t0, sec["bars"], scale, m["density"], phrase_len=4,
                               target_roots=roots, base_midi=m["base"], range_span=m["span"])
            lg = LEAD_GAIN.get(sec["energy"], 0.7)
            for start, midi, dur, vel in notes:
                lead.add(start, lambda f=nf(midi), d=dur: lead_note(f, d), vel * 0.75 * lg, 0)
        lead_t0 += sec["bars"] * BAR
    # 尾声长音收束 (最后 6 小节 D4)
    last_start = (N_BARS - 6) * BAR
    lead.add(last_start, lambda: lead_note(nf(62), 6.5 * BAR), 0.55, 0)
    lead.add(last_start + 1 * BAR, lambda: lead_note(nf(69), 6 * BAR), 0.4, 0)

    print("渲染各轨 ...")
    drums.render(); bass.render(); guitar.render(); pad.render(); lead.render()

    # ---------------- 效果: 混响 / 延迟 ----------------
    print("混响 + 延迟 ...")
    ir_rng = np.random.default_rng(7)
    n_ir = int(1.6 * SR)
    ir = ir_rng.standard_normal(n_ir) * np.exp(-np.arange(n_ir) / (0.55 * SR))
    ir = fft_filter(ir, 300, 7000)
    ir /= np.sqrt(np.sum(ir ** 2))

    padL = fft_conv(pad.L, ir) * 0.35 + pad.L
    padR = fft_conv(pad.R, ir) * 0.35 + pad.R
    gL = fft_conv(guitar.L, ir) * 0.12 + guitar.L
    gR = fft_conv(guitar.R, ir) * 0.12 + guitar.R
    sL = fft_conv(drums.L, ir) * 0.10 + drums.L
    sR = fft_conv(drums.R, ir) * 0.10 + drums.R
    bL = fft_conv(bass.buf, ir) * 0.05 + bass.buf
    bR = fft_conv(bass.buf, ir) * 0.05 + bass.buf

    # lead: 附点八分延迟 (300ms) + 混响
    dl = int(DOTTED8 * SR)
    dly = np.zeros(N_SAMPLES)
    dly[dl:] += lead.buf[:-dl] * 0.4
    dly[2 * dl:] += lead.buf[:-2 * dl] * 0.16
    lL = lead.buf + dly + fft_conv(lead.buf, ir) * 0.25
    lR = lead.buf + dly + fft_conv(lead.buf, ir) * 0.25

    # ---------------- 母带 ----------------
    print("母带: 平衡 + 胶合压缩 + 峰值归一化 + 淡出 ...")
    mixL = bL * 1.0 + gL * 0.7 + padL * 0.55 + sL * 0.85 + lL * 0.8
    mixR = bR * 1.0 + gR * 0.7 + padR * 0.55 + sR * 0.85 + lR * 0.8
    mix = np.stack([mixL, mixR], axis=1)
    # 轻胶合: 低驱动软削波, 不做 RMS 归一化 (段落增益已承担动态)
    mix = soft_clip(mix / 1.6, 1.8)
    peak = np.abs(mix).max()
    mix *= 0.891 / peak  # 峰值 -1 dB
    # 结尾 6s 淡出
    fade = np.ones(N_SAMPLES)
    nfade = int(6 * SR)
    fade[-nfade:] = np.linspace(1, 0, nfade)
    mix *= fade[:, None]
    out = np.clip(mix, -1, 1)
    out16 = (out * 32767.0).astype(np.int16)

    with wave.open(os.path.join(BASE, "tmp", "new.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(out16.tobytes())
    print(f"WAV -> tmp/new.wav  ({len(out16) / SR:.1f}s)")

    score = {"bpm": BPM, "key": "D minor (+harmonic-minor A7)", "duration_s": len(out16) / SR,
             "sections": [{"name": s["name"], "bars": s["bars"], "energy": s["energy"],
                           "prog": s["prog"]} for s in S]}
    with open(os.path.join(BASE, "tmp", "composition.json"), "w", encoding="utf-8") as f:
        json.dump(score, f, ensure_ascii=False, indent=1)
    print("score -> tmp/composition.json")

if __name__ == "__main__":
    main()
