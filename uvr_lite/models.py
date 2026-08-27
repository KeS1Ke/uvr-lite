"""模型注册表：名称 -> 权重/配置/校验信息。

权重由 scripts/strip_model.py 从原版转换（fp16 + safetensors：体积减半、
加载更快、无 pickle 载入面），托管于本仓库（KeS1Ke/uvr-lite）GitHub
Releases 的 models tag。filename 决定本地文件扩展名（engine 按 `.safetensors`
走 safetensors.torch.load_file，`.ckpt` 走 torch.load weights_only=True）。

镜像注意：mirror_urls 里的文件必须与主源内容一致（同 sha256）才能充当
回退；本项目未发布同内容镜像，故列表为空——下载器对单源内建多段并发 +
重试，主源不可达时失败信息会如实上报。
"""

MODEL_REGISTRY: dict[str, dict] = {
    "bs_roformer_ep317": {
        "model_type": "bs_roformer",
        "config": "model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "filename": "bs_roformer_ep317.lite.safetensors",
        "ckpt_url": (
            "https://github.com/KeS1Ke/uvr-lite/releases/download/"
            "models/bs_roformer_ep317.lite.safetensors"
        ),
        "mirror_urls": [],
        # fp16 safetensors 版 SHA（由 strip_model.py 转换生成）；旧格式本地
        # 缓存校验不匹配会被自动删除重下（一次性迁移机制）
        "sha256": "97307b43fa9a830a80e7d382fab39a746b2a0777ba38cf4d1c7432ef77ad19a2",
        "description": ("BS-RoFormer ep317（viperx 训练，fp16 safetensors 版）："
                        "人声/伴奏分离主力模型，SDR ≈ 10.9-12.9 dB"),
    },
    "mel_band_karaoke": {
        "model_type": "mel_band_roformer",
        "config": "mel_band_roformer_karaoke_aufr33_viperx_config.yaml",
        "filename": "mel_band_roformer_karaoke_aufr33_viperx.lite.safetensors",
        "ckpt_url": (
            "https://github.com/KeS1Ke/uvr-lite/releases/download/"
            "models/mel_band_roformer_karaoke_aufr33_viperx.lite.safetensors"
        ),
        "mirror_urls": [],
        "sha256": "fbf51a0baf307334c93f7162e8515f32cad2bd453f3f82d0a1eaf0f02424e60b",
        "description": ("Mel-Band RoFormer Karaoke（aufr33 & viperx 训练，"
                        "fp16 safetensors 版）：备选模型"),
    },
}

DEFAULT_MODEL = "bs_roformer_ep317"


def get_model_info(name: str) -> dict:
    if name not in MODEL_REGISTRY:
        known = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"未知模型: {name}（可用: {known}）")
    return MODEL_REGISTRY[name]
