"""模型注册表：名称 -> 权重/配置/校验信息。

默认模型权重托管于本仓库（KeS1Ke/uvr-lite）的 GitHub Releases：fp16 瘦身版
（体积减半，加载时自动转回 fp32 推理，输出差异约 -80dB 不可闻；由
scripts/strip_model.py 从原版转换）。原版（TRvlvr/model_repo，UVR 项目官方
模型仓库）与 HuggingFace 镜像保留为回退源，模型同源（viperx / aufr33
社区训练，MIT 许可使用须署名）。
"""


MODEL_REGISTRY: dict[str, dict] = {
    "bs_roformer_ep317": {
        "model_type": "bs_roformer",
        "config": "model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "ckpt_url": (
            "https://github.com/KeS1Ke/uvr-lite/releases/download/"
            "models/bs_roformer_ep317.lite.ckpt"
        ),
        "mirror_urls": [
            "https://github.com/TRvlvr/model_repo/releases/download/"
            "all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "https://huggingface.co/Eddycrack864/Music-Source-Separation-Training/resolve/"
            "main/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "https://huggingface.co/KitsuneX07/Music_Source_Sepetration_Models/resolve/"
            "main/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        ],
        # fp16 瘦身版 SHA（由 strip_model.py 转换生成）；旧版 fp32 本地缓存
        # 校验不匹配会被自动删除重下（一次性迁移，之后磁盘省 320MB）
        "sha256": "2760037bd59b4227829562b93d665b6e223a8d747fb6aed6b534420e2a92e0a6",
        "description": ("BS-RoFormer ep317（viperx 训练，fp16 瘦身版）："
                        "人声/伴奏分离主力模型，SDR ≈ 10.9-12.9 dB"),
    },
    "mel_band_karaoke": {
        "model_type": "mel_band_roformer",
        "config": "mel_band_roformer_karaoke_aufr33_viperx_config.yaml",
        "ckpt_url": (
            "https://github.com/TRvlvr/model_repo/releases/download/"
            "all_public_uvr_models/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
        ),
        "mirror_urls": [
            "https://huggingface.co/shiromiya/audio-separation-models/resolve/"
            "main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        ],
        "sha256": "1de20d459332fe8869aeb01327a31df0032262706e1365114e852dc271779813",
        "description": "Mel-Band RoFormer Karaoke（aufr33 & viperx 训练）：备选模型",
    },
}

DEFAULT_MODEL = "bs_roformer_ep317"


def get_model_info(name: str) -> dict:
    if name not in MODEL_REGISTRY:
        known = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"未知模型: {name}（可用: {known}）")
    return MODEL_REGISTRY[name]
