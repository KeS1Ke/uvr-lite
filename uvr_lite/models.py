# coding: utf-8
"""模型注册表：名称 -> 权重/配置/校验信息。

权重均托管于 TRvlvr/model_repo（UVR 项目官方模型仓库，GitHub Releases），
与 Ultimate Vocal Remover 使用的模型同源（viperx / aufr33 社区训练，MIT 许可使用须署名）。
"""

from typing import Dict

MODEL_REGISTRY: Dict[str, Dict] = {
    "bs_roformer_ep317": {
        "model_type": "bs_roformer",
        "config": "model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "ckpt_url": (
            "https://github.com/TRvlvr/model_repo/releases/download/"
            "all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        ),
        "mirror_urls": [
            "https://huggingface.co/Eddycrack864/Music-Source-Separation-Training/resolve/"
            "main/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "https://huggingface.co/KitsuneX07/Music_Source_Sepetration_Models/resolve/"
            "main/vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        ],
        "sha256": "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
        "description": "BS-RoFormer ep317（viperx 训练）：人声/伴奏分离主力模型，SDR ≈ 10.9-12.9 dB",
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


def get_model_info(name: str) -> Dict:
    if name not in MODEL_REGISTRY:
        known = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"未知模型: {name}（可用: {known}）")
    return MODEL_REGISTRY[name]
