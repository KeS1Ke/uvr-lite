import os
import yaml
import numpy as np
import torch
import argparse
from typing import Dict, List, Tuple, Union
from ml_collections import ConfigDict
from torch import nn
import soundfile as sf

def parse_args_inference(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for inference configuration.

    Builds the CLI for model selection, configuration path, input/output handling,
    device/runtime options, test-time augmentation, and optional LoRA checkpoints.
    If `dict_args` is provided, its key–value pairs override or supply CLI options
    programmatically; otherwise, arguments are read from `sys.argv`.

    Args:
        dict_args (Union[Dict, None]): Optional mapping of argument names to values
            used to override or supply CLI options programmatically.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all inference
        configuration values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
                             " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", type=str, default="", help="path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="Code will generate spectrograms for resulted stems."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true',
                        help="invert vocals to get instrumental if provided")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    parser.add_argument("--flac_file", action='store_true', help="Output flac file instead of wav")
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24', 'FLOAT'], default='FLOAT',
                        help="PCM type for FLAC files (PCM_16 or PCM_24)")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--bigshifts", type=int, default=1,
                        help="Number of circular time shifts to average during demix. Values <= 0 are treated as 1.")
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--filename_template", type=str, default='{file_name}/{instr}',
                        help="Output filename template, without extension, using '/' for subdirectories. Default: '{file_name}/{instr}'")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")
    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()
    args.pcm_type = validate_sndfile_subtype(args)

    return args


def validate_sndfile_subtype(args):
    codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
    subtype = args.pcm_type
    if subtype in sf.available_subtypes(codec):
        return subtype
    default = sf.default_subtype(codec)
    print(f"WARNING: codec {codec} doesn't support subtype {subtype}, defaulting to {default}")
    return default


def load_config(model_type: str, config_path: str) -> ConfigDict:
    """
    Load a model configuration from a file.

    Based on `model_type`, returns a YAML-parsed ConfigDict
    or a YAML-parsed ConfigDict for other models.

    Args:
        model_type (str): Model identifier that determines the loader behavior
            (e.g., 'htdemucs', 'mdx23c', etc.).
        config_path (str): Path to the configuration file (YAML).

    Returns:
        ConfigDict: Loaded configuration object.

    Raises:
        FileNotFoundError: If `config_path` does not point to an existing file.
        ValueError: If the configuration cannot be parsed or is otherwise invalid.
    """
    try:
        with open(config_path, 'r') as f:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_model_from_config(model_type: str, config_path: str) -> Tuple[nn.Module, ConfigDict]:
    """
    Load and instantiate a model using a configuration file.

    Given a `model_type` and a path to a configuration, this function loads the
    configuration (YAML) and constructs the corresponding model.

    Args:
        model_type (str): Identifier of the model family (e.g., 'mdx23c', 'htdemucs',
            'scnet', 'mel_band_conformer', etc.).
        config_path (str): Filesystem path to the configuration file used to
            initialize the model.

    Returns:
        Tuple[nn.Module, ConfigDict]: A tuple containing the
        initialized PyTorch model and the loaded configuration object.

    Raises:
        ValueError: If `model_type` is unknown or model initialization fails.
        FileNotFoundError: If `config_path` does not exist (may be raised by the
            underlying config loader).
    """

    config = load_config(model_type, config_path)
    if 'model_type' in config.training:
        model_type = config.training.model_type
    if model_type == 'mel_band_roformer':
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'mel_band_conformer':
        from models.bs_roformer import MelBandConformer
        model = MelBandConformer(**dict(config.model))
    elif model_type == 'bs_roformer':
        from models.bs_roformer import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'bs_conformer':
        from models.bs_roformer import BSConformer
        model = BSConformer(**dict(config.model))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


