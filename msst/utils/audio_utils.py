
import numpy as np
import os
import soundfile as sf
from typing import Dict, Tuple, Optional


def normalize_audio(audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize an audio signal using mean and standard deviation.

    Computes the mean and standard deviation from the mono mix of the input
    signal, then applies normalization to each channel.

    Args:
        audio (np.ndarray): Input audio array of shape (channels, time) or (time,).

    Returns:
        Tuple[np.ndarray, Dict[str, float]]: A tuple containing:
            - Normalized audio with the same shape as the input.
            - A dictionary with keys "mean" and "std" from the original audio.
    """

    mono = audio.mean(0)
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: Dict[str, float]) -> np.ndarray:
    """
    Reverse normalization on an audio signal.

    Applies the stored mean and standard deviation to restore the original
    scale of a previously normalized signal.

    Args:
        audio (np.ndarray): Normalized audio array to be denormalized.
        norm_params (Dict[str, float]): Dictionary containing the keys
            "mean" and "std" used during normalization.

    Returns:
        np.ndarray: Denormalized audio with the same shape as the input.
    """

    return audio * norm_params["std"] + norm_params["mean"]


