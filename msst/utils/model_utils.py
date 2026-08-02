# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Union, Optional
import torch.distributed as dist
def bigshifts_wrapper(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: torch.Tensor,
    device: torch.device,
    model_type: str,
    pbar: bool = False,
    bigshifts: int = 1,
    progress_cb=None,
    demix_progress_cb=None
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """BigShifts wrapper for inference-time demixing.

    progress_cb(done, total): called once per BigShifts pass (1-based done).
    demix_progress_cb(done, total): forwarded to demix for chunk-level progress.
    Both may raise to abort.
    """

    should_print = not dist.is_initialized() or dist.get_rank() == 0

    if bigshifts <= 0:
        bigshifts = 1

    if isinstance(mix, torch.Tensor):
        mix = mix.detach().cpu().numpy()

    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]
    results = []

    if pbar and should_print:
        shifts_iterator = tqdm(shifts, desc="BigShifts passes...", leave=False)
    else:
        shifts_iterator = shifts

    for pass_idx, shift in enumerate(shifts_iterator):
        shifted_mix = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis=-1)
        sources = demix(config, model, shifted_mix, device, model_type, pbar,
                        progress_cb=demix_progress_cb)

        if isinstance(sources, dict):
            unshifted = {
                k: np.concatenate((v[..., shift:], v[..., :shift]), axis=-1)
                for k, v in sources.items()
            }
            results.append(unshifted)
        elif isinstance(sources, np.ndarray):
            unshifted = np.concatenate((sources[..., shift:], sources[..., :shift]), axis=-1)
            results.append(unshifted)
        else:
            raise ValueError("Unsupported return type from demix")

        if progress_cb is not None:
            progress_cb(pass_idx + 1, bigshifts)

    if isinstance(results[0], dict):
        avg_result = {}
        for k in results[0]:
            avg_result[k] = np.mean([r[k] for r in results], axis=0)
        return avg_result
    return np.mean(results, axis=0)


def demix(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: torch.Tensor,
    device: torch.device,
    model_type: str,
    pbar: bool = False,
    progress_cb=None
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Perform audio source separation with a given model.

    Supports both Demucs-specific and generic processing modes, including
    overlapping chunk-based inference with optional progress bar display.
    Handles padding, fading, and batching to reduce artifacts during separation.

    Args:
        config (ConfigDict): Configuration object with audio and inference
            parameters (chunk size, overlap, batch size, etc.).
        model (torch.nn.Module): Source separation model for inference.
        mix (torch.Tensor): Input audio tensor of shape (channels, time).
        device (torch.device): Device on which to run inference (CPU or CUDA).
        model_type (str): Type of model (e.g., 'htdemucs', 'mdx23c') that
            determines processing mode.
        pbar (bool, optional): If True, show a progress bar during chunk
            processing. Defaults to False.
        progress_cb (callable, optional): Called as progress_cb(done, total)
            after each processed chunk; may raise to abort inference.

    Returns:
        Union[Dict[str, np.ndarray], np.ndarray]:
            - Dictionary mapping instrument names to separated waveforms if
              multiple instruments are predicted.
            - NumPy array of separated audio if only a single instrument is
              present (Demucs mode).
    """

    should_print = not dist.is_initialized() or dist.get_rank() == 0

    mix = torch.tensor(mix, dtype=torch.float32)

    if model_type == 'htdemucs':
        mode = 'demucs'
    else:
        mode = 'generic'
    # Define processing parameters based on the mode
    if mode == 'demucs':
        chunk_size = config.training.samplerate * config.training.segment
        num_instruments = len(config.training.instruments)
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        if 'chunk_size' in config.inference:
            chunk_size = config.inference.chunk_size
        else:
            chunk_size = config.audio.chunk_size
        num_instruments = len(prefer_target_instrument(config))
        num_overlap = config.inference.num_overlap

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = _getWindowingArray(chunk_size, fade_size)
        # Add padding for generic mode to handle edge artifacts
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size

    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # Initialize result and counter tensors
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []
            if pbar and should_print:
                progress_bar = tqdm(
                    total=mix.shape[1], desc="Processing audio chunks", leave=False
                )
            else:
                progress_bar = None

            while i < mix.shape[1]:
                # Extract chunk and apply padding if necessary
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                if mode == "generic" and chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                # Process batch if it's full or the end is reached
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    if mode == "generic":
                        window = windowing_array.clone() # using clone() fixes the clicks at chunk edges when using batch_size=1
                        if i - step == 0:  # First audio chunk, no fadein
                            window[:fade_size] = 1
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window[-fade_size:] = 1

                    for j, (start, seg_len) in enumerate(batch_locations):
                        if mode == "generic":
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                            counter[..., start:start + seg_len] += window[..., :seg_len]
                        else:
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu()
                            counter[..., start:start + seg_len] += 1.0

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

                if progress_cb is not None:
                    progress_cb(min(i, mix.shape[1]), mix.shape[1])

            if progress_bar:
                progress_bar.close()

            # Compute final estimated sources
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            # Remove padding for generic mode
            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    # Return the result as a dictionary or a single array
    if mode == "demucs":
        instruments = config.training.instruments
    else:
        instruments = prefer_target_instrument(config)

    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}

    if mode == "demucs" and num_instruments <= 1:
        return estimated_sources
    else:
        return ret_data


def apply_tta(
    config,
    model: torch.nn.Module,
    mix: torch.Tensor,
    waveforms_orig: Union[dict[str, np.ndarray], np.ndarray],
    device: torch.device,
    model_type: str,
    bigshifts: int = 1,
    pbar: bool = False,
    progress_cb=None
) -> Union[dict[str, np.ndarray], np.ndarray]:
    """
    Enhance source separation results using Test-Time Augmentation (TTA).

    Applies augmentations such as channel reversal and polarity inversion to
    the input mixture, reprocesses with the model, and combines the results
    with the original predictions by averaging.

    Args:
        config: Configuration object with model and inference parameters.
        model (torch.nn.Module): Trained source separation model.
        mix (torch.Tensor): Input mixture tensor of shape (channels, time).
        waveforms_orig (Dict[str, torch.Tensor]): Dictionary of separated
            sources before augmentation.
        device (torch.device): Computation device (CPU or CUDA).
        model_type (str): Model type identifier used for demixing.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of separated sources after applying TTA.
    """

    # Create augmentations: channel inversion and polarity inversion
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]

    # Process each augmented mixture
    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = bigshifts_wrapper(
            config,
            model,
            augmented_mix,
            device,
            model_type=model_type,
            bigshifts=bigshifts,
            pbar=pbar
        )
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]

        if progress_cb is not None:
            progress_cb(i + 1, len(track_proc_list))

    # Average the results across augmentations
    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1

    return waveforms_orig


def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate a windowing array with a linear fade-in at the beginning and a fade-out at the end.

    This function creates a window of size `window_size` where the first `fade_size` elements
    linearly increase from 0 to 1 (fade-in) and the last `fade_size` elements linearly decrease
    from 1 to 0 (fade-out). The middle part of the window is filled with ones.

    Parameters:
    ----------
    window_size : int
        The total size of the window.
    fade_size : int
        The size of the fade-in and fade-out regions.

    Returns:
    -------
    torch.Tensor
        A tensor of shape (window_size,) containing the generated windowing array.

    Example:
    -------
    If `window_size=10` and `fade_size=3`, the output will be:
    tensor([0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0.0000])
    """

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def prefer_target_instrument(config: ConfigDict) -> List[str]:
    """
        Return the list of target instruments based on the configuration.
        If a specific target instrument is specified in the configuration,
        it returns a list with that instrument. Otherwise, it returns the list of instruments.

        Parameters:
        ----------
        config : ConfigDict
            Configuration object containing the list of instruments or the target instrument.

        Returns:
        -------
        List[str]
            A list of target instruments.
        """
    if getattr(config.training, 'target_instrument', None):
        return [config.training.target_instrument]
    else:
        return config.training.instruments


def load_not_compatible_weights(model: torch.nn.Module, old_model: dict, verbose: bool = False) -> None:
    """
    Load a possibly incompatible state dict into `model` with best-effort matching.

    Accepts either a raw state_dict or a checkpoint dict with weights under "state" or "state_dict".
    For each param/buffer in `model`: if the name exists and shapes match → copy;
    if ndim matches but shapes differ → zero-pad/crop the source to fit the target;
    if the name is missing or ndim differs → skip. Optional logging on rank 0 when `verbose=True`.

    Args:
        model: Target PyTorch module.
        old_model: Source weights (state_dict or checkpoint dict).
        verbose: Print brief load decisions.

    Returns:
        None
    """

    should_print = verbose and (not dist.is_initialized() or dist.get_rank() == 0)

    new_model = model.state_dict()

    if 'state' in old_model:
        # Fix for htdemucs weights loading
        old_model = old_model['state']
    if 'state_dict' in old_model:
        # Fix for apollo weights loading
        old_model = old_model['state_dict']
    if 'model_state_dict' in old_model:
        # Fix for full_check_point
        old_model = old_model['model_state_dict']

    for el in new_model:
        if el in old_model:
            if should_print:
                print(f'Match found for {el}!')
            if new_model[el].shape == old_model[el].shape:
                if should_print:
                    print('Action: Just copy weights!')
                new_model[el] = old_model[el]
            else:
                if len(new_model[el].shape) != len(old_model[el].shape) and should_print:
                    print('Action: Different dimension! Too lazy to write the code... Skip it')
                else:
                    if should_print:
                        print(f'Shape is different: {tuple(new_model[el].shape)} != {tuple(old_model[el].shape)}')
                    ln = len(new_model[el].shape)
                    max_shape = []
                    slices_old = []
                    slices_new = []
                    for i in range(ln):
                        max_shape.append(max(new_model[el].shape[i], old_model[el].shape[i]))
                        slices_old.append(slice(0, old_model[el].shape[i]))
                        slices_new.append(slice(0, new_model[el].shape[i]))
                    # print(max_shape)
                    # print(slices_old, slices_new)
                    slices_old = tuple(slices_old)
                    slices_new = tuple(slices_new)
                    max_matrix = np.zeros(max_shape, dtype=np.float32)
                    for i in range(ln):
                        max_matrix[slices_old] = old_model[el].cpu().numpy()
                    max_matrix = torch.from_numpy(max_matrix)
                    new_model[el] = max_matrix[slices_new]
        else:
            if should_print:
                print(f'Match not found for {el}!')
    model.load_state_dict(
        new_model
    )


def load_lora_weights(model: torch.nn.Module, lora_path: str, device: str = 'cpu') -> None:
    """
    Load LoRA weights into a model.
    This function updates the given model with LoRA-specific weights from the specified checkpoint file.
    It does not require the checkpoint to match the model's full state dictionary, as only LoRA layers are updated.

    Parameters:
    ----------
    model : Module
        The PyTorch model into which the LoRA weights will be loaded.
    lora_path : str
        Path to the LoRA checkpoint file.
    device : str, optional
        The device to load the weights onto, by default 'cpu'. Common values are 'cpu' or 'cuda'.

    Returns:
    -------
    None
        The model is updated in place.
    """
    lora_state_dict = torch.load(lora_path, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)


def load_start_checkpoint(args: argparse.Namespace,
                          model: torch.nn.Module,
                          old_model,
                          type_: str = 'train') -> None:
    """
    Load an initial checkpoint into `model`.

    For `type_ == "train"`, performs a tolerant load using `old_model` (a state dict or a
    checkpoint dict) via `load_not_compatible_weights`, allowing partial shape mismatches.
    For other modes, loads a strict state dict from `args.start_check_point`, with special
    handling for HTDemucs/Apollo checkpoints (keys under "state"/"state_dict"). If
    `args.lora_checkpoint` is set, LoRA weights are applied after the base load.

    Args:
        args: Namespace with at least `start_check_point`, `model_type`, and optionally `lora_checkpoint`.
        model: Target PyTorch module to receive weights.
        old_model: Source weights for tolerant loading in train mode (state dict or checkpoint dict).
        type_: Loading strategy; "train" uses tolerant loading, otherwise strict loading from path.

    Returns:
        None
    """
    should_print = not dist.is_initialized() or dist.get_rank() == 0

    if should_print:
        print(f'Start from checkpoint: {args.start_check_point}')
    if type_ in ['train']:
        if not args.load_only_compatible_weights:
            load_not_compatible_weights(model, old_model, verbose=False)
        else:
            model.load_state_dict(torch.load(args.start_check_point))
    else:
        device = 'cpu'
        if args.model_type in ['htdemucs', 'apollo']:
            old_model = torch.load(args.start_check_point, map_location=device, weights_only=False)
            # Fix for htdemucs pretrained models
            if 'state' in old_model:
                old_model = old_model['state']
            # Fix for apollo pretrained models
            if 'state_dict' in old_model:
                old_model = old_model['state_dict']
        else:
            if 'state' in old_model:
                # Fix for htdemucs weights loading
                old_model = old_model['state']
            if 'state_dict' in old_model:
                # Fix for apollo weights loading
                old_model = old_model['state_dict']
            if 'model_state_dict' in old_model:
                # Fix for full_check_point
                old_model = old_model['model_state_dict']
        model.load_state_dict(old_model)

    if args.lora_checkpoint_loralib:
        if should_print:
            print(f"Loading LoRA weights from: {args.lora_checkpoint_loralib}")
        load_lora_weights(model, args.lora_checkpoint_loralib)


