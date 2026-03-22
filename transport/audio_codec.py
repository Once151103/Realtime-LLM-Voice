"""
Audio codec — PCM resampling utilities.
"""

import numpy as np


def resample_pcm(
    audio_data: bytes,
    from_rate: int,
    to_rate: int,
    dtype=np.int16,
) -> bytes:
    """
    Simple linear resampling of PCM audio.

    For production, consider using torchaudio.transforms.Resample or
    scipy.signal.resample for better quality.
    """
    if from_rate == to_rate:
        return audio_data

    audio = np.frombuffer(audio_data, dtype=dtype).astype(np.float32)
    ratio = to_rate / from_rate
    new_length = int(len(audio) * ratio)

    # Linear interpolation resampling
    indices = np.linspace(0, len(audio) - 1, new_length)
    resampled = np.interp(indices, np.arange(len(audio)), audio)

    return resampled.astype(dtype).tobytes()


def pcm_to_float32(pcm_data: bytes) -> np.ndarray:
    """Convert PCM 16-bit bytes to float32 numpy array [-1.0, 1.0]."""
    return np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array to PCM 16-bit bytes."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
