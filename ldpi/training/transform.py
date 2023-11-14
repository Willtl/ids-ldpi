from typing import Tuple

import numpy as np
from scipy.ndimage import zoom


def random_crop_resize(serie, scale=(0.1, 1.0)):
    seq_len = serie.shape[0]
    lambd = np.random.uniform(scale[0], scale[1])
    win_len = int(round(seq_len * lambd))

    if win_len == seq_len:
        return serie

    start = np.random.randint(0, seq_len - win_len)
    cropped_serie = serie[start: start + win_len]

    # Resizing to original length
    resized_serie = zoom(cropped_serie, (seq_len / cropped_serie.shape[0],))

    return resized_serie


def jitter_augmentation(series, noise_level: float = 0.05):
    """
    Apply jitter augmentation to a time series by adding random noise.

    Args:
        series (Series): The input time series.
        noise_level (float): The standard deviation of the Gaussian noise added to the series.

    Returns:
        Series: The augmented time series.
    """
    jitter = np.random.normal(0, noise_level, series.shape)
    return series + jitter


def time_warp(series, warp_factor: Tuple[float, float] = (0.9, 1.1)):
    """
    Apply time warping augmentation to a time series.

    Args:
        series (Series): The input time series.
        warp_factor (Tuple[float, float]): A tuple indicating the minimum and maximum
                                           scaling factors for time warping.

    Returns:
        Series: The time-warped series.
    """
    seq_len = series.shape[0]
    warp_scale = np.random.uniform(warp_factor[0], warp_factor[1])
    warped_len = int(round(seq_len * warp_scale))

    # Handling edge cases
    if warped_len > seq_len:
        warped_series = zoom(series, (warped_len / seq_len,))
        # Cropping to original length
        start = np.random.randint(0, warped_len - seq_len)
        return warped_series[start: start + seq_len]
    elif warped_len < seq_len:
        warped_series = zoom(series, (warped_len / seq_len,))
        # Padding to original length
        pad_width = seq_len - warped_len
        return np.pad(warped_series, (0, pad_width), 'constant')
    else:
        return series


def dropout_augmentation(sequence, dropout_rate=0.1):
    """
    Apply dropout augmentation to a sequence.

    Args:
        sequence (numpy.ndarray): The input sequence.
        dropout_rate (float): The probability of setting a value in the sequence to zero.

    Returns:
        numpy.ndarray: The augmented sequence.
    """
    # Create a dropout mask where some elements are set to zero
    dropout_mask = np.random.rand(*sequence.shape) > dropout_rate
    augmented_sequence = sequence * dropout_mask
    return augmented_sequence


