import numpy as np
from scipy.ndimage import zoom


def random_crop_resize(serie, scale=(0.5, 1.0)):
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
