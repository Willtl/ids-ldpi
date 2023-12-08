import os
import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, Dataset

# Type aliases for clarity (compatible with Python 3.8)
ArrayFloat = np.ndarray
ArrayInt = np.ndarray
DataFrame = pd.DataFrame
DataLoaderType = DataLoader


class CustomDataset(Dataset):
    """
    Custom dataset class for handling samples and targets.
    """

    def __init__(self, samples, targets, bin_targets):
        self.samples = samples
        self.targets = targets
        self.bin_targets = bin_targets
        self.n_samples = samples.shape[0]

        # Separate samples into normal and anomalies
        self.normal_samples = self.samples[self.bin_targets == 1]
        self.anomaly_samples = self.samples[self.bin_targets == -1]
        self.normal_targets = self.targets[self.bin_targets == 1]
        self.anomaly_targets = self.targets[self.bin_targets == -1]

    def __getitem__(self, index: int):
        # Check if there are any anomalies
        if len(self.anomaly_samples) > 0 and np.random.rand() > 0.5:
            # Return anomaly
            anomaly_index = np.random.randint(len(self.anomaly_samples))
            return self.anomaly_samples[anomaly_index], self.anomaly_targets[anomaly_index], -1
        else:
            # Return normal sample
            normal_index = index % len(self.normal_samples)
            return self.normal_samples[normal_index], self.normal_targets[normal_index], 1

    def __len__(self) -> int:
        return self.n_samples


class OneClassContrastiveDataset(Dataset):
    """
    Custom dataset class for pretraining with one class contrastive.
    """

    def __init__(self, samples: np.ndarray, targets: np.ndarray, bin_targets: np.ndarray):
        self.samples = samples
        self.targets = targets
        self.bin_targets = bin_targets
        self.n_samples = samples.shape[0]

        # Create fake outliers
        self._create_outliers()

    def _create_outliers(self):
        self.out_shuffled = self.shuffle_sequences(self.samples)
        self.out_random_insert = self.random_byte_insertion(self.samples)
        self.out_scaled = self.byte_value_scaling(self.samples)
        self.out_random_delete = self.random_byte_deletion(self.samples)
        self.out_periodic_noised = self.periodic_noise_addition(self.samples)
        self.out_slice = self.slice_swap(self.samples)
        self.all_outliers = [self.out_shuffled, self.out_random_insert, self.out_scaled, self.out_random_delete, self.out_periodic_noised, self.out_slice]

    def shuffle_sequences(self, samples):
        shuffled_samples = np.copy(samples)
        for s in shuffled_samples:
            np.random.shuffle(s)
        return shuffled_samples

    def slice_swap(self, samples):
        swapped_samples = np.copy(samples)
        for sample in swapped_samples:
            # Ensure the slice length is not too large, e.g., between 1/10th and 1/5th of the sample length
            slice_length = random.randint(1, max(1, len(sample) // 10))

            # Choose two different start points for slices
            start1 = random.randint(0, len(sample) - slice_length)
            start2 = random.randint(0, len(sample) - slice_length)

            # Ensuring the second slice doesn't overlap with the first
            while abs(start1 - start2) < slice_length:
                start2 = random.randint(0, len(sample) - slice_length)

            # Swap the slices
            temp = np.copy(sample[start1:start1 + slice_length])
            sample[start1:start1 + slice_length] = sample[start2:start2 + slice_length]
            sample[start2:start2 + slice_length] = temp

        return swapped_samples

    def random_byte_insertion(self, samples, padding_value=256):
        modified_samples = []
        for sample in samples:
            length = len(sample)
            num_insertions = random.randint(1, len(sample) // 10)  # Number of insertions
            for _ in range(num_insertions):
                insert_index = random.randint(0, len(sample) - 1)
                byte_to_insert = np.random.randint(0, 255)
                sample = np.insert(sample, insert_index, byte_to_insert)
            # Pad or trim the sample to max_length
            if len(sample) < length:
                sample = np.pad(sample, (0, length - len(sample)), 'constant', constant_values=(padding_value,))
            else:
                sample = sample[:length]
            modified_samples.append(sample)
        return np.array(modified_samples)

    def byte_value_scaling(self, samples):
        outlier_samples = np.copy(samples)
        for idx in range(len(outlier_samples)):
            sample = outlier_samples[idx]
            scale_factor = random.uniform(0.5, 2)
            sample = np.round(sample * scale_factor)  # Scale and round to nearest integer
            np.clip(sample, 0, 256, out=sample)  # Clipping to keep values in byte range
            outlier_samples[idx] = sample  # Store the modified sample back in the array
        return outlier_samples

    def random_byte_deletion(self, samples, padding_value=256):
        outlier_samples = np.copy(samples)
        for idx in range(len(outlier_samples)):
            sample = outlier_samples[idx]
            original_length = len(sample)
            num_deletions = random.randint(1, len(sample) // 10)  # Number of deletions
            for _ in range(num_deletions):
                delete_index = random.randint(0, len(sample) - 1)
                sample = np.delete(sample, delete_index)
            # Pad the sample back to original length
            padding_needed = original_length - len(sample)
            if padding_needed > 0:
                sample = np.pad(sample, (0, padding_needed), 'constant', constant_values=(padding_value,))
            # Store the modified sample back in the array
            outlier_samples[idx] = sample
        return outlier_samples

    def periodic_noise_addition(self, samples):
        outlier_samples = np.copy(samples)
        for idx in range(len(outlier_samples)):
            sample = outlier_samples[idx]
            frequency = random.uniform(0.1, 0.5) * np.pi
            amplitude = random.uniform(1, len(sample) // 10)
            noise = amplitude * np.sin(np.linspace(0, frequency * len(sample), len(sample)))
            sample = np.round(sample + noise)  # Add noise and round to nearest integer
            np.clip(sample, 0, 256, out=sample)  # Clipping to keep values in byte range
            outlier_samples[idx] = sample  # Store the modified sample back in the array
        return outlier_samples

    def transform(self, sample):
        # List of augmentation functions
        aug_funcs = [self.invert_values, self.reverse_sequences]

        # Shuffle the list of augmentation functions
        random.shuffle(aug_funcs)

        # Apply the first augmentation function
        augmented = aug_funcs[0](sample)

        # Initialize probability for subsequent functions
        probability = 0.5

        # Apply the remaining augmentation functions with decreasing probability
        for aug_func in aug_funcs[1:]:
            if random.random() < probability:
                augmented = aug_func(augmented)
                probability *= 0.5  # Decrease the probability by half

        # Apply crop_and_resize
        augmented = self.crop_and_resize(augmented)

        return augmented

    def crop_and_resize(self, sample):
        original_length = len(sample)
        scale = np.random.uniform(0.5, 1.0)
        crop_size = int(original_length * scale)
        start = np.random.randint(0, original_length - crop_size)
        cropped_sample = sample[start:start + crop_size]

        # Nearest-neighbor interpolation
        if crop_size != original_length:
            interp_function = interp1d(np.linspace(0, crop_size - 1, num=crop_size), cropped_sample, kind='nearest', fill_value='extrapolate')
            resized_sample = interp_function(np.linspace(0, crop_size - 1, num=original_length)).astype(int)
            return resized_sample
        else:
            return cropped_sample

    def reverse_sequences(self, sample):
        reversed_sample = np.flip(sample, axis=0)
        return reversed_sample

    def invert_values(self, sample):
        return 256 - sample

    def __getitem__(self, index: int) -> Tuple:
        if random.random() < 0.5:
            sample = self.samples[index]
        else:
            outlier_choice = random.choice(self.all_outliers)
            sample = outlier_choice[index]
        v1, v2 = self.transform(sample), self.transform(sample)
        v1, v2 = torch.from_numpy(v1), torch.from_numpy(v2)

        return v1, v2

    def __len__(self) -> int:
        return self.n_samples


class BalancedBatchSampler(Sampler):
    """
    Sampler for creating balanced batches from a dataset with two classes.
    Ensures each batch contains an equal number of samples from each class.

    Attributes:
        normal_indices (List[int]): Indices of samples from the normal class.
        anomaly_indices (List[int]): Indices of samples from the anomaly class.
        batch_size (int): Size of the batch.
        num_batches (int): Total number of batches.
        current_batch (int): Index of the current batch.

    Args:
        dataset (Dataset): Dataset from which to draw samples.
        batch_size (int): Size of each batch, should be an even number.
    """

    def __init__(self, dataset: Dataset, batch_size: int):
        # Verify if batch_size is even
        if batch_size % 2 != 0:
            raise ValueError("Batch size for BalancedBatchSampler must be an even number.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.normal_indices = [i for i, (_, target, _) in enumerate(dataset) if target == 1]
        self.anomaly_indices = [i for i, (_, target, _) in enumerate(dataset) if target != 1]

        self.num_batches = min(len(self.normal_indices), len(self.anomaly_indices)) // (batch_size // 2)
        self.current_batch = 0

    def __iter__(self):
        # Randomize the order of indices in each class
        np.random.shuffle(self.normal_indices)
        np.random.shuffle(self.anomaly_indices)

        half_batch = self.batch_size // 2

        # Generate balanced batches
        for _ in range(self.num_batches):
            normal_batch = self.normal_indices[self.current_batch * half_batch:(self.current_batch + 1) * half_batch]
            anomaly_batch = self.anomaly_indices[self.current_batch * half_batch:(self.current_batch + 1) * half_batch]
            self.current_batch += 1

            yield normal_batch + anomaly_batch

    def __len__(self):
        return self.num_batches


def make_weights_for_balanced_classes(targets: ArrayInt) -> List[float]:
    """
        Create weights for balanced class sampling.
    """
    count = {1: 0, -1: 0}
    for item in targets:
        count[item] += 1

    N = float(sum(count.values()))
    weight_per_class = {1: N / float(count[1]) if count[1] > 0 else 0,
                        -1: N / float(count[-1]) if count[-1] > 0 else 0}

    return [weight_per_class[val] for val in targets]


def load_data_from_folder(dataset_name: str, category: str, counter: int) -> Tuple[int, List[ArrayFloat], List[ArrayInt]]:
    """
    Load data from subdirectories in a folder, normalizing and structuring it.

    Args:
        dataset_name (str): Name of the dataset directory.
        category (str): Category to load ('benign' or 'malicious').
        counter (int): Starting label counter.

    Returns:
        Tuple[int, List[ArrayFloat], List[ArrayInt]]: Updated label counter, list of loaded samples, and corresponding targets.
    """
    samples: List[ArrayFloat] = []
    targets: List[ArrayInt] = []

    parent_path = os.path.join('samples', dataset_name, 'pcap', category)

    # Loop through subdirectories
    for traffic_type in os.listdir(parent_path):
        traffic_path = os.path.join(parent_path, traffic_type)

        for sub_traffic_type in os.listdir(traffic_path):
            sub_traffic_path = os.path.join(traffic_path, sub_traffic_type)
            print(sub_traffic_path, counter)

            folder_data: Optional[ArrayFloat] = None
            for file_name in os.listdir(sub_traffic_path):
                if file_name.endswith('.npy'):
                    data = np.load(os.path.join(sub_traffic_path, file_name)).reshape(1, -1)
                    folder_data = data if folder_data is None else np.concatenate((folder_data, data))

            if folder_data is None:
                continue

            # folder_data = folder_data.astype(np.float32) / 255.0
            folder_data = folder_data.astype(np.int32)
            samples.append(folder_data)
            labels = np.ones(folder_data.shape[0]) * counter

            targets.append(labels)
            counter += 1

    return counter, samples, targets


def load_data(dataset: str, test_size: float = 0.20, only_normal: bool = False) -> Tuple[ArrayFloat, ArrayInt, ArrayInt, ArrayFloat, ArrayInt, ArrayInt]:
    """
    Load and split data into training and testing sets.

    Args:
        dataset (str): The dataset name.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.20.
        only_normal (bool, optional): If True, only load normal data. Defaults to False.

    Returns:
        Tuple[ArrayFloat, ArrayInt, ArrayInt, ArrayFloat, ArrayInt, ArrayInt]: Train samples, train targets, train binary targets,
        test samples, test targets, test binary targets.
    """

    # Assuming load_data_from_folder is a function that loads the data
    # Replace with the actual data loading logic as necessary
    label_counter, benign_data, benign_labels = load_data_from_folder(dataset, 'benign', counter=0)

    if not only_normal:
        label_counter, malware_data, malware_labels = load_data_from_folder(dataset, 'malicious', counter=label_counter)
        anomaly = np.concatenate(malware_data).astype(np.int32)
        anomaly_targets = np.concatenate(malware_labels).astype(int)
    else:
        anomaly = np.array([]).astype(np.int32)
        anomaly_targets = np.array([]).astype(int)

    normal = np.concatenate(benign_data).astype(np.int32)
    normal_targets = np.concatenate(benign_labels).astype(int)

    data = np.concatenate((normal, anomaly)) if not only_normal else normal
    targets = np.concatenate((normal_targets, anomaly_targets)) if not only_normal else normal_targets
    bin_targets = np.concatenate((np.ones(normal.shape[0]), -np.ones(anomaly.shape[0]))) if not only_normal else np.ones(normal.shape[0])

    df = pd.DataFrame({'sample': data.tolist(), 'target': targets, 'bin_target': bin_targets})

    if only_normal:
        df = df[df['bin_target'] == 1]

    if test_size > 0:
        train, test = train_test_split(df, test_size=test_size, random_state=42)
    else:
        # When test_size is 0, use all data for training and create empty test sets
        train = df
        test = pd.DataFrame(columns=['sample', 'target', 'bin_target'])

    train_samples = np.array(train['sample'].tolist()).astype(np.int32)
    train_targets = np.array(train['target'].tolist()).astype(int)
    train_bin_targets = np.array(train['bin_target'].tolist()).astype(int)

    test_samples = np.array(test['sample'].tolist()).astype(np.int32)
    test_targets = np.array(test['target'].tolist()).astype(int)
    test_bin_targets = np.array(test['bin_target'].tolist()).astype(int)

    return train_samples, train_targets, train_bin_targets, test_samples, test_targets, test_bin_targets


def get_training_dataloader(dataset: str, batch_size: int = 64) -> Tuple[DataLoaderType, DataLoaderType]:
    """
    Prepare DataLoader for training and testing datasets.

    Args:
        dataset (Callable): A function to load and return dataset.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 64.

    Returns:
        Tuple containing test samples, test targets, training DataLoader, and testing DataLoader.
    """
    train_samples, train_targets, train_bin_targets, test_samples, test_targets, test_bin_targets = load_data(dataset, only_normal=False)
    print(f'{train_samples.shape[0]} training samples')
    print(train_samples.shape, test_samples.shape)

    train_ds = CustomDataset(train_samples, train_targets, train_bin_targets)
    test_ds = CustomDataset(test_samples, test_targets, test_bin_targets)

    print('len train ds', len(train_ds))

    # Create sampler

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, drop_last=True, num_workers=0)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    return train_loader, test_loader


def get_pretrain_dataloader(dataset: str, batch_size: int, contrastive: bool = False, shuffle: bool = True, drop_last: bool = True) -> DataLoaderType:
    """
        Prepare DataLoader for pretraining with normal samples only.

    Args:
        dataset (Callable): A function to load and return dataset.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 64.
        contrastive (bool): Dataloader for contrastive learning.
    Returns:
        DataLoader for pretraining.
    """
    train_samples, train_targets, train_bin_targets, _, _, _ = load_data(dataset, test_size=0.0, only_normal=True)
    print(f'Pretraining with {train_samples.shape[0]} normal samples')

    if contrastive:
        train_ds = OneClassContrastiveDataset(train_samples, train_targets, train_bin_targets)
    else:
        train_ds = CustomDataset(train_samples, train_targets, train_bin_targets)

    pretrain_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=True)

    return pretrain_loader
