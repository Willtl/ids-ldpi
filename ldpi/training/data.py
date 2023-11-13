import os
from typing import List, Optional
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from transform import random_crop_resize, dropout_augmentation

# Type aliases for clarity (compatible with Python 3.8)
ArrayFloat = np.ndarray
ArrayInt = np.ndarray
DataFrame = pd.DataFrame
DataLoaderType = DataLoader


class CustomDataset(Dataset):
    """
        Custom dataset class for handling samples and targets.
    """

    def __init__(self, samples: ArrayFloat, targets: ArrayInt, bin_targets: ArrayInt, apply_transform: bool = False):
        self.samples = samples
        self.targets = targets
        self.bin_targets = bin_targets
        self.n_samples = samples.shape[0]
        self.apply_transform = apply_transform

    def __getitem__(self, index: int) -> Tuple[ArrayFloat, int, int]:
        aug = self.samples[index]
        if self.apply_transform:
            aug = random_crop_resize(aug)
            aug = dropout_augmentation(aug, dropout_rate=0.01)
        return aug, self.targets[index], self.bin_targets[index]

    def __len__(self) -> int:
        return self.n_samples


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

            folder_data = folder_data.astype(np.float32) / 255.0
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
        anomaly = np.concatenate(malware_data).astype(np.float32)
        anomaly_targets = np.concatenate(malware_labels).astype(np.int)
    else:
        anomaly = np.array([]).astype(np.float32)
        anomaly_targets = np.array([]).astype(np.int)

    normal = np.concatenate(benign_data).astype(np.float32)
    normal_targets = np.concatenate(benign_labels).astype(np.int)

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

    train_samples = np.array(train['sample'].tolist()).astype(np.float32)
    train_targets = np.array(train['target'].tolist()).astype(np.int)
    train_bin_targets = np.array(train['bin_target'].tolist()).astype(np.int)

    test_samples = np.array(test['sample'].tolist()).astype(np.float32)
    test_targets = np.array(test['target'].tolist()).astype(np.int)
    test_bin_targets = np.array(test['bin_target'].tolist()).astype(np.int)

    return train_samples, train_targets, train_bin_targets, test_samples, test_targets, test_bin_targets


def get_training_dataloader(dataset: str, batch_size: int = 64) -> Tuple[ArrayFloat, ArrayInt, DataLoaderType, DataLoaderType]:
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

    train_ds = CustomDataset(train_samples, train_targets, train_bin_targets, apply_transform=True)
    test_ds = CustomDataset(test_samples, test_targets, test_bin_targets, apply_transform=False)

    weights = make_weights_for_balanced_classes(train_bin_targets)
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=4)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    return train_loader, test_loader


def get_pretrain_dataloader(dataset: str, batch_size) -> DataLoaderType:
    """
        Prepare DataLoader for pretraining with normal samples only.

    Args:
        dataset (Callable): A function to load and return dataset.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 64.

    Returns:
        DataLoader for pretraining.
    """
    train_samples, train_targets, train_bin_targets, _, _, _ = load_data(dataset, test_size=0.0, only_normal=True)
    print(f'Pretraining with {train_samples.shape[0]} normal samples')

    train_ds = CustomDataset(train_samples, train_targets, train_bin_targets)
    pretrain_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    return pretrain_loader
