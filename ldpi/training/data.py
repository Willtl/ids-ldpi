import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, samples, targets, bin_targets):
        self.samples = samples
        self.targets = targets
        self.bin_targets = bin_targets
        self.n_samples = samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.targets[index], self.bin_targets[index]

    def __len__(self):
        return self.n_samples


def make_weights_for_balanced_classes(targets):
    # Only two classes: 1 (benign) and -1 (malicious)
    count = {
        1: 0,  # benign
        -1: 0  # malicious
    }

    for item in targets:
        count[item] += 1

    N = float(sum(count.values()))

    # Compute the weights
    weight_per_class = {
        1: N / float(count[1]) if count[1] > 0 else 0,
        -1: N / float(count[-1]) if count[-1] > 0 else 0
    }

    weight = [0] * len(targets)
    for idx, val in enumerate(targets):
        weight[idx] = weight_per_class[val]

    return weight


def get_loaders(dataset='gaussian'):
    samples, targets, bin_targets, test_samples, test_targets, test_bin_targets = load_data(dataset)
    print(f'{samples.shape[0]} samples')

    train_ds = CustomDataset(samples, targets, bin_targets)
    test_ds = CustomDataset(test_samples, test_targets, test_bin_targets)

    # Compute weights for balanced sampling
    weights = make_weights_for_balanced_classes(bin_targets)  # Assuming two classes: benign (1) and malicious (-1)
    weights = torch.DoubleTensor(weights)  # Convert to DoubleTensor
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # Create the DataLoader with the sampler for the training set
    train_loader = DataLoader(dataset=train_ds, batch_size=64, sampler=sampler, drop_last=True, pin_memory=True, num_workers=1, persistent_workers=True)

    # No sampler for the test set
    test_loader = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False, pin_memory=True)

    return test_samples, test_targets, train_loader, test_loader


def get_pretrain(dataset):
    train_samples, train_targets, train_bin_targets, _, _, _ = load_data(dataset, only_normal=True)
    print(f'Pretraining with {train_samples.shape[0]} samples')

    train_ds = CustomDataset(train_samples, train_targets, train_bin_targets)

    pretrain_loader = DataLoader(dataset=train_ds, batch_size=64, shuffle=True, drop_last=True, pin_memory=True)
    return pretrain_loader


def load_data_from_folder(dataset_name, parent_folder, samples, targets, counter):
    parent_path = f'samples/{dataset_name}/{parent_folder}/'

    # Loop through the folders e.g., audio, video, etc.
    for traffic_type in os.listdir(parent_path):
        traffic_path = os.path.join(parent_path, traffic_type)

        # Dive one layer deeper e.g., audio/audio, video/http, etc.
        for sub_traffic_type in os.listdir(traffic_path):
            sub_traffic_path = os.path.join(traffic_path, sub_traffic_type)

            folder_data: np.ndarray = None
            for i in os.listdir(sub_traffic_path):
                if i.endswith('.npy'):
                    if folder_data is None:
                        folder_data = np.load(os.path.join(sub_traffic_path, i)).reshape(1, -1)
                    else:
                        data = np.load(os.path.join(sub_traffic_path, i)).reshape(1, -1)
                        folder_data = np.concatenate((folder_data, data))

            # If no .npy files were found in the folder, skip to the next folder.
            if folder_data is None:
                continue

            # Cast to float32 and normalize in [0, 1]
            folder_data = folder_data.astype(np.float32) / 255.0
            samples.append(folder_data)
            labels = np.ones(folder_data.shape[0]) * counter
            targets.append(labels)
            counter += 1
    return counter


def load_data(dataset, test_size=0.30, only_normal=False):
    dataset_name, benign, malware = dataset()
    print(dataset_name, benign, malware)

    label_counter = 0
    benign_data, benign_labels, malware_data, malware_labels = [], [], [], []

    label_counter = load_data_from_folder(dataset_name, 'benign', benign_data, benign_labels, label_counter)
    normal = np.array(np.concatenate(benign_data)).astype(np.float32)
    normal_targets = np.array(np.concatenate(benign_labels)).astype(np.int)

    load_data_from_folder(dataset_name, 'malicious', malware_data, malware_labels, label_counter)
    anomaly = np.array(np.concatenate(malware_data)).astype(np.float32)
    anomaly_targets = np.array(np.concatenate(malware_labels)).astype(np.int)

    data = np.concatenate((normal, anomaly))
    targets = np.concatenate((normal_targets, anomaly_targets))
    bin_targets = np.concatenate((np.ones(normal.shape[0]), np.ones(anomaly.shape[0]) * -1))

    df = pd.DataFrame()
    df['sample'] = data.tolist()
    df['target'] = targets
    df['bin_target'] = bin_targets

    if only_normal:
        df = df[df['bin_target'] == 1]

    train, test = train_test_split(df, test_size=test_size, random_state=42)
    train_samples = np.array(train['sample'].tolist()).astype(np.float32)
    train_targets = np.array(train['target'].tolist()).astype(np.int)
    train_bin_targets = np.array(train['bin_target'].tolist()).astype(np.int)

    test_samples = np.array(test['sample'].tolist()).astype(np.float32)
    test_targets = np.array(test['target'].tolist()).astype(np.int)
    test_bin_targets = np.array(test['bin_target'].tolist()).astype(np.int)

    return train_samples, train_targets, train_bin_targets, test_samples, test_targets, test_bin_targets
