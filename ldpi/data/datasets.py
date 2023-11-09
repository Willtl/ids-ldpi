import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

''' MNIST '''


def get_mnist_anomaly_dataset(train_dataset, test_dataset, abn_class_index=0):
    # Get images and labels.
    train_samples, train_labels = train_dataset.data, train_dataset.targets
    test_samples, test_labels = test_dataset.data, test_dataset.targets

    # Find normal abnormal indexes.
    norm_train_index = torch.from_numpy(np.where(train_labels.numpy() != abn_class_index)[0])
    abn_train_index = torch.from_numpy(np.where(train_labels.numpy() == abn_class_index)[0])
    norm_test_index = torch.from_numpy(np.where(test_labels.numpy() != abn_class_index)[0])
    abn_test_index = torch.from_numpy(np.where(test_labels.numpy() == abn_class_index)[0])

    # Find normal and abnormal images
    nrm_trn_img = train_samples[norm_train_index]  # Normal training images
    abn_trn_img = train_samples[abn_train_index]  # Abnormal training images.
    nrm_tst_img = test_samples[norm_test_index]  # Normal training images
    abn_tst_img = test_samples[abn_test_index]  # Abnormal training images.

    # Find normal and abnormal labels.
    nrm_trn_lbl = train_labels[norm_train_index]  # Normal training labels
    abn_trn_lbl = train_labels[abn_train_index]  # Abnormal training labels.
    nrm_tst_lbl = test_labels[norm_test_index]  # Normal training labels
    abn_tst_lbl = test_labels[abn_test_index]  # Abnormal training labels.

    # nrm_trn_lbl[:] = 0
    # Assign labels to normal (0) and abnormal (1)
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    train_dataset.data = nrm_trn_img.clone()
    train_dataset.targets = nrm_trn_lbl.clone()

    test_dataset.data = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    test_dataset.targets = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return train_dataset, test_dataset


''' CIFAR '''


def get_cifar_anomaly_dataset(train_ds, valid_ds, abn_cls_idx=0):
    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, np.array(train_ds.targets)
    tst_img, tst_lbl = valid_ds.data, np.array(valid_ds.targets)

    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]

    nrm_trn_img = trn_img[nrm_trn_idx]  # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]  # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]  # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]  # Abnormal training labels.

    nrm_tst_img = tst_img[nrm_tst_idx]  # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]  # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]  # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]  # Abnormal training labels.

    # Assign labels to normal (0) and abnormals (1)
    # nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset
    train_ds.data = np.copy(nrm_trn_img)
    valid_ds.data = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    train_ds.targets = np.copy(nrm_trn_lbl)
    valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return train_ds, valid_ds


''' USTC '''


class USTC(Dataset):
    def __init__(self, samples, labels):
        self.samples, self.labels, self.n_samples = samples, labels, samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return self.n_samples


def get_ustc_anomaly_dataset(args, np_samples):
    classes = list(range(20))
    benign_labels = classes[:10]
    malware_labels = classes[10:]

    df_data = pd.DataFrame(np_samples, columns=['SAMPLE', 'TARGET'])

    # Separate data into benign and malware
    benign_df = df_data[df_data['TARGET'].isin(benign_labels)]
    malware_df = df_data[df_data['TARGET'].isin(malware_labels)]

    # Split the benign into train (5400) and test (600)
    test_indexes = []
    for label in benign_labels:
        indexes = benign_df.TARGET[benign_df.TARGET.eq(label)].sample(600).index
        test_indexes.append(indexes)
    indexes = test_indexes[0]
    for i in range(1, len(test_indexes)):
        indexes = indexes.union(test_indexes[i])

    # Filter the test samples given respective indexes
    benign_df_test = benign_df.loc[indexes]
    benign_df = benign_df.drop(indexes)

    # Remove additional malware classes
    for label in malware_labels:
        indexes = malware_df.TARGET[malware_df.TARGET.eq(label)].sample(5400).index
        malware_df = malware_df.drop(indexes)

    # keep = 600
    # for i in range(10, 20):
    #     tmp = malware_df[malware_df['TARGET'] == i]
    #     remove = len(tmp) - keep
    #     print(f'Removing {remove} of class {i}')
    #     drop_indices = np.random.choice(tmp.index, remove, replace=False)
    #     malware_df = malware_df.drop(drop_indices)

    # Process dataframe and create tensors,,
    ben_train_samples, ben_train_labels = df_to_tensor(benign_df)
    ben_test_samples, ben_test_labels = df_to_tensor(benign_df_test)
    mal_test_samples, mal_test_labels = df_to_tensor(malware_df)

    if args.architecture == '2D':
        ben_train_samples = ben_train_samples.reshape(-1, args.nc, args.img_size, args.img_size)
        ben_test_samples = ben_test_samples.reshape(-1, args.nc, args.img_size, args.img_size)
        mal_test_samples = mal_test_samples.reshape(-1, args.nc, args.img_size, args.img_size)

    # Assign label 0 to normal and 1 to abnormal
    # ben_train_anom_labels = torch.zeros_like(ben_train_labels)
    ben_test_anom_labels = torch.zeros_like(ben_test_labels)
    mal_test_anom_labels = torch.ones_like(mal_test_labels)

    # Concatenate benign and malware test samples
    test_samples = torch.cat((ben_test_samples, mal_test_samples), axis=0)
    test_labels = torch.cat((ben_test_anom_labels, mal_test_anom_labels), axis=0)

    # Create training and testing datasets
    train_ds = USTC(ben_train_samples, ben_train_labels)
    test_ds = USTC(test_samples, test_labels)

    return train_ds, test_ds


def df_to_tensor(df):
    # Define shape
    n_samples = df.shape[0]
    l_samples = len(df.iloc[0][0])  # first row, first column
    # Define numpy arrays based on the shape
    samples = np.zeros([n_samples, 1, l_samples], dtype='float32')
    labels = np.zeros([n_samples], dtype='int64')
    for i in range(n_samples):
        samples[i][0] = df.iloc[i]['SAMPLE']
        labels[i] = df.iloc[i]['TARGET']
    return torch.from_numpy(samples), torch.from_numpy(labels)
