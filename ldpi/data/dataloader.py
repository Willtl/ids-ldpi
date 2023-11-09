import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from lib.data.datasets import get_mnist_anomaly_dataset, get_cifar_anomaly_dataset, get_ustc_anomaly_dataset


class Data:
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid


def load_data(args):
    if args.dataset == 'mnist':
        train_ds, test_ds = load_mnist(args)
        train_ds, test_ds = get_mnist_anomaly_dataset(train_ds, test_ds, int(args.abnormal_class))
    elif args.dataset == 'cifar10':
        train_ds, test_ds = load_cifar(args)
        train_ds, test_ds = get_cifar_anomaly_dataset(train_ds, test_ds, train_ds.class_to_idx[args.abnormal_class])
    elif args.dataset == 'ustc':
        np_samples = load_ustc(args)
        train_ds, test_ds = get_ustc_anomaly_dataset(args, np_samples)

    # Ensure the number of samples is divisible by the batch size (it is required even when drop_last=True)
    keep_valid = len(test_ds) - (len(test_ds) % args.batchsize)
    test_ds, _ = torch.utils.data.random_split(test_ds, [keep_valid, len(test_ds) - keep_valid])

    # Define dataloaders
    train_dl = DataLoader(dataset=train_ds, batch_size=args.batchsize, shuffle=True, drop_last=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=args.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, test_dl)


def load_mnist(args):
    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_ds = MNIST(root='./data', train=True, download=True, transform=transform)
    valid_ds = MNIST(root='./data', train=False, download=True, transform=transform)
    return train_ds, valid_ds


def load_cifar(args):
    transform = transforms.Compose(
        [
            # transforms.Resize(args.img_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_ds = CIFAR10(root='./data', train=True, download=True, transform=transform)
    valid_ds = CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_ds, valid_ds


def load_ustc(args):
    path = './data/ustc/raw/samples.npy'
    np_samples = np.load(path, allow_pickle=True)
    return np_samples
