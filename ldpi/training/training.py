import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn

import data
import utils
from ldpi.options import Options
from network import MLP
from options import Options as SnifferOptions

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def init_center_c(loader, net, device, eps=0.1):
    n_samples = 0
    c = torch.zeros(net.rep_dim, device=device)

    net.eval()
    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.to(device)
            outputs = net.encode(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    print(f'Computed center: {c}')
    return c


def pretrain(net, loader, device, epochs=1):
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6)
    mse = nn.MSELoss()

    for i in range(epochs):
        n_batches, total_loss = 0, 0
        for inputs, _, _ in loader:
            inputs = inputs.to(device)
            opt.zero_grad()
            outputs = net(inputs)
            loss = mse(inputs, outputs)
            total_loss += loss.item()
            loss.backward()
            opt.step()
            n_batches += 1
        print(f'Pretrain epoch: {i + 1}, mean loss: {total_loss / n_batches}')

    c = init_center_c(loader, net, device)

    return c


#
# def train(net, loader, c, device, epochs=1, eta=1.0, eps=1e-9):
#     opt = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-6)
#     milestones = [int(epochs * 0.7), int(epochs * 0.8), int(epochs * 0.9)]
#     scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs * 0.75), int(epochs * 0.9)], gamma=0.1)
#
#     net.train()
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         n_batches = 0
#         epoch_start_time = time.time()
#         for inputs, _, bin_targets in loader:
#             inputs, bin_targets = inputs.to(device), bin_targets.to(device)
#
#             outputs = net.encode(inputs)
#             dist = torch.sum((outputs - c) ** 2, dim=1)
#             losses = torch.where(bin_targets == 0, dist, eta * ((dist + eps) ** bin_targets.float()))
#             loss = torch.mean(losses)
#
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#
#             epoch_loss += loss.item()
#             n_batches += 1
#
#             scheduler.step()
#
#         if epoch in milestones:
#             print(f'Adjusted learning rate: {float(scheduler.get_last_lr()[0])}')
#
#         # log epoch statistics
#         epoch_train_time = time.time() - epoch_start_time
#         print(
#             f'| Epoch: {epoch + 1:03}/{epochs:03} | Train Time: {epoch_train_time:.3f}s | Train Loss: {epoch_loss / n_batches:.6f} |')
#
#     return c


def train(net, loader, test_loader, c, device, epochs=1, eta=1.0, eps=1e-9):
    opt = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs * 0.75), int(epochs * 0.9)], gamma=0.1)

    net.train()

    best_dr = 0.0
    best_model_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for inputs, _, bin_targets in loader:
            inputs, bin_targets = inputs.to(device), bin_targets.to(device)

            outputs = net.encode(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            losses = torch.where(bin_targets == 0, dist, eta * ((dist + eps) ** bin_targets.float()))
            loss = torch.mean(losses)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

            scheduler.step()

        # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time
        print(
            f'| Epoch: {epoch + 1:03}/{epochs:03} | Train Time: {epoch_train_time:.3f}s | Train Loss: {epoch_loss / n_batches:.6f} |')

        # periodic testing
        if (epoch + 1) % 10 == 0:
            print("Testing after epoch:", epoch + 1)
            test_auroc, rec_label, threshold = test(net, test_loader, c, device)
            if rec_label > best_dr:
                best_dr = rec_label
                best_model_state = net.state_dict().copy()
                print("New best model found!")

    if best_model_state:
        print('Loading best model')
        net.load_state_dict(best_model_state)
    return c


def test(net, loader, c, device, plot=False):
    net.eval()
    scores = torch.zeros(size=(len(loader.dataset),), dtype=torch.float32, device=device)
    labels = torch.zeros(size=(len(loader.dataset),), dtype=torch.long, device=device)

    with torch.no_grad():
        for i, (inputs, _, bin_targets) in enumerate(loader):
            inputs, bin_targets = inputs.to(device), bin_targets.to(device)

            outputs = net.encode(inputs)
            # outputs = net(inputs)

            dist = torch.sum((outputs - c) ** 2, dim=1)

            c_targets = torch.where(bin_targets == 1, 0, 1)

            scores[i * 64: i * 64 + 64] = dist
            labels[i * 64: i * 64 + 64] = c_targets

    scores = scores.to('cpu').numpy()
    labels = labels.to('cpu').numpy()

    auroc = utils.roc(labels, scores, plot=False)
    print(f'AUROC: {auroc}')

    bool_abnormal = labels.astype(bool)
    bool_normal = ~bool_abnormal
    normal = scores[bool_normal]
    threshold = np.percentile(normal, 99.99)
    threshold = np.max(normal) + (1e-15 * np.max(normal))

    acc, prec, rec, f_score = utils.perf_measure(threshold, labels, scores)
    prec_label = f"{round((1 - prec) * 100, 5):.5f}%"
    rec_label = f"{round(rec * 100, 5):.5f}%"
    print('FAR', prec_label)
    print('detection rate', rec_label)
    print('f1', f_score)

    if plot:
        utils.plot_anomaly_score_dists(test_scores=scores, labels=labels, threshold=np.max(threshold))

    # Fit a skew normal
    # ae, loce, scalee = stats.skewnorm.fit(normal)
    # print(ae, loce, scalee)
    # r = stats.skewnorm.rvs(a=ae, loc=loce, scale=scalee, size=100000)
    # print(r.shape)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    # ax.set_xlim([np.min(r), np.max(r)])
    # ax.legend(loc='best', frameon=False)
    # plt.show()
    # print(f'mean {r.mean()}, std {r.std()}, median {np.median(r)}, min {np.min(r)}, max {np.max(r)}')

    return auroc, rec, threshold


def main():
    epochs_pre, epochs_tra = 250, 500
    args = Options().parse_options()
    general_args = SnifferOptions().parse_options()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simple MLP with a symmetric decoder for pretraining
    net = MLP(input_size=args.n * args.l, num_features=512, rep_dim=128, device=device)

    pretrain_loader = data.get_pretrain(dataset='TII-SSRC-23')
    c = pretrain(net, pretrain_loader, device, epochs=epochs_pre)

    # Generate data, create datasets and dataloaders
    test_samples, test_targets, train_loader, test_loader = data.get_loaders(dataset=general_args.dataset_path)

    # Pretrain, compute c, and train network
    train(net, train_loader, test_loader, c, device, epochs=epochs_tra)

    # Test network and plot ROC
    auroc, rec, threshold = test(net, test_loader, c, device, plot=True)

    # Store model
    net = net.to(torch.device('cpu'))
    torch.save(net.state_dict(), 'output/trained_model.pt')
    np.save(f'output/threshold.npy', threshold)
    np.save(f'output/center.npy', c.to('cpu').numpy())


if __name__ == '__main__':
    main()
