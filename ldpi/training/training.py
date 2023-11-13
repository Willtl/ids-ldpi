import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

import data
import utils
from ldpi.ldpioptions import LDPIOptions
from network import MLP

# Setting seeds for reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


class Trainer:
    def __init__(self, args) -> None:
        """
        Initialize the Trainer class.

        Args:
            net (MLP): The neural network model.
            device (torch.device): The device (CPU or GPU) to use for training.
        """
        self.train_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.args: LDPIOptions = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MLP(input_size=args.n * args.l, num_features=512, rep_dim=128, device=self.device)
        self.center = None

    def init_center_c(self, loader: DataLoader, eps: float = 0.1) -> torch.Tensor:
        """
        Initialize the center vector for the network.

        Args:
            loader (DataLoader): The DataLoader for the dataset.
            eps (float, optional): A small epsilon value to avoid division by zero. Defaults to 0.1.

        Returns:
            torch.Tensor: The initialized center vector.
        """
        n_samples = 0
        self.center = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for inputs, _, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model.encode(inputs)
                n_samples += outputs.shape[0]
                self.center += torch.sum(outputs, dim=0)

        self.center /= n_samples

        # Apply epsilon threshold
        self.center[(abs(self.center) < eps) & (self.center < 0)] = -eps
        self.center[(abs(self.center) < eps) & (self.center > 0)] = eps
        print(f'Computed center: {self.center}')

    def pretrain(self) -> torch.Tensor:
        """
        Pretrain the network.

        Args:
            loader (DataLoader): The DataLoader for the dataset.
            epochs (int, optional): The number of epochs for pretraining. Defaults to 1.

        Returns:
            torch.Tensor: The center vector computed after pretraining.
        """
        # Generate data, create datasets and dataloaders
        loader = data.get_pretrain_dataloader(dataset='TII-SSRC-23', batch_size=self.args.batch_size)

        opt = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        mse = nn.MSELoss()

        for epoch in range(self.args.pretrain_epochs):
            n_batches, total_loss = 0, 0
            for inputs, _, _ in loader:
                inputs = inputs.to(self.device)
                opt.zero_grad()
                outputs = self.model(inputs)
                loss = mse(inputs, outputs)
                total_loss += loss.item()
                loss.backward()
                opt.step()
                n_batches += 1

            print(f'Pretrain epoch: {epoch + 1}, mean loss: {total_loss / n_batches}')

        self.init_center_c(loader)

    def train(self, eta=1.0, eps=1e-10, per_validation: int = 5):
        self.train_loader, self.test_loader = data.get_training_dataloader(dataset='TII-SSRC-23')

        opt = optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[int(self.args.epochs * 0.75), int(self.args.epochs * 0.9)], gamma=0.1)

        self.model.train()

        best_dr = 0.0
        best_model_state = None

        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for inputs, targets, bin_targets in self.train_loader:
                inputs, bin_targets = inputs.to(self.device), bin_targets.to(self.device)

                outputs = self.model.encode(inputs)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
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
            print(f'| Epoch: {epoch + 1:03}/{self.args.epochs:03} | Train Time: {epoch_train_time:.3f}s | Train Loss: {epoch_loss / n_batches:.6f} |')

            # periodic testing
            if (epoch + 1) % per_validation == 0:
                print("Validation for early stopping at epoch:", epoch + 1)
                test_auroc, rec_label, threshold = self.test(plot=False, multiclass=False)
                if rec_label > best_dr:
                    best_dr = rec_label
                    best_model_state = self.model.state_dict().copy()
                    print("New best model found!")

        if best_model_state:
            print('Loading best model')
            self.model.load_state_dict(best_model_state)

    def test(self, plot=False, multiclass=False):
        self.model.eval()
        scores = torch.zeros(size=(len(self.test_loader.dataset),), dtype=torch.float32, device=self.device)
        bin_labels = torch.zeros(size=(len(self.test_loader.dataset),), dtype=torch.long, device=self.device)
        mult_labels = torch.zeros(size=(len(self.test_loader.dataset),), dtype=torch.long, device=self.device)

        with torch.no_grad():
            for i, (inputs, targets, bin_targets) in enumerate(self.test_loader):
                inputs, targets, bin_targets = inputs.to(self.device), targets.to(self.device), bin_targets.to(self.device)

                outputs = self.model.encode(inputs)
                # outputs = net(inputs)

                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                c_targets = torch.where(bin_targets == 1, 0, 1)

                scores[i * 64: i * 64 + 64] = dist
                bin_labels[i * 64: i * 64 + 64] = c_targets
                mult_labels[i * 64: i * 64 + 64] = targets

        scores = scores.to('cpu').numpy()
        bin_labels = bin_labels.to('cpu').numpy()
        mult_labels = mult_labels.to('cpu').numpy()

        auroc = utils.roc(scores, bin_labels, plot=False)
        print(f'AUROC: {auroc}')

        bool_abnormal = bin_labels.astype(bool)
        bool_normal = ~bool_abnormal
        normal = scores[bool_normal]
        nine_nine_threshold = np.percentile(normal, 99.99)
        threshold = np.nextafter(nine_nine_threshold, np.inf)
        # threshold = np.max(normal)
        # threshold = np.nextafter(threshold, np.inf)

        acc, prec, rec, f_score = utils.perf_measure(threshold, bin_labels, scores)
        prec_label = f"{round((1 - prec) * 100, 5):.5f}%"
        rec_label = f"{round(rec * 100, 5):.5f}%"
        print('FAR', prec_label)
        print('detection rate', rec_label)
        print('f1', f_score)

        if plot:
            if not multiclass:
                utils.plot_anomaly_score_dists(test_scores=scores, labels=bin_labels, threshold=np.max(threshold))
            else:
                utils.plot_multiclass_anomaly_scores(test_scores=scores, labels=mult_labels, threshold=np.max(threshold))

        return auroc, rec, threshold

    def plot_results(self):
        pass


def main():
    args = LDPIOptions().parse_options()
    trainer = Trainer(args)
    trainer.pretrain()
    trainer.train()
    trainer.plot_results()
    quit()

    # Store model
    net = net.to(torch.device('cpu'))
    torch.save(net.state_dict(), 'output/trained_model.pt')
    np.save(f'output/threshold.npy', threshold)
    np.save(f'output/center.npy', c.to('cpu').numpy())


if __name__ == '__main__':
    main()

    # Additional main logic for training and testing would go here
