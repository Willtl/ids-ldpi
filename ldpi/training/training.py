import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import data
import utils
from ldpi.options_ldpi import LDPIOptions
from losses import OneClassContrastiveLoss
from network import MLP, ResCNNContrastive


class Trainer:
    def __init__(self, args: LDPIOptions, model_type: str = 'ResCNN') -> None:
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.args: LDPIOptions = args
        self.model_type = model_type
        self.device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == 'MLP':
            self.model: MLP = MLP(input_size=args.n * args.l, num_features=512, feat_dim=128, device=self.device)
        elif model_type == 'ResCNN':
            self.model = ResCNNContrastive().to(self.device)
        else:
            print('Error no model type found')

        self.center: Optional[Tensor] = None

    def init_center_c(self, loader: DataLoader, eps: float = 0.01) -> Tensor:
        """
        Initialize the center vector for the network.

        Args:
            loader (DataLoader): The DataLoader for the dataset.
            eps (float, optional): A small epsilon value to avoid division by zero. Defaults to 0.1.

        Returns:
            torch.Tensor: The initialized center vector.
        """
        self.center = torch.zeros(self.model.feat_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            total_outputs = 0
            for inputs, *_ in loader:
                inputs = inputs.unsqueeze(1).to(self.device)

                # Encode the inputs and unsqueeze to get [bs, 1, feat] shape
                outputs = self.model.encode(inputs.to(self.device))

                # Sum the outputs for the current batch and add to the total
                total_outputs += outputs.sum(0)

        n_samples = sum(len(inputs) for inputs, *_ in loader)
        self.center = total_outputs / n_samples

        # Apply epsilon threshold
        # eps_mask = (abs(self.center) < eps)
        # self.center[eps_mask & (self.center < 0)] = -eps
        # self.center[eps_mask & (self.center > 0)] = eps
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
        loader = data.get_pretrain_dataloader(dataset='TII-SSRC-23', batch_size=self.args.batch_size, contrastive=self.model_type != 'MLP')

        if self.model_type == 'MLP':
            self._pretrain_ae(loader)
        elif self.model_type == 'ResCNN':
            self._pretrain_contrastive(loader)

        loader = data.get_pretrain_dataloader(dataset='TII-SSRC-23', batch_size=self.args.batch_size, contrastive=False)
        self.init_center_c(loader)

    def _pretrain_contrastive(self, loader):
        # Check if the pretrained model exists
        model_path = 'output/pretrained_model.pth'
        if os.path.exists(model_path):
            print("Pretrained model found. Loading model...")
            self.model.load_state_dict(torch.load(model_path))
            return

        initial_lr = 0.1
        warmup_epochs = 100
        total_epochs = self.args.pretrain_epochs
        optimizer = optim.SGD(self.model.parameters(), lr=initial_lr, weight_decay=0.0003)
        criterion = OneClassContrastiveLoss(tau=0.2)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=0.0)

        with tqdm(total=total_epochs, desc="Training Progress") as pbar:
            for epoch in range(total_epochs):
                # Warmup phase
                if epoch < warmup_epochs:
                    warmup_lr = ((initial_lr - 1e-07) / warmup_epochs) * epoch + 1e-07
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr

                n_batches, total_loss = 0, 0
                for v1, v2 in loader:
                    if torch.cuda.is_available():
                        v1, v2 = v1.cuda(), v2.cuda()

                    v1, v2 = v1.unsqueeze(1), v2.unsqueeze(1)
                    concatenated = torch.cat((v1, v2), dim=0)
                    features = self.model(concatenated)
                    features = torch.stack(features.split(features.size(0) // 2), dim=1)

                    # Compute loss
                    loss = criterion(features)

                    # Zero the gradients before running the backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                # Update the learning rate with cosine annealing after warmup
                if epoch >= warmup_epochs:
                    scheduler.step()

                # Update tqdm bar instead of printing
                lr = optimizer.param_groups[0]["lr"]
                mean_loss = total_loss / n_batches
                pbar.set_description(f"Epoch: {epoch + 1}, LR: {lr:.5f}, Mean loss: {mean_loss:.5f}")
                pbar.update()

    def _pretrain_ae(self, loader):
        opt = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        mse = nn.MSELoss()

        for epoch in range(self.args.pretrain_epochs):
            n_batches, total_loss = 0, 0
            for inputs, _, _ in loader:
                inputs = inputs.to(self.device)
                opt.zero_grad()
                if self.model_type != 'MLP':
                    inputs = inputs.unsqueeze(1)
                outputs = self.model(inputs)
                loss = mse(inputs, outputs)
                total_loss += loss.item()
                loss.backward()
                opt.step()
                n_batches += 1

            print(f'Pretrain epoch: {epoch + 1}, mean loss: {total_loss / n_batches}')

    def train(self, eta=1.0, eps=1e-10, per_validation: int = 5):
        self.train_loader, self.test_loader = data.get_training_dataloader(dataset='TII-SSRC-23')

        # Check if the best model is trained and center are already saved
        model_save_path = 'output/best_model_with_center.pth'
        if os.path.isfile(model_save_path):
            print('Loading best model and center from saved state')
            saved_state = torch.load(model_save_path)
            print(saved_state)
            quit()

            self.model.load_state_dict(saved_state['model_state_dict'])
            self.center = saved_state['center']
            return

        # Optimizer configuration
        initial_lr = 0.01
        optimizer = optim.SGD(self.model.parameters(), lr=initial_lr, weight_decay=0.0003)

        # Scheduler configuration
        warmup_epochs = min(100, 0.1 * self.args.epochs)
        warmup_lr = 1e-10
        lr_increment = (initial_lr - 1e-10) / warmup_epochs
        total_epochs = self.args.epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=0.0)

        self.model.train()

        best_dr = 0.0
        best_model_state = None
        continue_warmup = True

        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            # Warmup LR
            if continue_warmup and epoch < warmup_epochs:
                warmup_lr += lr_increment
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Set model for training
            self.model.train()
            for inputs, targets, bin_targets in self.train_loader:
                optimizer.zero_grad()

                inputs, bin_targets = inputs.to(self.device).unsqueeze(1), bin_targets.to(self.device)

                outputs = self.model.encode(inputs)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                losses = torch.where(bin_targets == 0, dist, eta * ((dist + eps) ** bin_targets.float()))
                loss = torch.mean(losses)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Update the learning rate with cosine annealing after warmup
            if epoch >= warmup_epochs:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print(f'| Epoch: {epoch + 1:03}/{self.args.epochs:03} | Train Time: {epoch_train_time:.3f}s | Train Loss: {epoch_loss / n_batches:.6f} | LR: {current_lr}')

            # periodic testing
            if (epoch + 1) % per_validation == 0 or epoch < warmup_epochs:
                print("Validation for early stopping at epoch:", epoch + 1)
                test_auroc, rec_label, threshold = self.test(plot=False, multiclass=False)
                if rec_label > best_dr:
                    best_dr = rec_label
                    best_model_state = self.model.state_dict().copy()
                    print("New best model found!")
                    continue_warmup = True

                    output_dir = 'output'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Create a dictionary to hold both model state and center
                    save_dict = {
                        'model_state_dict': best_model_state,
                        'center': self.center
                    }

                    # Save the dictionary to a file
                    model_save_path = os.path.join(output_dir, 'best_model_with_center.pth')
                    torch.save(save_dict, model_save_path)
                else:
                    continue_warmup = False

        if best_model_state:
            print('Loading best model')
            self.model.load_state_dict(best_model_state)

    def test(self, plot: bool = False, multiclass: bool = False) -> Tuple[float, float, float]:
        self.model.eval()
        dataset_size = len(self.test_loader.dataset)
        scores = torch.zeros(size=(dataset_size,), dtype=torch.float32, device=self.device)
        bin_labels = torch.zeros(size=(dataset_size,), dtype=torch.long, device=self.device)
        mult_labels = torch.zeros(size=(dataset_size,), dtype=torch.long, device=self.device)

        with torch.no_grad():
            for i, (inputs, targets, bin_targets) in enumerate(self.test_loader):
                inputs, targets, bin_targets = inputs.to(self.device).unsqueeze(1), targets.to(self.device), bin_targets.to(self.device)

                outputs = self.model.encode(inputs)

                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                c_targets = torch.where(bin_targets == 1, 0, 1)

                batch_index = i * 64
                scores[batch_index: batch_index + 64] = dist
                bin_labels[batch_index: batch_index + 64] = c_targets
                mult_labels[batch_index: batch_index + 64] = targets

        scores_np = scores.to('cpu').numpy()
        bin_labels_np = bin_labels.to('cpu').numpy()
        mult_labels_np = mult_labels.to('cpu').numpy()

        auroc = utils.roc(scores_np, bin_labels_np, plot=False)
        results = {'auc': auroc}

        bool_abnormal = bin_labels_np.astype(bool)
        bool_normal = ~bool_abnormal
        normal = scores_np[bool_normal]
        abnormal = scores_np[bool_abnormal]

        # Compute 99.99th threshold and max threshold
        nine_nine_threshold = np.percentile(normal, 99.99)
        nine_nine_threshold = np.nextafter(nine_nine_threshold, np.inf)
        max_threshold = np.max(normal)
        max_threshold = np.nextafter(max_threshold, np.inf)

        for name, threshold in [('99.99th', nine_nine_threshold), ('max', max_threshold)]:
            acc, prec, rec, f_score = utils.perf_measure(threshold, bin_labels_np, scores_np)
            results[name] = {'threshold': threshold, 'acc': acc, 'prec': prec, 'rec': rec, 'f_score': f_score}

            if plot:
                # Calculate 4 * max_threshold for plotting
                right_limit = np.percentile(abnormal, 0.1)

                # Filter out scores and corresponding labels that are above 4 * max_threshold for plotting
                valid_indices = scores_np <= right_limit
                plot_scores_np = scores_np[valid_indices]
                plot_bin_labels_np = bin_labels_np[valid_indices]
                plot_mult_labels_np = mult_labels_np[valid_indices]

                if not multiclass:
                    utils.plot_anomaly_score_dists(test_scores=plot_scores_np, labels=plot_bin_labels_np, name=name, threshold=threshold)
                else:
                    utils.plot_multiclass_anomaly_scores(test_scores=plot_scores_np, labels=plot_mult_labels_np, name=name, threshold=threshold)

        return results

    def plot_results(self):
        pass


def main():
    args = LDPIOptions().parse_options()
    trainer = Trainer(args)
    trainer.pretrain()
    trainer.train()
    results = trainer.test(plot=True)
    print(results)


if __name__ == '__main__':
    main()
