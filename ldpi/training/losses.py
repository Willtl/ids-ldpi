import torch
import torch.nn as nn
import torch.nn.functional as F


class OneClassContrastiveLoss(nn.Module):
    def __init__(self, tau=0.07, reduction='mean'):
        super(OneClassContrastiveLoss, self).__init__()
        self.tau = tau
        self.reduction = reduction

    # Parameter `features` should be L2-normalized during forward pass
    def forward(self, features):
        # Split the features into two views
        f1, f2 = features[:, 0, :], features[:, 1, :]

        # Compute the cosine similarity (given that f1 and f2 are L2-normalized)
        cos_similarity = torch.mm(f1, f2.t())

        # Scale the cosine similarities by the temperature tau
        logits = cos_similarity / self.tau

        # Labels for each entry in a batch are the indices themselves since the diagonal corresponds to the positive examples (each entry with its own augmented version)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Calculate the cross-entropy loss, which automatically applies softmax
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss
