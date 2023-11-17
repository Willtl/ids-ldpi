import torch
import torch.nn.functional as F
from torch import nn
from tsai.models.all import ResCNN


class ResCNNContrastive(nn.Module):
    """A module for supervised contrastive learning."""

    def __init__(self, dim_mid=128, feat_dim=128, verbose=True):
        super().__init__()
        self.dim_mid = dim_mid
        self.feat_dim = feat_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate encoder and head
        self._create_encoder()
        self._create_head()
        self._initialize_weights()

        # Print model and size information if verbose is True
        if verbose:
            print(self)
            print(f'Model size: {sum([param.nelement() for param in self.parameters()]) / 1000000} (M)')

    def _create_encoder(self):
        """Create an encoder model based on the given name and pretrained option."""
        self.encoder = ResCNN(1, 10, separable=True).to(self.device)
        self.dim_in = self.encoder.lin.in_features
        self.encoder.lin = nn.Identity()

    def _create_head(self, depth: int = 4):
        """Create a head model based on the given head type and dimensions."""
        layers = []
        layers.extend([
            nn.Linear(self.dim_in, self.dim_mid, bias=False),
            nn.BatchNorm1d(self.dim_mid),
            nn.ReLU(inplace=True)
        ])
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(self.dim_mid, self.dim_mid, bias=False),
                nn.BatchNorm1d(self.dim_mid),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Linear(self.dim_mid, self.feat_dim, bias=True))
        self.head = nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize encoder and head weights randomly."""
        # Apply He (Kaiming) initialization to the encoder layers
        for layer in self.encoder.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        # Apply Glorot (Xavier) initialization to the head layers
        for layer in self.head.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """Compute forward pass."""
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

    def encode(self, x):
        """Compute forward pass."""
        feat = self.encoder(x)
        feat = self.head(feat)
        return feat
