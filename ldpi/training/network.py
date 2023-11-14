import torch
import torch.nn.functional as F
from torch import nn
from tsai.models.all import ResCNN


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        with torch.no_grad():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
                torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class MLP(nn.Module):
    def __init__(self, input_size: int, num_features: int, rep_dim: int, device: torch.device, bias: bool = True, num_layers: int = 2) -> None:
        super(MLP, self).__init__()

        self.input_size = input_size
        self.num_features = num_features
        self.rep_dim = rep_dim
        self.device = device

        self.encoder = self._build_encoder(input_size, num_features, rep_dim, bias, num_layers)
        self.decoder = self._build_decoder(input_size, num_features, rep_dim, bias, num_layers)

        self.dropout = nn.Dropout(0.2)

    def _build_encoder(self, input_size: int, num_features: int, rep_dim: int, num_layers: int = 2) -> nn.Sequential:
        layers = [
            nn.Linear(input_size, num_features, bias=False),  # Exclude bias when using BatchNorm
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(num_features, num_features, bias=False),  # Exclude bias when using BatchNorm
                nn.BatchNorm1d(num_features),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Linear(num_features, rep_dim, bias=True))
        return nn.Sequential(*layers).to(self.device)

    def _build_decoder(self, input_size: int, num_features: int, rep_dim: int, num_layers: int = 2) -> nn.Sequential:
        layers = [
            nn.Linear(rep_dim, num_features, bias=False),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(num_features, num_features, bias=False),
                nn.BatchNorm1d(num_features),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Linear(num_features, input_size, bias=True))
        return nn.Sequential(*layers).to(self.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(x)
        return self.decoder(dropped)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class ResCNNContrastive(nn.Module):
    """A module for supervised contrastive learning."""

    def __init__(self, head='mlp', dim_mid=128, feat_dim=128, verbose=True):
        super().__init__()
        self.dim_mid = dim_mid
        self.feat_dim = feat_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate encoder and head
        self._create_encoder()
        self._create_head(head)

        # Print model and size information if verbose is True
        if verbose:
            print(self)
            print(f'Model size: {sum([param.nelement() for param in self.parameters()]) / 1000000} (M)')

    def _create_encoder(self):
        """Create an encoder model based on the given name and pretrained option."""
        self.encoder = ResCNN(1, 10, separable=True).to(self.device)
        self.dim_in = self.encoder.lin.in_features
        self.encoder.lin = nn.Identity()

    def _create_head(self, head: str, depth: int = 4):
        """Create a head model based on the given head type and dimensions."""

        layers = []
        if head == 'linear':
            layers.append(nn.Linear(self.dim_in, self.feat_dim, bias=True))
        elif head == 'mlp':
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
        else:
            raise NotImplementedError(f'Head not supported: {head}')

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

    def features(self, x):
        """Compute features without head."""

        feat = self.encoder(x)
        return feat
