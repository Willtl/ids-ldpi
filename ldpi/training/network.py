import torch
from torch import nn


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
    def __init__(self,
                 input_size: int,
                 num_features: int,
                 rep_dim: int,
                 device: torch.device,
                 bias: bool = True,
                 num_layers: int = 2) -> None:
        super(MLP, self).__init__()

        self.input_size = input_size
        self.num_features = num_features
        self.rep_dim = rep_dim
        self.device = device

        self.encoder = self._build_encoder(input_size, num_features, rep_dim, bias, num_layers)
        self.decoder = self._build_decoder(input_size, num_features, rep_dim, bias, num_layers)

        self.dropout = nn.Dropout(0.2)

    def _build_encoder(self, input_size: int, num_features: int, rep_dim: int, bias: bool = True, num_layers: int = 2) -> nn.Sequential:
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

        layers.append(nn.Linear(num_features, rep_dim, bias=bias))

        return nn.Sequential(*layers).to(self.device)

    def _build_decoder(self,
                       input_size: int,
                       num_features: int,
                       rep_dim: int,
                       bias: bool = True,
                       num_layers: int = 2) -> nn.Sequential:
        layers = [
            nn.Linear(rep_dim, num_features, bias=False),  # Exclude bias when using BatchNorm
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True)
        ]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(num_features, num_features, bias=False),  # Exclude bias when using BatchNorm
                nn.BatchNorm1d(num_features),
                nn.ReLU(inplace=True)
            ])

        layers.append(nn.Linear(num_features, input_size, bias=bias))

        return nn.Sequential(*layers).to(self.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(x)
        return self.decoder(dropped)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class OneDCNN(nn.Module):
    def __init__(self, input_size, nz, nc=1, nf=16, bias=False):
        super().__init__()
        self.input_size = input_size
        self.nc = nc
        self.rep_dim = nz
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear_size = (nf * 4) * int(input_size / 8)

        self.encoder = nn.Sequential(
            nn.Conv1d(nc, nf, 4, 2, 1, bias=bias),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(nf, nf * 2, 4, 2, 1, bias=bias),
            nn.BatchNorm1d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(nf * 2, nf * 4, 4, 2, 1, bias=bias),
            nn.BatchNorm1d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(self.linear_size, nz, bias=bias)
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(nz, self.linear_size, bias=bias),
            nn.BatchNorm1d(self.linear_size),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (nf * 4, -1)),

            nn.ConvTranspose1d(nf * 4, nf * 2, 4, 2, 1, bias=bias),
            nn.BatchNorm1d(nf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(nf * 2, nf, 4, 2, 1, bias=bias),
            nn.BatchNorm1d(nf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(nf, nc, 4, 2, 1, bias=bias),
            nn.Sigmoid()
        ).to(self.device)

        # init_weights(self.encoder, init_type='normal')
        # init_weights(self.decoder, init_type='normal')

    def encode(self, x):
        return self.encoder(x.view(x.shape[0], 1, -1))

    def decode(self, x):
        return self.decoder(x).view(x.shape[0], -1)

    def forward(self, x):
        return self.decode(self.encode(x))
