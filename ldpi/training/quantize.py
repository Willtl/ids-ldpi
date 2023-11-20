import random
import time

import numpy as np
import torch
from torch import nn

import data
import utils
from model import MLP
from options import LDPIOptions
from training import test

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

"""
Here we describe the use of the PyTorch library torch.quantization, which provides tools for quantization-aware 
training and post-training quantization, for quantizing the LDPI model for running at the CM4s. 

We use post-training static quantization. Static, different than dynamic, quantization is a technique where the 
quantization of the model's parameters and activations is done before deployment, following the training phase. This 
means that the quantization process is done offline and the quantized model is then deployed. One main advantage of 
static quantization is that it allows for more control over the quantization process, as well as the ability to 
optimize for a specific target device. 

The quantize() function is used to quantize a PyTorch neural network for ARM. It does this by performing the following steps:
1. It sets the backend for quantization to 'qnnpack' and sets the device to 'cpu'.
2. It loads the trained model, center, threshold, and net using the function init_model().
3. It loads the dataset and creates datasets and data loaders using the function data.get_loaders()
4. It extracts the encoder part of the net and applies quantization to it.
5. Then it applies quantization on the model using torch.quantization.fuse_modules() and torch.quantization.prepare() functions.
6. It saves the quantized model using the torch.jit.save() function.
7. Then it runs the inference on the quantized model and saves the output.
8. Finally, it runs the test function on the quantized model to check the performance.

'qnnpack' is a library for accelerating neural network inference on mobile devices. It is specifically optimized for 
ARM processors and aims to deliver high performance while maintaining low power consumption. The library provides 
implementations of common neural network operators such as convolution, fully connected layers, and pooling, 
as well as support for quantized 8-bit integer arithmetic, which allows for faster computations and less memory usage. 
"""


def init_model(args, device):
    try:
        center = torch.from_numpy(np.load('output/center.npy')).to(device)
        threshold = torch.from_numpy(np.load('output/threshold.npy')).to(device)

        # Instantiate model architecture without trained weights
        model = MLP(input_size=args.n * args.l, num_features=512, rep_dim=128, device=device)

        # Set trainable weights to the model
        model.load_state_dict(
            torch.load(f'output/trained_model.pt', map_location=device)
        )

        return center, threshold, model
    except Exception as e:
        print(f'Exception loading model {e}')
        quit()


def inference(args, net):
    net.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(10000):
            inputs = torch.rand((64, args.n * args.l))
            net(inputs)
    print('TIME >>> ', time.time() - start)


def main():
    backend = 'qnnpack'
    # backend = 'fbgemm'

    args = LDPIOptions().parse()
    device = torch.device('cpu')

    dataset = utils.get_dataset_local

    # Simple MLP with a symmetric decoder for pretraining
    center, threshold, net = init_model(args, device)

    # Generate data, create datasets and dataloaders
    test_samples, test_targets, train_loader, test_loader = data.get_loaders(dataset=dataset)
    samples = torch.from_numpy(test_samples)
    print(samples)

    m = net.encoder

    # Post training static quantization
    auroc, rec, threshold = test(m, test_loader, center, device)
    print(auroc, rec, threshold)

    # Test network and plot ROC
    inference(args, m)

    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
    torch.quantization.fuse_modules(m, ['2', '3'], inplace=True)
    m = nn.Sequential(torch.quantization.QuantStub(),
                      *m,
                      torch.quantization.DeQuantStub())

    m.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend

    torch.quantization.prepare(m, inplace=True)

    with torch.inference_mode():
        for i in range(samples.shape[0]):
            m(samples[i])

    torch.quantization.convert(m, inplace=True)

    m = torch.jit.trace(m, samples)
    torch.jit.save(m, f'output/quantized.pt')
    print('saved')

    inference(args, m)

    m_loaded = torch.jit.load(f'output/quantized.pt')
    # Test network and plot ROC
    inference(args, m_loaded)

    # Post training static quantization
    auroc, rec, threshold = test(m, test_loader, center, device)
    print(auroc, rec, threshold)


if __name__ == '__main__':
    main()
