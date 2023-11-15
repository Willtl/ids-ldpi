import argparse


class LDPIOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Sniffing and buffering params
        self.parser.add_argument('--interface', default='enp5s0', help='interface to sniff')
        self.parser.add_argument('--attacker', default='172.20.10.5', help='flow or session filtering')
        self.parser.add_argument('--n', type=int, default=4, help='number of packets per sample (default: 4)')
        self.parser.add_argument('--l', type=int, default=60, help='size of each packet in the samples (default: 64)')

        # Training params
        self.parser.add_argument('--pretrain_epochs', type=int, default=2000, help='number of packets per sample (default: 4)')
        self.parser.add_argument('--epochs', type=int, default=100, help='number of packets per sample (default: 4)')
        self.parser.add_argument('--batch_size', type=int, default=64, help='training batch size (default: 64)')

        # Detector params
        self.parser.add_argument('--log_interval', type=int, default=1.0, help='logging interval (default: 1.0)')
        self.parser.add_argument('--detect_interval', type=int, default=1.0, help='detection interval in seconds (default: 0.5)')

    def parse_options(self):
        return self.parser.parse_args()
