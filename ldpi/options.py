import argparse


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Sniffing and buffering params
        self.parser.add_argument('--interface', default='enp5s0', help='interface to sniff')
        self.parser.add_argument('--attacker', default='172.20.10.5', help='flow or session filtering')
        self.parser.add_argument('--n', type=int, default=4, help='number of packets per sample (default: 2)')
        self.parser.add_argument('--l', type=int, default=100, help='size of each packet in the samples (default: 128)')

        # Detector params
        self.parser.add_argument('--log_interval', type=int, default=1.0, help='logging interval (default: 1.0)')
        self.parser.add_argument('--detect_interval', type=int, default=1.0,
                                 help='detection interval in seconds (default: 0.5)')
        self.parser.add_argument('--img_size', type=int, default=16, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=1, help='input image channels')

        # Model architecture related
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--noise', type=bool, default=False, help='apply noise during forward pass')
        self.parser.add_argument('--ngf', type=int, default=64, help='number of features generator')
        self.parser.add_argument('--ndf', type=int, default=64, help='number of features discriminator')
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')

        # Training related
        self.parser.add_argument('--w_adv', type=float, default=1.0, help='Weight for adversarial loss. default=1')
        self.parser.add_argument('--w_con', type=float, default=50.0, help='Weight for reconstruction loss. default=50')
        self.parser.add_argument('--w_lat', type=float, default=1.0, help='Weight for latent space loss. default=1')

    def parse_options(self):
        return self.parser.parse_args()
