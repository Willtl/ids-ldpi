import argparse


class SnifferOptions:
    def __init__(self):
        self.dataset_path: str = '../../datasets/TII-SSRC-23/pcap/'
        self.json_path: str = ''
        self.precision_floats: int = 4
        self.delay: bool = False
        self.session: bool = False
        self.aggregate_csvs: bool = True
        self.interface: str = 'enp5s0'
        self.timeout: int = 120
        self.cleaning_cycle: int = 60
        self.time_window: int = 60
        self.debug: bool = True
        self.parse_options()

    def parse_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--delay', type=bool, default=self.delay, help='If debug true, then consider or not delay between packets while reading .pcap.')
        parser.add_argument('--session', type=bool, default=self.session, help='If true, consider bidirectional flows, otherwise unidirectional.')
        parser.add_argument('--interface', default=self.interface, help='interface to sniff')
        parser.add_argument('--timeout', type=int, default=self.timeout, help='seconds to consider connection teardown')
        parser.add_argument('--cleaning_cycle', type=int, default=self.cleaning_cycle, help='loop through flows and check if no packet was received since')
        parser.add_argument('--time_window', type=int, default=self.time_window, help='time interval to compute features of flows (seconds).')

        parser.add_argument('--dataset_path', type=str, default=self.dataset_path, help='Dataset folder path containing the .pcap files.')
        parser.add_argument('--json_path', type=str, default=self.json_path, help='JSON path containing the mapping of for labeling (Optional).')

        args = parser.parse_args()

        self.delay = args.delay
        self.session = args.session
        self.interface = args.interface
        self.timeout = args.timeout
        self.cleaning_cycle = args.cleaning_cycle
        self.time_window = args.time_window

        self.dataset_path = args.dataset_path
        self.json_path = args.json_path


class LDPIOptions:
    def __init__(self):
        # Initialize default values
        self.interface = 'enp5s0'
        self.n = 4
        self.l = 60

        # Threshold type
        self.threshold_type = 'max'

        # Parse the options
        self.parse_options()

    def parse_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Sniffing and buffering params
        parser.add_argument('--interface', default=self.interface, help='interface to sniff')
        parser.add_argument('--n', type=int, default=self.n, help='number of packets per sample')
        parser.add_argument('--l', type=int, default=self.l, help='size of each packet in the samples')

        # Anomaly detection sensitivity parameters
        parser.add_argument('--threshold_type', choices=['ninety_nine', 'near_max', 'max', 'hundred_one'], default=self.threshold_type,
                            help='Threshold strategy for anomaly detection. A higher threshold results in a lower False Alarm Rate (FAR).')

        args = parser.parse_args()

        # Update class attributes with parsed arguments
        self.interface = args.interface
        self.n = args.n
        self.l = args.l
        self.threshold_type = args.threshold_type
