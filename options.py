import argparse


class Options:
    def __init__(self):
        self.dataset_path: str = 'data/TII-SSRC-23'
        self.json_path: str = ''
        self.precision_floats: int = 4
        self.delay: bool = False
        self.session: bool = False
        self.aggregate_csvs: bool = True
        self.interface: str = 'enp5s0'
        self.timeout: int = 60
        self.cleaning_cycle: int = 10
        self.time_window: int = 60
        self.parse_options()

    def parse_options(self) -> 'Options':
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

        return self
