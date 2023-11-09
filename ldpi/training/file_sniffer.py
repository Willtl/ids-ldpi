import logging
import os
import time
from typing import List

import dpkt
from tqdm import tqdm

from ldpi.training.preprocessing import LDPIPreProcessing
from options import Options
from sniffer.sniffer import Sniffer
from utils import get_dataset_local, sec_to_ns


class SnifferPcap(Sniffer):
    def __init__(self, args: Options):
        super(SnifferPcap, self).__init__(args)
        self.current_pcap_path: str = ''
        self.subscribers: List[LDPIPreProcessing] = []
        self.max_samples_per_pcap: int = 10000

    def set_pcap_path(self, path: str):
        """
        Sets the path of the current pcap file to process.
        """
        self.current_pcap_path = path

    def sniff(self, log_periodicity: int = 10000) -> None:
        """
        Sniffs packets from a pcap file and processes them.
        """
        self.flows_tcp: dict = {}
        self.flows_udp: dict = {}

        dataset_name, benign, malware = get_dataset_local()
        # dataset_name, benign, malware = get_dataset_debug()
        logging.basicConfig(level=logging.INFO)
        logging.info(f'{dataset_name}, {benign}, {malware}')

        for pcap_path in benign + malware:
            dir_only = os.path.dirname(pcap_path)
            if 'TII-SSRC-23' in dir_only:
                index = dir_only.find('TII-SSRC-23')
                filename = os.path.basename(pcap_path).split('.pcap')[0]
                storing_path: str = dir_only[index:]
                self.current_pcap_path = f'{storing_path}/{filename}/'
                self.subscribers[0].set_current_path(self.current_pcap_path)

            with open(pcap_path, 'rb') as file:
                logging.info(f'Preprocessing {self.current_pcap_path}')
                pcap_file = list(dpkt.pcap.Reader(file))
                time.sleep(0.1)  # Sleep so IO does not overlap with tqdm
                for i, (ts, buf) in tqdm(enumerate(pcap_file), desc=f'Processing {pcap_path}', total=len(pcap_file)):
                    ts_ns = sec_to_ns(ts)
                    self.process_packet(ts_ns, buf)

                    if not self.subscribers[0].to_process.empty():
                        self.subscribers[0].preprocess_samples()
                        if self.subscribers[0].sample_counter > self.max_samples_per_pcap:
                            break
