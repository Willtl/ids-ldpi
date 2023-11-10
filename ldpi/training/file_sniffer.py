import glob
import logging
import os
from typing import List

import dpkt
from tqdm import tqdm

from ldpi.training.preprocessing import LDPIPreProcessing
from options import Options
from sniffer.sniffer import Sniffer
from utils import sec_to_ns


class SnifferPcap(Sniffer):
    """
    SnifferPcap extends the Sniffer class to sniff packets from pcap files and processes them.

    Attributes:
        current_pcap_path: The file path of the current pcap file being processed.
        subscribers: A list of subscribers that will process the sniffed packets.
        max_samples_per_pcap: The maximum number of samples to process per pcap file.
    """

    def __init__(self, args: Options) -> None:
        """
        Initializes the SnifferPcap instance with the given options.

        Parameters:
            args: Configuration options for the sniffer.
        """
        super().__init__(args)
        self.current_pcap_path: str = ''
        self.subscribers: List[LDPIPreProcessing] = []
        self.max_samples_per_pcap: int = 10000

    def sniff(self, log_periodicity: int = 10000) -> None:
        """
        Sniffs packets from a pcap file and processes them.

        Parameters:
            log_periodicity: How often to log progress (default is 10000 packets).

        Raises:
            ValueError: If subscribers list is empty.
        """
        if not self.subscribers:
            raise ValueError("Subscribers list cannot be empty.")

        dataset_name, benign, malware = self._get_dataset_info()
        logging.basicConfig(level=logging.INFO)

        for pcap_path in benign + malware:
            self._set_current_pcap_path(pcap_path)
            self._process_pcap_file(pcap_path)

    def _get_dataset_info(self) -> tuple:
        """
        Retrieves dataset information.

        Returns:
            A tuple containing the dataset name, list of benign pcap paths, and list of malware pcap paths.

        Raises:
            FileNotFoundError: If no pcap files are found in the specified dataset directories.
        """

        # Resolve the relative path to an absolute path
        dataset_name = os.path.abspath(self.args.dataset_path)

        # Path for benign pcap files, recursively search through subdirectories
        benign_path = os.path.join(dataset_name, 'benign', '**', '*.pcap')
        benign_files = glob.glob(benign_path, recursive=True)

        # Path for malware pcap files, recursively search through subdirectories
        malicious_path = os.path.join(dataset_name, 'malicious', '**', '*.pcap')
        malicious_files = glob.glob(malicious_path, recursive=True)

        if not benign_files:
            raise FileNotFoundError(f"No benign pcap files found in directory: {benign_path}")
        if not malicious_files:
            raise FileNotFoundError(f"No malicious pcap files found in directory: {malicious_path}")

        # Extract the dataset name from the path for return value
        dataset_basename = os.path.basename(dataset_name)

        return dataset_basename, benign_files, malicious_files

    def _set_current_pcap_path(self, pcap_path: str) -> None:
        """
        Sets the current pcap path based on the directory name and pcap filename.

        Parameters:
            pcap_path: The path to the pcap file.
        """
        dir_only = os.path.dirname(pcap_path)
        if 'TII-SSRC-23' in dir_only:
            index = dir_only.find('TII-SSRC-23')
            filename = os.path.basename(pcap_path).split('.pcap')[0]
            storing_path = dir_only[index:]
            self.current_pcap_path = f'{storing_path}/{filename}/'
            self.subscribers[0].set_current_path(self.current_pcap_path)

    def _process_pcap_file(self, pcap_path: str) -> None:
        """
        Processes the pcap file and pre-processes the packets.

        Parameters:
            pcap_path: The path to the pcap file.
        """
        self.flows_tcp.clear()
        self.flows_udp.clear()

        with open(pcap_path, 'rb') as file:
            logging.info(f'Preprocessing {self.current_pcap_path}')
            pcap_file = list(dpkt.pcap.Reader(file))  # convert to list so tqdm progress bar works

            for i, (ts, buf) in tqdm(enumerate(pcap_file), desc=f'Processing {pcap_path}', total=len(pcap_file)):
                ts_ns = sec_to_ns(ts)
                self.process_packet(ts_ns, buf)
                sample_counter = self._handle_subscriber_processing()

                if sample_counter > self.max_samples_per_pcap:
                    break

    def _handle_subscriber_processing(self) -> int:
        """
        Handles the processing of packets by the subscriber.

        Returns:
            The number of processed flows.
        """
        subscriber = self.subscribers[0]
        if not subscriber.to_process.empty():
            subscriber.preprocess_samples()

        return subscriber.sample_counter
