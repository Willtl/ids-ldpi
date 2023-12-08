import glob
import logging
import os
from queue import Queue
from typing import Dict, Tuple, List

import dpkt
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from options import LDPIOptions, SnifferOptions
from sniffer.sniffer import Sniffer
from utils import SnifferSubscriber, sec_to_ns


class SnifferPcap(Sniffer):
    """
    SnifferPcap extends the Sniffer class to sniff packets from pcap files and processes them.

    Attributes:
        current_pcap_path: The file path of the current pcap file being processed.
        subscribers: A list of subscribers that will process the sniffed packets.
        max_samples_per_pcap: The maximum number of samples to process per pcap file.
    """

    def __init__(self, args: SnifferOptions) -> None:
        """
        Initializes the SnifferPcap instance with the given SnifferOptions.

        Parameters:
            args: Configuration options for the sniffer.
        """
        super().__init__(args)
        self.current_pcap_path: str = ''
        self.subscribers: List[LDPIPreProcessing] = []
        self.max_samples_per_pcap: int = 5000

    def sniff(self) -> None:
        """
        Sniffs packets from a pcap file and processes them.

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


class LDPIPreProcessing(SnifferSubscriber):
    def __init__(self) -> None:
        super(LDPIPreProcessing, self).__init__()
        self.args = LDPIOptions()
        self.flows_tcp: Dict[Tuple[bytes, int, bytes, int], List[bytes]] = {}
        self.flows_udp: Dict[Tuple[bytes, int, bytes, int], List[bytes]] = {}
        self.c_tcp: Dict[Tuple[bytes, int, bytes, int], bool] = {}
        self.c_udp: Dict[Tuple[bytes, int, bytes, int], bool] = {}
        self.to_process: Queue = Queue()
        self.sample_counter: int = 0
        self.current_path: str = ""

    def run(self) -> None:
        pass

    def new_packet(self, flow_key: Tuple[bytes, int, bytes, int], protocol: int, timestamp: int, ip: dpkt.ip.IP) -> None:
        """
        Processes a new packet and determines if it should be added to a flow for analysis.

        Parameters:
            flow_key: The unique identifier of the flow.
            protocol: The protocol number (e.g., TCP=6, UDP=17).
            timestamp: The timestamp of the packet.
            ip: The IP layer of the packet.
        """
        flows, checked_flows = self.get_buffers(protocol)

        # Drop packet if it's part of a flow that's already processed
        if checked_flows.get(flow_key, False):
            return

        # For TCP packets, check for FIN/RST flags
        if protocol == 6:
            if ip.data.flags & dpkt.tcp.TH_RST and ip.data.flags & dpkt.tcp.TH_ACK:
                self.teardown(flow_key, protocol)
                return

        # Extract packet bytes, anonymize if necessary
        ip_bytes = anonymize_packet(ip)

        # Manage flow
        flow = flows.setdefault(flow_key, [])
        flow.append(ip_bytes)

        # Check for FIN flag
        if protocol == 6 and (ip.data.flags & dpkt.tcp.TH_FIN):
            self.to_process.put((flow_key, flow))
            self.teardown(flow_key, protocol)
        elif len(flow) == self.args.n:
            # When a flow reaches the desired packet count (n)
            checked_flows[flow_key] = True
            self.to_process.put((flow_key, flow))
            del flows[flow_key]

    # Remove flows entries in case of teardown
    def teardown(self, flow_key: tuple, protocol: int) -> None:
        # Buffer of flows and checked flows of given protocol
        flows, checked_flows = self.get_buffers(protocol)
        removed_flows = flows.pop(flow_key, False)
        removed_checked_flows = checked_flows.pop(flow_key, False)
        # if removed_flows or removed_checked_flows:
        #     str_key = flow_key_to_str(flow_key)
        #     print(Color.UNDERLINE + f'{str_key} teardown ({self.to_process.qsize()})' + Color.ENDC)

    def preprocess_samples(self) -> np.ndarray:
        # Dequeue all samples for this iteration
        samples = []
        while not self.to_process.empty():
            value = self.to_process.get()
            samples.append(value[1])

        # Create 3-dimensional np array (flow->packet->byte)
        sample_size = self.args.n * self.args.l
        norm_flows = np.zeros([len(samples), sample_size], dtype=np.int32)

        # Clean/anonymize packets, normalize and trim bytes and fill norm_flows
        for i in range(len(samples)):
            flow = samples[i]
            flow_length = len(flow)

            for j in range(self.args.n):
                if j < flow_length:
                    # Process actual packets in the flow
                    packet = flow[j]
                    np_buff = trim_or_pad_packet(packet, self.args.l)
                else:
                    # Create zero byte packets for missing packets
                    np_buff = np.full(self.args.l, 256, dtype=np.int32)

                norm_flows[i][j * self.args.l: (j + 1) * self.args.l] = np_buff
        if False:
            print(f'Storing samples {norm_flows.shape[0]} samples at {self.current_path} - Sample counter: {self.sample_counter}')

        for i in range(norm_flows.shape[0]):
            np.save(f'{self.current_path}/{self.sample_counter}.npy', norm_flows[i])
            self.sample_counter += 1
            # print(norm_flows[i].shape)

    def perf_measure(self, y_true, scores) -> float:
        y_pred = np.empty_like(y_true)
        for i in range(len(y_true)):
            if scores[i] < self.threshold:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def get_buffers(self, protocol: int) -> tuple:
        """ Return the correct buffers """
        return (self.flows_tcp, self.c_tcp) if protocol == 6 else (self.flows_udp, self.c_udp)

    def set_current_path(self, storing_path: str):
        self.flows_tcp: Dict[tuple, int] = {}
        self.flows_udp: Dict[tuple, int] = {}
        self.c_tcp: Dict[tuple, int] = {}
        self.c_udp: Dict[tuple, int] = {}
        self.to_process: Queue = Queue()
        self.sample_counter = 0
        self.current_path = f'samples/{storing_path}'

        if not os.path.isdir(self.current_path):
            os.makedirs(self.current_path)


def anonymize_packet(ip: dpkt.ip.IP) -> bytes:
    """
    Anonymizes an IP packet by removing certain bytes.

    Parameters:
        ip: The IP layer of the packet.

    Returns:
        Anonymized bytes of the packet.
    """
    ip_bytes = bytes(ip)

    # IP header is the first 20 bytes of the IP packet
    # Source IP is at bytes 12-15 and Destination IP is at bytes 16-19
    # We remove these bytes to anonymize the packet
    anonymized_ip_bytes = ip_bytes[:12] + ip_bytes[20:]

    return anonymized_ip_bytes


def trim_or_pad_packet(packet: bytes, length: int) -> np.ndarray:
    """
    Trims or pads the packet to the desired length.

    Args:
        packet (bytes): The packet to process.
        length (int): Desired length of the packet.

    Returns:
        np.ndarray: The processed packet.
    """
    # Trim to length in case packet is larger than length
    if len(packet) > length:
        packet = packet[:length]

    np_buff = np.frombuffer(bytes(packet), dtype=np.uint8).astype(np.int32)

    # If smaller than the desired length, pad with zeros
    if len(np_buff) < length:
        padded_np_buff = np.full(length, 256, dtype=np.int32)
        padded_np_buff[0:len(np_buff)] = np_buff
        return padded_np_buff

    return np_buff


def main():
    # Process PCAP into network flow samples
    sniffer_args = SnifferOptions()

    # Configure logging based on the debug flag
    if sniffer_args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    fsnf = SnifferPcap(sniffer_args)
    fsnf.subscribers.append(LDPIPreProcessing())
    fsnf.run(daemon=False)


if __name__ == '__main__':
    main()
