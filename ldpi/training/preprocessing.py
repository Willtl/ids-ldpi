import os
from queue import Queue
from typing import Dict, Tuple, List

import dpkt
import numpy as np
from sklearn.metrics import accuracy_score

from ldpi.options import Options
from utils import SnifferSubscriber


class LDPIPreProcessing(SnifferSubscriber):
    def __init__(self) -> None:
        super(LDPIPreProcessing, self).__init__()
        self.args = Options().parse()
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

        # Early exit conditions
        if self._should_drop_packet(protocol, ip, flow_key, checked_flows):
            return

        # Extract packet bytes, anonymize if necessary
        ip_bytes = self._anonymize_packet(ip)

        # Manage flow
        flow = flows.setdefault(flow_key, [])
        flow.append(ip_bytes)

        # When a flow reaches the desired packet count (n)
        if len(flow) == self.args.n:
            checked_flows[flow_key] = True  # Mark flow as checked
            self.to_process.put((flow_key, flow))
            del flows[flow_key]  # Remove flow to free up memory

    def _anonymize_packet(self, ip: dpkt.ip.IP) -> bytes:
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

    def _should_drop_packet(self, protocol: int, ip: dpkt.ip.IP, flow_key: Tuple[bytes, int, bytes, int], checked_flows: Dict) -> bool:
        """
        Determines if a packet should be dropped based on various conditions.

        Parameters:
            protocol: The protocol number of the packet.
            ip: The IP layer of the packet.
            flow_key: The flow key associated with the packet.
            checked_flows: Dictionary tracking checked flows.

        Returns:
            A boolean indicating whether the packet should be dropped.
        """
        # Drop packet if it's part of a flow that's already processed
        if checked_flows.get(flow_key, False):
            return True

        # For TCP packets, drop on FIN or RST-ACK flags
        if protocol == 6:
            tcp_flags = ip.data.flags
            if tcp_flags & dpkt.tcp.TH_FIN or (tcp_flags & dpkt.tcp.TH_RST and tcp_flags & dpkt.tcp.TH_ACK):
                return True

        return False

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
        norm_flows = np.zeros([len(samples), sample_size], dtype=np.uint8)

        # Clean/anonymize packets, normalize and trim bytes and fill norm_flows
        for i in range(len(samples)):
            flow = samples[i]
            for j in range(self.args.n):
                ip = flow[j]
                anon_ip = ip[:12] + ip[20:]
                if len(anon_ip) > self.args.l:
                    anon_ip = anon_ip[:self.args.l]

                np_buff_tmp = np.frombuffer(bytes(anon_ip), dtype=np.uint8)

                # If smaller than l pad with zeros, trim otherwise
                if len(np_buff_tmp) < self.args.l:
                    np_buff = np.zeros(self.args.l)
                    np_buff[0:len(np_buff_tmp)] = np_buff_tmp
                else:
                    np_buff = np_buff_tmp[:self.args.l]

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
