import math
import sys
import time
from queue import Queue
from typing import Dict, Set, Optional
from typing import Tuple, List

import dpkt
import numpy
import numpy as np
import torch
from torch.nn import Module

from ldpi.networks import MLP
from ldpi.options_ldpi import LDPIOptions
from utils import SnifferSubscriber, Color, flow_key_to_str


class LightDeepPacketInspection(SnifferSubscriber):
    def __init__(self) -> None:
        super(LightDeepPacketInspection, self).__init__()
        self.args = LDPIOptions().parse()

        # LDPI related attributes
        self.model: Optional[Module] = None
        self.backend: str = 'qnnpack'
        self.center: Optional[torch.Tensor] = None
        self.device: str = 'cpu'
        self.threshold: float = 0.0
        self.model_loaded: bool = False

        # Sniffer related attributes
        self.flows_tcp: Dict[Tuple[bytes, int, bytes, int], int] = {}
        self.flows_udp: Dict[Tuple[bytes, int, bytes, int], int] = {}
        self.c_tcp: Dict[Tuple[bytes, int, bytes, int], int] = {}
        self.c_udp: Dict[Tuple[bytes, int, bytes, int], int] = {}
        self.black_list: Set[Tuple[bytes, int, bytes, int]] = set()
        self.to_process: Queue = Queue()

        self.model_loaded = self.init_model()

    def init_model(self) -> bool:
        print('Initializing model...')
        try:
            self.center = torch.from_numpy(np.load('ldpi/trained_model/center.npy'))
            self.threshold = np.load('ldpi/trained_model/threshold.npy')

            self.model = MLP(input_size=self.args.n * self.args.l, num_features=512, rep_dim=50)
            if self.backend == 'qnnpack':
                torch.backends.quantized.engine = 'qnnpack'
                self.model = torch.jit.load('ldpi/trained_model/quantized.pt')
            else:
                self.model.load_state_dict(torch.load('ldpi/trained_model/trained_model.pt', map_location=torch.device(self.device)))
            self.model.eval()
            return True
        except Exception as e:
            print(f'Exception loading model: {e}')
            return False

    def run(self) -> None:
        try:
            while not self.stopped():
                # Continue in case queue is empty
                if self.to_process.empty() or not self.model_loaded:
                    time.sleep(1.0)
                    continue

                keys, samples = self.preprocess_samples()
                anomaly_scores = self.infer(samples)
                json_data = {'module': 'ldpi', 'normal': 0, 'anomaly': 0}
                for i, score in enumerate(anomaly_scores):
                    str_flow_key = flow_key_to_str(keys[i])
                    if keys[i] not in self.black_list:
                        if score > self.threshold:
                            sqrt = math.pow(score, 2)
                            confidence = round((sqrt - self.threshold) / sqrt * 100, 2)
                            print(Color.BOLD + Color.FAIL +
                                  f'Malicious flow detected {str_flow_key}, '
                                  f'Distance to the center: {anomaly_scores[i]}, '
                                  f'Anomaly confidence: {confidence:.2f}%' + Color.ENDC)

                            json_data['anomaly'] += 1

                            # self.black_list.add(keys[i])
                            # print(Color.BOLD + Color.FAIL +
                            #       f'Adding {str_flow_key} to blacklist...'
                            #       f'Blacklist size: {len(self.black_list)}'
                            #       + Color.ENDC)
                        else:
                            confidence = round((1.0 - (self.threshold - score) / self.threshold) * 100, 2)
                            print(Color.OKGREEN + f'Benign flow {str_flow_key}, '
                                                  f'Distance to the center: {anomaly_scores[i]}, '
                                                  f'Anomaly confidence: {confidence:.2f}%' + Color.ENDC)
                            json_data['normal'] += 1

                # self.report(json_data)

                time.sleep(self.args.detect_interval)
        except (KeyboardInterrupt, SystemExit):
            print('Stopping LDPI')
            self.stop()
            sys.exit()

    def new_packet(self, flow_key: tuple, protocol: str, timestamp: int, ip: dpkt.ip.IP) -> None:
        # Drop all packets in case src_ip is on black list
        if flow_key in self.black_list:
            return

        # Drop in case of reset ack
        if protocol == 'tcp' and ip.data.flags & dpkt.tcp.TH_RST and ip.data.flags & dpkt.tcp.TH_ACK:
            return

        # Buffer of flows and checked flows of given protocol
        flows, c_flows = self.get_buffers(protocol)

        # Remove flow data from memory and drop FIN packet
        if protocol == 'tcp' and ip.data.flags & dpkt.tcp.TH_FIN:
            print('FIN')
            self.teardown(flow_key, protocol)
            return

        # Drop packet in case packet's flow is already checked
        if c_flows.get(flow_key, False):
            return

        # Remove IP bytes from packet to anonymize it (from 12th to 20th octet)
        ip_bytes = bytes(ip)

        # np_anon_bytes = np.array(np.frombuffer(anon_ip_bytes, dtype=np.uint8) / 255.0, dtype='float32')

        # Create/append packet to the flow buffer (this assumes that args.n is higher than 1)
        flow = flows.get(flow_key, False)
        if flow:
            flow.append(ip_bytes)

            # Check if data is enough to process it
            if len(flow) == self.args.n:
                c_flows[flow_key] = True
                self.to_process.put((flow_key, flow))

                # Print
                str_key = flow_key_to_str(flow_key)
                print(Color.OKBLUE + f'{str_key} waiting detection ({self.to_process.qsize()})' + Color.ENDC)
        else:
            flows[flow_key] = [ip_bytes]

    # Remove flows entries in case of teardown
    def teardown(self, flow_key: tuple, protocol: str) -> None:
        # Buffer of flows and checked flows of given protocol
        flows, c_flows = self.get_buffers(protocol)

        removed_flows = flows.pop(flow_key, False)
        removed_c_flows = c_flows.pop(flow_key, False)
        if removed_flows or removed_c_flows:
            print('teardown ldpi')

    def preprocess_samples(self, raw_flows: List[List[bytes]]) -> Tuple[List[Tuple[bytes, int, bytes, int]], torch.Tensor]:
        """
        Preprocess the raw network flow samples for the neural network.

        Args:
            raw_flows (List[List[bytes]]): A list of flows, each a list of packet data as bytes.

        Returns:
            Tuple containing:
                - A list of flow keys.
                - A tensor of normalized and preprocessed flow data.
        """
        # Assuming self.args.n is the number of packets per flow
        # and self.args.l is the number of bytes to consider from each packet
        sample_size = self.args.n * self.args.l
        norm_flows = np.zeros([len(raw_flows), sample_size], dtype=np.float32)

        flow_keys = []
        for i, flow in enumerate(raw_flows):
            flow_key, packets = flow
            flow_keys.append(flow_key)
            preprocessed_packets = []

            for packet in packets:
                # Remove IP headers to anonymize
                anon_packet = self._anonymize_ip(packet)
                # Ensure the packet is of the length expected by the model
                packet = self._trim_or_pad_packet(anon_packet, self.args.l)
                preprocessed_packets.append(packet)

            # Flatten the list of packets into a single 1D array per flow
            norm_flows[i] = np.concatenate(preprocessed_packets, axis=0)

        # Normalize the flows by scaling byte values to the range [0, 1]
        norm_flows /= 255.0

        # Convert to a PyTorch tensor
        tensor_flows = torch.tensor(norm_flows, dtype=torch.float32)

        return flow_keys, tensor_flows

    def _anonymize_ip(self, packet: bytes) -> bytes:
        """
        Remove sensitive information from the packet data.

        Args:
            packet (bytes): The raw packet data.

        Returns:
            The anonymized packet data.
        """
        # Assuming IPv4 and the IP header is always the first 20 bytes of the packet
        # Remove first 12 bytes (version, IHL, TOS, Total Length, Identification, Flags, Fragment Offset)
        # and the next 8 bytes (TTL, Protocol, Header Checksum, Source IP, Destination IP)
        return packet[:12] + packet[20:]

    def _trim_or_pad_packet(self, packet: bytes, desired_length: int) -> np.ndarray:
        """
        Trim or pad the packet data to the desired length.

        Args:
            packet (bytes): The packet data to trim or pad.
            desired_length (int): The desired length of the packet data.

        Returns:
            The packet data as an ndarray, trimmed or padded to the desired length.
        """
        # Convert packet data to a NumPy array
        packet_array = np.frombuffer(packet, dtype=np.uint8)
        # Trim packet if it's longer than the desired length
        if len(packet_array) > desired_length:
            return packet_array[:desired_length]
        # If packet is shorter, pad with zeros
        elif len(packet_array) < desired_length:
            padded_array = np.zeros(desired_length, dtype=np.uint8)
            padded_array[:len(packet_array)] = packet_array
            return padded_array
        else:
            return packet_array

    def infer(self, samples) -> numpy.ndarray:
        with torch.no_grad():
            # Move the data to the device
            samples = samples.to(self.device)

            if self.backend == 'qnnpack':
                outputs = self.model(samples)
            else:
                outputs = self.model.encode(samples)

            anomaly_score = torch.sum((outputs - self.center) ** 2, dim=1)
            anomaly_score = anomaly_score.to('cpu').numpy()

        return anomaly_score

    def get_buffers(self, protocol: str) -> tuple:
        """ Return the correct buffers """
        return (self.flows_tcp, self.c_tcp) if protocol == 'tcp' else (self.flows_udp, self.c_udp)
