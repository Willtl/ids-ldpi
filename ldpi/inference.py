import os
import threading
import time
from queue import Queue
from typing import Dict, Set, Optional, Tuple, List

import dpkt
import numpy as np
import torch

from ldpi.training.preprocessing import anonymize_packet, trim_or_pad_packet
from options import LDPIOptions
from utils import SnifferSubscriber, Color, flow_key_to_str

# Define aliases
FlowKeyType = Tuple[bytes, int, bytes, int]


class TrainedModel:
    """
    TrainedModel encapsulates the functionality related to loading and managing the
    trained deep learning model for network anomaly detection.

    Attributes:
        threshold_type (str): Sensitivity for the anomaly detection.
        store_models_path (str): Path to the directory where the model and its parameters are stored.
        quantized (bool): Flag indicating if the model is quantized.
        backend (str): The backend to be used for the quantized model.
        model (Optional[torch.jit.ScriptModule]): The loaded PyTorch model, None until loaded.
        center (Optional[torch.Tensor]): Center of the hypersphere.
        nn_threshold (Optional[float]): 99th percentile threshold for anomaly detection.
        max_threshold (Optional[float]): Maximum threshold for anomaly detection.
        chosen_threshold float: Chosen threshold given arguments.
    """

    def __init__(self, threshold_type: str = 'max', quantized: bool = False, backend: str = 'qnnpack', store_models_path: str = 'ldpi/training/output'):
        """
        Initializes the TrainedModel instance with specified parameters.

        Args:
            store_models_path (str): Path to the model storage directory.
            quantized (bool): Specifies if the model is quantized.
            backend (str): Backend used for the quantized model.
        """

        self.threshold_type: str = threshold_type
        self.store_models_path: str = store_models_path
        self.quantized: bool = quantized
        self.backend: str = backend
        self.model: Optional[torch.jit.ScriptModule] = None
        self.center: Optional[torch.Tensor] = None
        self.ninety_nine_threshold: Optional[float] = None
        self.near_max_threshold: Optional[float] = None
        self.max_threshold: Optional[float] = None
        self.hundred_one_threshold: Optional[float] = None

        self._init_model()
        self.chosen_threshold: float = self._initialize_threshold()

    def _init_model(self) -> None:
        """
        Loads the model and its parameters from the specified storage path.
        Handles both quantized and non-quantized models.
        """

        try:
            print('Loading best model and center from saved state')
            # Load model from disk
            model_save_path = os.path.join(self.store_models_path, 'best_model_with_center.pth')
            if self.quantized:
                print('Loading quantized model')
                if self.backend == 'qnnpack':
                    torch.backends.quantized.engine = 'qnnpack'
                quantized_model_path = os.path.join(self.store_models_path, 'scripted_quantized_model.pth')
                self.model = torch.jit.load(quantized_model_path, map_location=torch.device('cpu'))
            else:
                print('Loading traced model')
                traced_model_path = os.path.join(self.store_models_path, 'traced_model.pth')
                self.model = torch.jit.load(traced_model_path, map_location=torch.device('cpu'))

            # Set model to eval mode
            self.model.eval()

            # Load center and thresholds
            saved_state = torch.load(model_save_path, map_location=torch.device('cpu'))
            self.center = saved_state['center']
            self.ninety_nine_threshold = saved_state['results']['ninety_nine']['threshold']
            self.near_max_threshold = saved_state['results']['near_max']['threshold']
            self.max_threshold = saved_state['results']['max']['threshold']
            self.hundred_one_threshold = saved_state['results']['hundred_one']['threshold']
        except Exception as e:
            print(f'Exception loading model: {e}')
            raise

    def _initialize_threshold(self) -> float:
        """
        Initializes the threshold based on the threshold_type.
        """
        if self.threshold_type == 'ninety_nine':
            return self.ninety_nine_threshold
        elif self.threshold_type == 'near_max':
            return self.near_max_threshold
        elif self.threshold_type == 'max':
            return self.max_threshold
        elif self.threshold_type == 'hundred_one':
            return self.hundred_one_threshold
        else:
            raise ValueError(f"Invalid threshold_type: {self.threshold_type}. Supported values are 'max' and 'nn'.")

    def prepare_tensors(self, to_process: Queue) -> Tuple[List[FlowKeyType], torch.Tensor]:
        """
        Prepares tensors for processing by concatenating numpy arrays into torch tensors.

        This method extracts items from a queue, concatenates numpy arrays into tensors,
        and stacks all tensors into a single tensor. It returns a list of keys and a stacked tensor.

        Returns:
            Tuple[List[FlowKeyType], torch.Tensor]: A tuple containing a list of keys and a stacked tensor.
        """

        keys, samples = [], []
        while not to_process.empty():
            flow_key, np_flows = to_process.get()  # Retrieve an item from the queue
            keys.append(flow_key)  # Append the key to the keys list

            # Normalize and convert concatenated numpy arrays to a torch tensor with float32 data type
            concatenated_tensor = torch.tensor(np.concatenate(np_flows) / 255.0, dtype=torch.float32)
            samples.append(concatenated_tensor)  # Append the tensor to the samples list

        # Stack all sample tensors along the first dimension if there are any samples
        if samples:
            samples = torch.stack(samples)
        return keys, samples

    def infer(self, network_flows: torch.Tensor) -> torch.Tensor:
        """
        Analyzes network flows to detect anomalies based on a pre-trained model and a defined threshold.

        Args:
            network_flows (torch.Tensor): A tensor representing network flow data.

        Returns:
            torch.Tensor: A tensor of booleans, each indicating whether the corresponding network flow is anomalous (True) or not (False).
        """
        # Add an extra dimension to the network_flows tensor for processing
        network_flows = network_flows.unsqueeze(1)

        # Use the pre-trained model to encode the network flows into an embedding
        embedding = self.model.encode(network_flows)

        # Calculate the Euclidean distance of each embedding from a central point
        distance = torch.sum((embedding - self.center) ** 2, dim=1)

        # Compare each distance with the chosen_threshold to determine anomalies
        anomaly_array = (distance >= self.chosen_threshold)

        return anomaly_array


class LightDeepPacketInspection(SnifferSubscriber):
    def __init__(self) -> None:
        super(LightDeepPacketInspection, self).__init__()

        # Initialize LDPIOptions
        self.args = LDPIOptions()

        # Initialize the trained model with specified parameters
        self.trained_model = TrainedModel(threshold_type=self.args.threshold_type, quantized=False)

        # Sniffer related attributes
        self.flows_tcp: Dict[FlowKeyType, List[np.ndarray]] = {}
        self.flows_udp: Dict[FlowKeyType, List[np.ndarray]] = {}
        self.checked_tcp: Set[FlowKeyType] = set()
        self.checked_udp: Set[FlowKeyType] = set()
        self.black_list: Set[bytes] = set()
        self.to_process: Queue = Queue()

        # Set up threading to run sniff() in a separate thread
        self.thread = threading.Thread(target=self.analyze_flows)
        self.thread.daemon = True

    def run(self) -> None:
        self.thread.start()

    def analyze_flows(self):
        while not self.stopped():
            keys, samples = self.trained_model.prepare_tensors(self.to_process)
            if keys:
                anomalies = self.trained_model.infer(samples)

                # Process each key and corresponding anomaly detection result
                for key, is_anomaly in zip(keys, anomalies):
                    if is_anomaly:
                        # If anomaly detected, add the source IP to the blacklist
                        self.black_list.add(key[0])
                        print(Color.FAIL + f'Anomaly detected in flow {flow_key_to_str(key)}, LDPI blacklisted (dropping packets from): {key[0]}' + Color.ENDC)
                    else:
                        print(Color.OKGREEN + f'No anomaly detected in flow {flow_key_to_str(key)}' + Color.ENDC)
            time.sleep(0.1)

    def new_packet(self, flow_key: tuple, protocol: int, timestamp: int, ip: dpkt.ip.IP) -> None:
        # Drop all packets in case src_ip is on black list
        if flow_key[0] in self.black_list:
            return

        # Buffer of flows and checked flows of given protocol
        flows, checked_flows = self.get_buffers(protocol)

        # Drop packet if it's part of a flow that's already processed
        if flow_key in checked_flows:
            return

        # For TCP packets, check for FIN & RST flags
        if protocol == 6:
            if ip.data.flags & dpkt.tcp.TH_RST and ip.data.flags & dpkt.tcp.TH_ACK:
                self.teardown(flow_key, protocol)
                return

        # Extract packet bytes, anonymize IPs
        ip_bytes = anonymize_packet(ip)

        # Trim and/or pad packet
        np_packet = trim_or_pad_packet(ip_bytes, self.args.l)

        # Append packet
        flow = flows.setdefault(flow_key, [])
        flow.append(np_packet)

        # Queue to analysis on FIN or when a flow reaches the desired packet count 'n'
        if protocol == 6 and (ip.data.flags & dpkt.tcp.TH_FIN):
            print('put from FIN')
            self.to_process.put((flow_key, flow))
            self.teardown(flow_key, protocol)
            # str_key = flow_key_to_str(flow_key)
            # print(Color.OKBLUE + f'{str_key} queued for detection (FIN) ({self.to_process.qsize()})' + Color.ENDC)
        elif len(flow) == self.args.n:
            checked_flows.add(flow_key)
            self.to_process.put((flow_key, flow))
            del flows[flow_key]
            # str_key = flow_key_to_str(flow_key)
            # print(Color.OKBLUE + f'{str_key} queued for detection (RDY) ({self.to_process.qsize()})' + Color.ENDC)

    # Remove flows entries in case of teardown
    def teardown(self, flow_key: FlowKeyType, protocol: int) -> None:
        """
        Remove flow entries in case of teardown.

        This method removes entries related to a flow identified by its flow_key from the respective buffers based on the
        protocol (TCP or UDP).

        Args:
            flow_key (FlowKeyType): A tuple representing the flow key.
            protocol (int): The protocol number (6 for TCP, 17 for UDP).
        """
        # Determine the appropriate buffers based on the protocol
        flows, checked_flows = self.get_buffers(protocol)

        # Remove the flow entry from the flows and checked_flows sets
        removed_flows = flow_key in flows and flows.pop(flow_key)
        removed_checked_flows = flow_key in checked_flows and checked_flows.remove(flow_key)

        # if removed_flows or removed_checked_flows:
        #     print('Teardown LDPI')

    def get_buffers(self, protocol: int) -> Tuple[Dict[FlowKeyType, List[np.ndarray]], Set[FlowKeyType]]:
        """ Return the correct buffers given protocol number """
        return (self.flows_tcp, self.checked_tcp) if protocol == 6 else (self.flows_udp, self.checked_udp)
