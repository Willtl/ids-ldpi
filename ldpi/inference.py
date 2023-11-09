import errno
import json
import math
import socket
import sys
import time
from queue import Queue
from typing import Dict, Tuple, Set, Any

import dpkt
import numpy
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from ldpi.networks import MLP
from ldpi.options import Options
from utils import SnifferSubscriber, Color, flow_key_to_str


class LightDeepPacketInspection(SnifferSubscriber):

    def __init__(self):
        super(LightDeepPacketInspection, self).__init__()
        self.args = Options().parse()

        # LDPI related atributes
        self.model: MLP = None
        self.backend = 'qnnpack'
        self.center = None
        self.device: str = 'cpu'
        self.threshold: float = 0.0
        self.model_loaded: bool = False

        # Sniffer related attributes
        self.flows_tcp: Dict[tuple, int] = {}
        self.flows_udp: Dict[tuple, int] = {}
        self.c_tcp: Dict[tuple, int] = {}
        self.c_udp: Dict[tuple, int] = {}
        self.black_list: Set = set()
        self.to_process: Queue = Queue()

        # Initialize model and Pytorch stuff
        self.model_loaded = self.init_model()

        # Socket for demo
        # self.socket = None
        # self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # host = "192.168.1.2"
        # port = 8800
        # self.socket.connect((host, port))
        # self.socket.setblocking(0)

    def init_model(self) -> bool:
        print('init model')
        try:
            self.center = torch.from_numpy(np.load('ldpi/trained_model/center.npy'))
            self.threshold = np.load('ldpi/trained_model/threshold.npy')

            # Instantiate model architecture without trained weightsnan
            self.model = MLP(input_size=self.args.n * self.args.l, num_features=512, rep_dim=50)
            if self.backend == 'qnnpack':
                torch.backends.quantized.engine = 'qnnpack'
                self.model = torch.jit.load('ldpi/trained_model/quantized.pt')
            else:
                self.model.load_state_dict(
                    torch.load(f'ldpi/trained_model/trained_model.pt', map_location=torch.device(self.device))
                )
            self.model.eval()
            return True
        except Exception as e:
            print(f'Exception loading model {e}')
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

    def preprocess_samples(self) -> Tuple[list, list]:
        # Dequeue all samples for this iteration
        keys, samples = [], []
        while not self.to_process.empty():
            value = self.to_process.get()
            keys.append(value[0])
            samples.append(value[1])

        # Start preprocessing for the samples set
        print(Color.OKCYAN + f'Pre-processing {len(samples)} flows' + Color.ENDC)

        # Create 3-dimensional np array (flow->packet->byte)
        sample_size = self.args.n * self.args.l
        norm_flows = np.zeros([len(samples), sample_size], dtype=np.float32)

        # Clean/anonymize packets, normalize and trim bytes and fill norm_flows
        for i in range(len(samples)):
            flow = samples[i]
            for j in range(self.args.n):
                ip = flow[j]
                anon_ip = ip[:12] + ip[20:]
                if len(anon_ip) > self.args.l:
                    anon_ip = anon_ip[:self.args.l]

                np_buff_tmp = np.array(np.frombuffer(bytes(anon_ip), dtype=np.uint8) / 255.0, dtype=np.float32)
                # If smaller than l pad with zeros, trim otherwise
                if len(np_buff_tmp) < self.args.l:
                    np_buff = np.zeros(self.args.l)
                    np_buff[0:len(np_buff_tmp)] = np_buff_tmp
                else:
                    np_buff = np_buff_tmp[:self.args.l]

                norm_flows[i][j * self.args.l: (j + 1) * self.args.l] = np_buff

        with torch.no_grad():
            tensor = torch.from_numpy(norm_flows)

        return keys, tensor

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

    def get_buffers(self, protocol: str) -> tuple:
        """ Return the correct buffers """
        return (self.flows_tcp, self.c_tcp) if protocol == 'tcp' else (self.flows_udp, self.c_udp)

    def report(self, json_message: Dict[str, Any]) -> None:
        if self.socket is not None:
            message = (json.dumps(json_message) + '\n').encode()
            self.socket.sendall(message)

            try:
                data = self.socket.recv(1024).decode()
                if data == 'close':
                    self.socket.close()
                    self.socket = None
                    print('server is closing')
            except socket.error as e:
                err = e.args[0]
                if not (err == errno.EAGAIN or err == errno.EWOULDBLOCK):
                    print('Socket error', e)
