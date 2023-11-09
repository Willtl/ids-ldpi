import os
from queue import Queue
from typing import Dict

import dpkt
import numpy as np
from sklearn.metrics import accuracy_score

from ldpi.options import Options
from utils import SnifferSubscriber, Color, flow_key_to_str


class LDPIPreProcessing(SnifferSubscriber):

    def __init__(self):
        super(LDPIPreProcessing, self).__init__()
        self.args = Options().parse()
        self.flows_tcp: Dict[tuple, int] = {}
        self.flows_udp: Dict[tuple, int] = {}
        self.c_tcp: Dict[tuple, int] = {}
        self.c_udp: Dict[tuple, int] = {}
        self.to_process: Queue = Queue()
        self.sample_counter = 0
        self.current_path = ""

    def run(self) -> None:
        pass

    def new_packet(self, flow_key: tuple, protocol: int, timestamp: int, ip: dpkt.ip.IP) -> None:
        # Buffer of flows and checked flows of given protocol
        flows, c_flows = self.get_buffers(protocol)

        # Remove flow data from memory and drop FIN packet
        if protocol == 6 and ip.data.flags & dpkt.tcp.TH_FIN:
            self.teardown(flow_key, protocol)
            return

        # Drop packet in case packet's flow is already checked
        if c_flows.get(flow_key, False):
            return

        # Drop in case of reset ack
        if protocol == 6 and ip.data.flags & dpkt.tcp.TH_RST and ip.data.flags & dpkt.tcp.TH_ACK:
            return

        # Remove IP bytes from packet to anonymize it (from 12th to 20th octet)
        ip_bytes = bytes(ip)

        # Create/append packet to the flow buffer (this assumes that args.n is higher than 1)
        flow = flows.get(flow_key, False)
        if flow:
            flow.append(ip_bytes)
        else:
            flows[flow_key] = [ip_bytes]
            flow = flows[flow_key]

        # Check if data is enough to process it
        if len(flow) == self.args.n:
            c_flows[flow_key] = True
            self.to_process.put((flow_key, flow))
            flows.pop(flow_key, False)

            # Print
            if False:
                str_key = flow_key_to_str(flow_key)
                print(Color.OKBLUE + f'{str_key} waiting detection ({self.to_process.qsize()})' + Color.ENDC)

    # Remove flows entries in case of teardown
    def teardown(self, flow_key: tuple, protocol: int) -> None:
        # Buffer of flows and checked flows of given protocol
        flows, c_flows = self.get_buffers(protocol)
        removed_flows = flows.pop(flow_key, False)
        removed_c_flows = c_flows.pop(flow_key, False)
        # if removed_flows or removed_c_flows:
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
