import datetime
import glob
import os
import socket
import threading
from abc import ABC, abstractmethod
from enum import Enum

import dpkt
import matplotlib.pyplot as plt
import numpy as np
from dpkt.compat import compat_ord
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ModuleInterface(ABC, threading.Thread):
    """ Basic abstract class for each feature's module """

    def __init__(self):
        super(ModuleInterface, self).__init__()
        self.stop_event = threading.Event()
        self.thread: threading.Thread = None

    @abstractmethod
    def run(self) -> None:
        self.thread.start()

    def terminate(self) -> None:
        del self

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()

    def stopped(self) -> bool:
        return self.stop_event.is_set()


class SnifferSubscriber(ModuleInterface):
    """ Abstract class that must be implemented for any module that subscribes to the sniffer module """

    def __init__(self):
        super(SnifferSubscriber, self).__init__()

    @abstractmethod
    def new_packet(self, flow_key: tuple, protocol: int, timestamp: int, ip: dpkt.ip.IP) -> None:
        """ Sniffer will call new_packet in case of sniffed packet """
        print('new packet')

    @abstractmethod
    def teardown(self, flow_key: tuple, protocol: int) -> None:
        """ Sniffer will call tear_down in case of flow timeout """


class Dataset(ABC):
    def __init__(self):
        self.path = ''

    @abstractmethod
    def get_files(self):
        pass


class USTC(Dataset):

    @staticmethod
    def get_files():
        path = 'USTC-TFC2016'
        benign = ['Benign/BitTorrent', 'Benign/Facetime', 'Benign/FTP', 'Benign/Gmail', 'Benign/MySQL',
                  'Benign/Outlook', 'Benign/Skype', 'Benign/WorldOfWarcraft']
        malicious = []

        return path, benign, malicious


class Flag(str, Enum):
    TH_FIN = 0x01  # end of data
    TH_SYN = 0x02  # synchronize sequence numbers
    TH_RST = 0x04  # reset connection
    TH_PUSH = 0x08  # push
    TH_ACK = 0x10  # acknowledgment number set
    TH_URG = 0x20  # urgent pointer set
    TH_ECE = 0x40  # ECN echo, RFC 3168
    TH_CWR = 0x80  # congestion window reduced
    TH_NS = 0x100  # nonce sum, RFC 3540


class Color(str, Enum):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_flow_key(src_ip: bytes, src_port: int, dst_ip: bytes, dst_port: int, protocol: int, session: bool):
    # If session is True, consider it as bidirectional
    if session:
        # Define a list with the source and destination info
        endpoints = [(src_ip, src_port), (dst_ip, dst_port)]

        # Sort this list - this will sort by IP first, then by port
        sorted_endpoints = sorted(endpoints)

        # Unpack the sorted list to get the ordered IPs and ports
        ip1, port1 = sorted_endpoints[0]
        ip2, port2 = sorted_endpoints[1]

        # Define the flow key as a tuple of these ordered elements
        flow_key = (ip1, port1, ip2, port2, protocol)
    else:
        # If session is False, consider it as unidirectional
        flow_key = (src_ip, src_port, dst_ip, dst_port, protocol)

    return flow_key


def inet_to_str(inet):
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def mac_addr(address):
    return ':'.join('%02x' % compat_ord(b) for b in address)


def flow_key_to_str(flow_key):
    value = (inet_to_str(flow_key[0]), flow_key[1], inet_to_str(flow_key[2]), flow_key[3], flow_key[4])
    return value


def print_packet(timestamp, buf):
    # Print out the timestamp in UTC
    print('Timestamp: ', str(datetime.datetime.utcfromtimestamp(timestamp)))

    # Unpack the Ethernet frame (mac src/dst, ethertype)
    eth = dpkt.ethernet.Ethernet(buf)
    print('Ethernet Frame: ', mac_addr(eth.src), mac_addr(eth.dst), eth.type)

    # Make sure the Ethernet data contains an IP packet
    if not isinstance(eth.data, dpkt.ip.IP):
        print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
        return

    # Now unpack the data within the Ethernet frame (the IP packet)
    # Pulling out src, dst, length, fragment info, TTL, and Protocol
    ip = eth.data

    # Pull out fragment information (flags and offset all packed into off field, so use bitmasks)
    do_not_fragment = bool(ip.off & dpkt.ip.IP_DF)
    more_fragments = bool(ip.off & dpkt.ip.IP_MF)
    fragment_offset = ip.off & dpkt.ip.IP_OFFMASK

    # Print out the info
    print('IP: %s -> %s   (len=%d ttl=%d DF=%d MF=%d offset=%d)\n' % \
          (inet_to_str(ip.src), inet_to_str(ip.dst), ip.len, ip.ttl, do_not_fragment, more_fragments,
           fragment_offset))


def plot_one_image(data1):
    plt.imshow(data1, interpolation='nearest')
    plt.show()


def plot_np_image(data1, data2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(data1, interpolation='nearest', cmap='gray')
    ax2.imshow(data2, interpolation='nearest', cmap='gray')
    plt.show()


def sec_to_ns(seconds):
    return int(seconds * 1e+9)


def ns_to_sec(nanoseconds):
    return nanoseconds / 1e+9


def get_dataset_local():
    dataset_name = '../../data/TII-SSRC-23'

    # Path for benign pcap files, recursively search through subdirectories
    benign_path = os.path.join(dataset_name, 'benign', '**', '*.pcap')
    benign_files = glob.glob(benign_path, recursive=True)

    # Path for malware pcap files, recursively search through subdirectories
    malicious_path = os.path.join(dataset_name, 'malicious', '**', '*.pcap')
    malicious_files = glob.glob(malicious_path, recursive=True)

    return 'TII-SSRC-23', benign_files, malicious_files


def get_dataset_debug():
    dataset_name = 'debug'
    benign, malware = [], []
    for path in os.listdir(f'../../data/{dataset_name}/benign/'):
        cr_path = path.replace('.pcap', '')
        benign.append(f'benign/{cr_path}')

    for path in os.listdir(f'../../data/{dataset_name}/malicious/'):
        cr_path = path.replace('.pcap', '')
        malware.append(f'malicious/{cr_path}')
    return dataset_name, benign, malware


def get_dataset_comms():
    dataset_name = 'CommsDataset_Adapted'
    benign, malware = [], []
    for path in os.listdir(f'../../Datasets/{dataset_name}/Benign/'):
        cr_path = path.replace('.pcap', '')
        benign.append(f'Benign/{cr_path}')

    for path in os.listdir(f'../../Datasets/{dataset_name}/Malware/'):
        cr_path = path.replace('.pcap', '')
        malware.append(f'Malware/{cr_path}')

    return dataset_name, benign, malware


def get_dataset_ustc():
    dataset_name = 'USTC-TFC2016'
    # benign = ['Benign/BitTorrent', 'Benign/Facetime', 'Benign/FTP', 'Benign/Gmail', 'Benign/MySQL',
    #          'Benign/Outlook', 'Benign/Skype', 'Benign/WorldOfWarcraft']
    malware = ['Malware/Cridex', 'Malware/Geodo', 'Malware/Htbot', 'Malware/Miuref', 'Malware/Neris',
               'Malware/Nsis-ay', 'Malware/Shifu', 'Malware/Tinba', 'Malware/Virut', 'Malware/Zeus']
    return dataset_name, malware


def numpy_folder_to_tensor(path):
    trainImages = []
    for i in os.listdir(path):
        data = np.load(path + i)
        trainImages.append(data)


def perf_measure(threshold, y_true, scores):
    y_pred = np.empty_like(y_true)
    for i in range(len(y_true)):
        if scores[i] < threshold:
            y_pred[i] = 0
        else:
            y_pred[i] = 1

    # from sklearn.metrics import confusion_matrix
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # print(tn, fp, fn, tp)
    # print('FAR', fp / (fp + tn) * 100)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f_score = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f_score
