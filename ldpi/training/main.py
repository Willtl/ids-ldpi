import logging

from file_sniffer import SnifferPcap
from options import Options as SnifferOptions
from preprocessing import LDPIPreProcessing


def main():
    # Suppose this is your debug flag
    debug = True

    # Configure logging based on the debug flag
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Process PCAP into network flow samples
    sniffer_args = SnifferOptions()
    fsnf = SnifferPcap(sniffer_args)
    fsnf.subscribers.append(LDPIPreProcessing())
    fsnf.run()

    # Train it


if __name__ == '__main__':
    main()
