import time

# from file_sniffer import SnifferFile
from file_sniffer import SnifferPcap
from options import Options as SnifferOptions
from preprocessing import LDPIPreProcessing


def main():
    # Preprocessing dataset
    sniffer_args = SnifferOptions()
    fsnf = SnifferPcap(sniffer_args)
    fsnf.subscribers.append(LDPIPreProcessing())
    fsnf.start()

    while True:
        if fsnf.stopped():
            fsnf.join()
            fsnf.terminate()
        time.sleep(0.5)


if __name__ == '__main__':
    main()
