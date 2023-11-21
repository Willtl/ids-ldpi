import time

from ldpi.inference import LightDeepPacketInspection
from options import SnifferOptions
from sniffer.sniffer import SnifferPcap


def main():
    args = SnifferOptions()
    snf = SnifferPcap(args)
    # snf = Sniffer(args)
    snf.set_pcap_path('datasets/TII-SSRC-23/pcap/malicious/bruteforce/bruteforce_ftp.pcap')
    ldpi = LightDeepPacketInspection()
    snf.add_subscriber(ldpi)
    snf.run()

    # Decision engine loop
    try:
        while True:
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down...")
        snf.stop()
        ldpi.stop()


if __name__ == '__main__':
    main()
