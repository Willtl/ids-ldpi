import time

from ldpi import LightDeepPacketInspection
from options import SnifferOptions
from sniffer import SnifferPcap


def main():
    args = SnifferOptions()
    args.parse_options()

    snf = SnifferPcap(args)
    # snf = Sniffer(args)
    snf.set_pcap_path('datasets/TII-SSRC-23/pcap/benign/video/rtp.pcap')
    # snf.set_pcap_path('datasets/TII-SSRC-23/pcap/malicious/dos/fin_tcp_dos.pcap')
    ldpi = LightDeepPacketInspection()
    snf.add_subscriber(ldpi)
    snf.run()

    # Decision engine loop
    try:
        while True:
            time.sleep(1.0)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down...")
        snf.stop()
        ldpi.stop()


# test

if __name__ == '__main__':
    main()
