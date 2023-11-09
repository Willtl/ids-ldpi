from ldpi.inference import LightDeepPacketInspection
from options import Options
from sniffer.sniffer import Sniffer


def main():
    args: Options = Options()
    snf = Sniffer(args)
    snf.add_subscriber(LightDeepPacketInspection())

    snf.start()
    snf.start_subscribers()


if __name__ == '__main__':
    main()
