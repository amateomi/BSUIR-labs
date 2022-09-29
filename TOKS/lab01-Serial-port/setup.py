#!/usr/bin/env python3

import argparse

import portio


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("port", metavar="port", help="Serial port")
    parser.add_argument("mode", metavar="mode", choices=["read", "write"], help="I/O option")

    args = parser.parse_args()

    port = args.port
    mode = args.mode

    match mode:
        case "read":
            portio.read(port)
        case "write":
            portio.write(port)


if __name__ == "__main__":
    main()
