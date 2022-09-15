import argparse
import os

import readchar
import serial


def read(port: str):
    with serial.Serial(port) as port:
        while True:
            if port.inWaiting():
                print(port.read().decode(), end="", flush=True)


def write(port1: str, port2: str):
    port_numbers = [int(port1[-1]), int(port2[-1])]
    with serial.Serial(port1) as port1, serial.Serial(port2) as port2:
        while True:
            try:
                command_parts = input("""
<port number> - Enter message
<port number> rate <baud rate> - Change baud rate 
>""").strip().split(" ")

                port_num = int(command_parts[0])

                match len(command_parts):
                    case 1:
                        if port_num in port_numbers:
                            char = ""
                            while char != "\n":
                                char = readchar.readchar()

                                if port_num == port_numbers[0]:
                                    port1.write(char.encode())

                                elif port_num == port_numbers[1]:
                                    port2.write(char.encode())
                        else:
                            raise ValueError
                    case 3:
                        rate = int(command_parts[2])
                        if command_parts[1] == "rate" and rate in serial.Serial.BAUDRATES:
                            if port_num == port_numbers[0]:
                                port1.baudrate = rate

                            elif port_num == port_numbers[1]:
                                port2.baudrate = rate
                        else:
                            raise IndexError
                    case _:
                        raise ValueError

            except IndexError:
                print(f"Invalid baudrate! Possible values: {serial.Serial.BAUDRATES}")
            except ValueError:
                print("Invalid port number or baud rate value!")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("port1", metavar="port1", help="First serial port")
    parser.add_argument("port2", nargs="?", default=None, metavar="port2", help="Second serial port (only writer mode)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--reader", action="store_true")
    group.add_argument("-w", "--writer", action="store_true")

    args = parser.parse_args()

    if not args.reader and not args.writer or args.reader and args.port2 or args.writer and not args.port2:
        parser.print_help()
        exit(1)

    if args.reader:
        print(f"Port: {args.port1}")
        read(args.port1)
    elif args.writer:
        os.system(f"/usr/bin/dbus-launch /usr/bin/gnome-terminal -- python serial_port.py -r {args.port1} ")
        os.system(f"/usr/bin/dbus-launch /usr/bin/gnome-terminal -- python serial_port.py -r {args.port2} ")
        write(args.port1, args.port2)


if __name__ == "__main__":
    main()
