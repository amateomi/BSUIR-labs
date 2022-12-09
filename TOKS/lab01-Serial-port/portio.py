#!/usr/bin/env python3

import argparse

import serial

WRITE_MESSAGE = """Enter option:
w) Write message
c) Change baudrate
>"""

BAUDRATES_TO_PRINT = []
for i, x in enumerate(serial.Serial.BAUDRATES):
    BAUDRATES_TO_PRINT.append(f"{i}) - {x}")
    if i % 2 == 0:
        BAUDRATES_TO_PRINT[i] += "\t"
        if i == 0:
            BAUDRATES_TO_PRINT[i] += "\t"
    else:
        BAUDRATES_TO_PRINT[i] += "\n"
BAUDRATES_TO_PRINT = "".join(BAUDRATES_TO_PRINT)


def read(port_name: str):
    with serial.Serial(port_name, timeout=None) as port:
        print(f"Reading from port {port_name}")

        while True:
            message = port.readline().decode()
            print(f"Decoded message: {message}", flush=True)


def write(port_name: str):
    with serial.Serial(port_name, timeout=None) as port:
        print(f"Ready to write to the {port_name}")

        while True:
            option = input(WRITE_MESSAGE)

            match option:
                case "w":
                    message = input("Enter message: ") + "\n"
                    print(f"Encoded message: {message}")
                    port.write(message.encode())

                case "c":
                    option = input(BAUDRATES_TO_PRINT + "Enter option: ")
                    try:
                        option = int(option)
                        if option < 0:
                            raise IndexError
                        rate = serial.Serial.BAUDRATES[option]
                        port.baudrate = rate
                        print(f"New port baud rate is {port.baudrate}")
                    except (ValueError, IndexError):
                        print("Invalid index number")

                case _:
                    print("Invalid option")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("port", metavar="port", help="Serial port")
    parser.add_argument("mode", metavar="mode", choices=["read", "write"], help="I/O option")

    args = parser.parse_args()

    port = args.port
    mode = args.mode

    match mode:
        case "read":
            read(port)
        case "write":
            write(port)


if __name__ == "__main__":
    main()
