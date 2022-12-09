#!/usr/bin/env python3

import textwrap
import time
from dataclasses import dataclass

SEGMENT_SIZE = 3


@dataclass
class Packet:
    sequence_number: int = 0
    acknowledgment_number: int = 1

    is_acknowledgment: bool = False
    is_synchronize: bool = True
    is_finish: bool = False

    window_size: int = SEGMENT_SIZE

    message: str = "None"

    def __repr__(self):
        return f"Packet(" \
               f"SN={self.sequence_number} " \
               f"AN={self.acknowledgment_number} " \
               f"ACK={self.is_acknowledgment} " \
               f"SYN={self.is_synchronize} " \
               f"FIN={self.is_finish} " \
               f"W={self.window_size} " \
               f"Message={self.message})"


class Client:
    @classmethod
    def send_packet(cls, packet: Packet):
        print(f"Client send {packet}")
        Server.packet = packet

    @classmethod
    def send_acknowledgement(cls, packet: Packet) -> Packet:
        ack_packet = Packet(packet.acknowledgment_number,
                            packet.acknowledgment_number + SEGMENT_SIZE,
                            True,
                            False,
                            packet.is_finish,
                            packet.window_size,
                            packet.message)

        print(f"Client send {ack_packet}")
        return ack_packet


class Server:
    packet: Packet

    @classmethod
    def send_acknowledgement(cls) -> Packet:
        ack_packet = Packet(Server.packet.acknowledgment_number,
                            Server.packet.acknowledgment_number + SEGMENT_SIZE,
                            True,
                            Server.packet.is_synchronize,
                            Server.packet.is_finish,
                            Server.packet.window_size,
                            Server.packet.message)

        print(f"Server send {ack_packet}")
        return ack_packet


def main():
    first_packet = Packet()
    Client.send_packet(first_packet)
    ack_packet = Server.send_acknowledgement()
    last_packet = Client.send_acknowledgement(ack_packet)

    while True:
        option = input("Change window size (y/n or q to quit):")
        if option == "q":
            break
        elif option == "y":
            window_size = int(input("Enter new window size:"))
            packet = Packet(last_packet.acknowledgment_number,
                            last_packet.acknowledgment_number + SEGMENT_SIZE,
                            False,
                            False,
                            False,
                            window_size)
            Client.send_packet(packet)
            last_packet = Server.send_acknowledgement()

        message = input("Enter message:")
        segments = textwrap.wrap(message, 3)

        i = 0
        for seg in segments:
            packet = Packet(last_packet.acknowledgment_number,
                            last_packet.acknowledgment_number + SEGMENT_SIZE,
                            False,
                            False,
                            seg == segments[-1],
                            last_packet.window_size,
                            seg)
            Client.send_packet(packet)
            last_packet = Server.send_acknowledgement()

            i += 1
            if i == last_packet.window_size or seg == segments[-1]:
                print(f"Send {last_packet.window_size} packets in window")
                time.sleep(1)
                i = 0


if __name__ == '__main__':
    main()
