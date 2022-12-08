#!/usr/bin/env python3

from dataclasses import dataclass

TOTAL_STATIONS = 4


@dataclass
class Sequence:
    priority: int = 0
    is_frame: bool = False

    destination_address: int = None
    source_address: int = None
    message: str = None

    is_address_recognized: bool = False
    is_frame_copied: bool = False

    def __repr__(self):
        if self.is_frame:
            return f"Frame(" \
                   f"P={self.priority} " \
                   f"T={int(self.is_frame)} " \
                   f"DA={self.destination_address} " \
                   f"SA={self.source_address} " \
                   f"INFO='{self.message}' " \
                   f"A={int(self.is_address_recognized)} " \
                   f"C={int(self.is_frame_copied)})"

        return f"Token(" \
               f"P={self.priority} T={int(self.is_frame)})"


def main():
    current_sequence = Sequence()
    stations_queue: dict[int, Sequence] = {i: None for i in range(TOTAL_STATIONS)}
    while True:
        for station_id in range(TOTAL_STATIONS):
            print(f"\nStation №{station_id}")
            print(f"Stations queue={stations_queue}")
            print(f"Get sequence={current_sequence}")

            if current_sequence.destination_address == station_id:
                print(f"Message has arrived: {current_sequence.message}")
                current_sequence.is_address_recognized = True
                current_sequence.is_frame_copied = True

            if current_sequence.source_address == station_id:
                print(f"Remove consumed message: {current_sequence.message}")
                if stations_queue[station_id] is not None:
                    current_sequence = stations_queue[station_id]
                    stations_queue[station_id] = None
                else:
                    current_sequence = Sequence()

            if stations_queue[station_id] is None:
                message = input("Enter new station message (`n` for skip):").strip()
                if message != "n":
                    destination_address = int(input("Enter destination station address:"))

                    new_sequence = Sequence(station_id, True,
                                            destination_address, station_id, message,
                                            False, False)

                    if current_sequence.is_frame and current_sequence.priority < new_sequence.priority:
                        print(f"Store current frame={current_sequence} in queue")
                        stations_queue[station_id] = current_sequence
                    else:
                        print("Capture token")

                    current_sequence = new_sequence


if __name__ == '__main__':
    main()
