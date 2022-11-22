#!/usr/bin/env python3

import random
import time

propagation_time = 1
transmission_time = propagation_time * 2


class Receiver:
    __transmission_message = ""

    @classmethod
    def set_transmission_message(cls, message: str):
        cls.__transmission_message = message

    @staticmethod
    def is_collision() -> bool:
        return random.choice([True, False])

    @classmethod
    def print_transmitted_message(cls):
        print(f'Collision not detected. Transmitted message is "{cls.__transmission_message}"')


class Transmitter:
    __try_counter = 0

    @staticmethod
    def send_message(message: str):
        print(f'Sending message "{message}"')
        Receiver.set_transmission_message(message)

    @staticmethod
    def send_jam():
        print("Jam signal")

    @staticmethod
    def wait_transmission_time():
        print(f"Waiting for {transmission_time}s...")
        time.sleep(transmission_time)

    @classmethod
    def increment_try_counter(cls):
        cls.__try_counter += 1
        print(f"Try counter is equal {cls.__try_counter}")

    @classmethod
    def is_try_counter_max(cls) -> bool:
        if cls.__try_counter > 10:
            print("Error: Too many tries")
            return True
        return False

    @classmethod
    def calculate_and_wait_random_delay(cls):
        time_to_wait = random.randint(0, 2 ** cls.__try_counter)
        print(f"Generated time is {time_to_wait} (random from 0 to {2 ** cls.__try_counter})")
        print(f"Waiting for {time_to_wait}s...")
        time.sleep(time_to_wait)

    @classmethod
    def clear_try_counter(cls):
        cls.__try_counter = 0


def main():
    while True:
        message = input("\nEnter message (`exit` to quit program): ")

        if message == "exit":
            break

        Transmitter.clear_try_counter()
        Transmitter.send_message(message)

        while Receiver.is_collision():
            Transmitter.send_jam()
            Transmitter.wait_transmission_time()

            Transmitter.increment_try_counter()
            if Transmitter.is_try_counter_max():
                exit(1)

            Transmitter.calculate_and_wait_random_delay()

        Receiver.print_transmitted_message()


if __name__ == '__main__':
    main()
