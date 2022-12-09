#!/usr/bin/env python3

polynomial = [int(bit) for bit in "10011"]


def calculate_hash(message: list[int]) -> list[int]:
    hash_len = len(polynomial) - 1
    align_len = len(message) - 1

    reminder = message.copy() + [0] * hash_len
    divisor = polynomial.copy() + [0] * align_len

    for i, _ in enumerate(message):
        if reminder[i] == divisor[i] == 1:
            reminder = [x ^ y for x, y in zip(reminder, divisor)]
        divisor.pop()
        divisor.insert(0, 0)

    return reminder[-hash_len:]


def to_string(bits: list[int]) -> str:
    return "".join(map(str, bits))


def main():
    while True:
        message = [int(bit) for bit in "1101011111"]

        crc_hash = calculate_hash(message)
        print(f"\nEncoded message: {to_string(message)} {to_string(crc_hash)}")

        option = int(input("0) - No errors, 1) - Add errors\n>"))
        if option == 1:
            i1, i2 = input("Enter two error indices\n>").strip().split(" ")
            i1 = int(i1)
            i2 = int(i2)

            message[i1] = 1 if message[i1] == 0 else 0
            message[i2] = 1 if message[i2] == 0 else 0

        print(f"Message to send: {to_string(message)} {to_string(crc_hash)}")
        crc_hash = calculate_hash(message + crc_hash)
        print(f"Sent message hash: {to_string(crc_hash)}")
        print(f"{'None' if crc_hash.count(1) == 0 else 'Some'} error detected")


if __name__ == '__main__':
    main()
