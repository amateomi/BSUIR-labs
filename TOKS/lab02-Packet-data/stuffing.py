#!/usr/bin/env python3

FLAG_BYTE = "|"
ESC_CHAR = ">"


def encode_message(message: str):
    parts = message.split(FLAG_BYTE)
    parts[0] = FLAG_BYTE + parts[0]
    return f"{FLAG_BYTE}{ESC_CHAR}".join(parts)


def decode_message(message: str):
    message = message[1:]
    parts = message.split(FLAG_BYTE)
    return "".join([parts[0]] + [FLAG_BYTE + x[1:] for x in parts[1:] if len(x) > 0 and x[0] == ESC_CHAR])


def main():
    while True:
        message = input("Enter message: ")
        encoded = encode_message(message)
        print(f"Encoded message: {encoded}")
        decoded = decode_message(encoded)
        print(f"Decoded message: {decoded}")


if __name__ == "__main__":
    main()
