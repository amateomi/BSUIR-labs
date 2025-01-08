#!/bin/env python

import math
import numpy as np
from fxpmath import Fxp

N = 16
FRAC = N-3
DATA = Fxp(val=None, signed=True, n_word=N, n_frac=FRAC)

ITERATIONS = 12

def compute_K(n):
    k = 1.0
    for i in range(n):
        k *= 1 / math.sqrt(1 + 2 ** (-2 * i))
    return k

K = compute_K(N)
print(f"K={Fxp(K).like(DATA).bin()}")

def print_stuff(stuff, str):
    print(f"{str}:\t{stuff}")
    print(f"{str} hex:\t{stuff.hex()}")
    print(f"{str} bin:\t{stuff.bin(frac_dot=True)}\n")

ROM = Fxp(np.array([
    math.atan(2.0 ** 0),
    math.atan(2.0 ** -1),
    math.atan(2.0 ** -2),
    math.atan(2.0 ** -3),
    math.atan(2.0 ** -4),
    math.atan(2.0 ** -5),
    math.atan(2.0 ** -6),
    math.atan(2.0 ** -7),
    math.atan(2.0 ** -8),
    math.atan(2.0 ** -9),
    math.atan(2.0 ** -10),
    math.atan(2.0 ** -11)
])).like(DATA)
print_stuff(ROM, "ROM")

for i, entry in enumerate(ROM):
    print(f"ROM[{i}] = 16'b{entry.bin()};")

# input
x_input = 0.99
y_input = 0.49
z_input = 0
is_direct = False

print(f"exact cos(z):\t{math.cos(z_input)}")
print(f"exact sin(z):\t{math.sin(z_input)}")

# preprocessing
if is_direct:
    if (-(math.pi / 2) <= z_input <= (math.pi / 2)):
        x = Fxp(x_input).like(DATA)
        y = Fxp(y_input).like(DATA)
        z = Fxp(z_input).like(DATA)
    elif ((math.pi / 2) < z_input < math.pi):
        x = Fxp(-y_input).like(DATA)
        y = Fxp(x_input).like(DATA)
        z = Fxp(z_input - math.pi / 2).like(DATA)
    elif (-math.pi < z_input < -(math.pi / 2)):
        x = Fxp(y_input).like(DATA)
        y = Fxp(-x_input).like(DATA)
        z = Fxp(z_input + math.pi / 2).like(DATA)
else:
    if (x_input > 0):
        x = Fxp(x_input).like(DATA)
        y = Fxp(y_input).like(DATA)
        z = Fxp(0).like(DATA)
    elif (x_input < 0 and y_input > 0):
        x = Fxp(y_input).like(DATA)
        y = Fxp(-x_input).like(DATA)
        z = Fxp(math.pi / 2).like(DATA)
    elif (x_input < 0 and y_input < 0):
        x = Fxp(-y_input).like(DATA)
        y = Fxp(x_input).like(DATA)
        z = Fxp(-math.pi / 2).like(DATA)
print_stuff(x, "x")
print_stuff(y, "y")
print_stuff(z, "z")

# cordic core
for i in range(ITERATIONS):
    if is_direct:
        sign = 1 if z >= 0 else -1
        x_next = x - sign * (y >> i)
        y_next = y + sign * (x >> i)
        z = z - sign * ROM[i]
    else:
        sign = 1 if y >= 0 else -1
        x_next = x + sign * (y >> i)
        y_next = y - sign * (x >> i)
        z = z + sign * ROM[i]
    x = x_next
    y = y_next
    print_stuff(x, f"x{i}")
    print_stuff(y, f"y{i}")
    print_stuff(z, f"z{i}")

# postprocessing
x *= K
print_stuff(x, "x result")

y *= K
print_stuff(y, "y result")

print_stuff(z, "z result")
