#!/bin/env python

from fxpmath import Fxp
import numpy as np

N = 17
FRAC = N-2
K = 8

ADDRtype = Fxp(val=None, signed=False, n_word= int(np.log2(2 ** (K-1))), n_frac=0)
DATA = Fxp(val=None, signed=True, n_word=N, n_frac=FRAC)
DATA_X = Fxp(val=None, signed=True, n_word=N - 1, n_frac=FRAC)

x = np.array([1, 0, 0, 0, 0, 0, 0, 0])
a = np.array([ 0.25, 0.25, 0.125, -0.25, 0.625, -0.5, 0.125, -0.25])
assert(x.size == a.size)

print(f"result={Fxp(np.array([sum(x * a)])).like(DATA).bin()}\n")

for i, value in enumerate(Fxp(a).like(DATA_X)):
    print(f"a[{i}] = 16'b{value.bin()};")
print("")
for i, value in enumerate(Fxp(x).like(DATA_X)):
    print(f"x[{i}] = 16'b{value.bin()};")
print("")

ROM_shift = Fxp(np.array([-(1 / 2) * (a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] + a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] + a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] + a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] + a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] + a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] + a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] - a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] - a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] - a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] - a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] - a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] - a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] - a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] + a[3] - a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] + a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] + a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] + a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] + a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] + a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] + a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] + a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] + a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] - a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] - a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] - a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] - a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] - a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] - a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] - a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] + a[2] - a[3] - a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] + a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] + a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] + a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] + a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] + a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] + a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] + a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] + a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] - a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] - a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] - a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] - a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] - a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] - a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] - a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] + a[3] - a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] + a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] + a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] + a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] + a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] + a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] + a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] + a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] + a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] - a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] - a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] - a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] - a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] - a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] - a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] - a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] + a[1] - a[2] - a[3] - a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] + a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] + a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] + a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] + a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] + a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] + a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] + a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] - a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] - a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] - a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] - a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] - a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] - a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] - a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] + a[3] - a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] + a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] + a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] + a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] + a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] + a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] + a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] + a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] + a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] - a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] - a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] - a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] - a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] - a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] - a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] - a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] + a[2] - a[3] - a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] + a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] + a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] + a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] + a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] + a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] + a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] + a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] + a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] - a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] - a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] - a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] - a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] - a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] - a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] - a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] + a[3] - a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] + a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] + a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] + a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] + a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] + a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] + a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] + a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] + a[4] - a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] - a[4] + a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] - a[4] + a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] - a[4] + a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] - a[4] + a[5] - a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] - a[4] - a[5] + a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] - a[4] - a[5] + a[6] - a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] - a[4] - a[5] - a[6] + a[7]),
                          -(1 / 2) * (a[0] - a[1] - a[2] - a[3] - a[4] - a[5] - a[6] - a[7]),
])).like(DATA)

for v, z in enumerate(ROM_shift):
    print(f"ROM[{v}] = 17'b{z.bin()};")
