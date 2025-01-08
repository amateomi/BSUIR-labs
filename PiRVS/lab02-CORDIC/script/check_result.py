#!/bin/env python

import sys
from fxpmath import Fxp

N = 16
FRAC = N-3
DATA = Fxp(val=None, signed=True, n_word=N, n_frac=FRAC)

print(f"x={Fxp(sys.argv[1]).like(DATA)}")
print(f"y={Fxp(sys.argv[2]).like(DATA)}")
print(f"z={Fxp(sys.argv[3]).like(DATA)}")
