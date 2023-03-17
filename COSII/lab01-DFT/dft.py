from cmath import *

import matplotlib.pyplot as plt
import numpy as np


def inner_function(x):
    return np.sin(3 * x) + np.cos(x)


def discretize(n):
    return [inner_function(i * 2 * pi / n) for i in range(n)]


def draw_inner_function():
    x = np.arange(0, 2 * pi, 0.01)
    y = inner_function(x)
    plt.plot(x, y)
    plt.grid()
    plt.show()


def draw_samples(samples):
    plt.plot(samples)
    plt.grid()
    plt.show()


def draw_peaks(data):
    plt.stem(data)
    plt.grid()
    plt.show()


def dft(x):
    n = len(x)
    c = [0.0 + 0.0j] * n
    for k in range(n):
        for m in range(n):
            c[k] += x[m] * exp(-2j * pi * k * m / n)
    return c


def idft(x):
    n = len(x)
    c = [0.0 + 0.0j] * n
    for k in range(n):
        for m in range(n):
            c[k] += x[m] * exp(2j * pi * k * m / n)
        c[k] *= 1 / n
    return c


def fft(x):
    n = len(x)
    r = n // 2
    out = [0. + 0.0j] * n

    for i in range(r):
        out[i] = x[i] + x[i + r]
        out[i + r] = (x[i] - x[i + r]) * exp(-2j * pi * i / n)

    if n > 2:
        top_list = fft(out[:r])
        bot_list = fft(out[r:])

        for i in range(r):
            out[i * 2] = top_list[i]
            out[(i + 1) * 2 - 1] = bot_list[i]

    return out


def main():
    draw_inner_function()

    n = 16
    samples = discretize(n)
    draw_samples(samples)

    spectrum = dft(samples)
    magnitudes = np.abs(spectrum)
    draw_peaks(magnitudes)
    phases = np.angle(spectrum)
    draw_peaks(phases)

    another_samples = [x.real for x in idft(spectrum)]
    draw_samples(another_samples)

    spectrum = fft(samples)
    magnitudes = np.abs(spectrum)
    draw_peaks(magnitudes)
    phases = np.angle(spectrum)
    draw_peaks(phases)


if __name__ == "__main__":
    main()
