import matplotlib.pyplot as plt
import numpy as np


def func_y(x):
    return np.sin(2 * x)


def func_z(x):
    return np.cos(7 * x)


def discretize(func, n):
    return [func(i * 2 * np.pi / n) for i in range(n)]


def draw_inner_function(func):
    x = np.arange(0, 2 * np.pi, 0.01)
    y = func(x)
    plt.plot(x, y)
    plt.grid()
    plt.show()


def draw_samples(samples):
    plt.stem(samples)
    plt.grid()
    plt.show()


def convolve(x, y):
    n = len(x)
    z = [0.0] * n

    for i in range(n):
        for j in range(n):
            z[i] += x[j] * y[i - j]

    return z


def convolve_fft(x, y):
    x_spectrum = np.fft.fft(x)
    y_spectrum = np.fft.fft(y)
    z_spectrum = x_spectrum * y_spectrum
    z = np.fft.ifft(z_spectrum)
    return np.real(z)


def correlate(x, y):
    n = len(x)
    z = [0.0] * n

    for i in range(n):
        for j in range(n):
            z[i] += x[j] * y[(i + j) % n]

    return z


def correlate_fft(x, y):
    x_spectrum = np.fft.fft(x).conj()
    y_spectrum = np.fft.fft(y)
    z_spectrum = x_spectrum * y_spectrum
    z = np.fft.ifft(z_spectrum)
    return np.real(z)


def main():
    draw_inner_function(func_y)
    draw_inner_function(func_z)

    n = 16
    y_samples = discretize(func_y, n)
    z_samples = discretize(func_z, n)

    x_samples = convolve(z_samples, y_samples)
    draw_samples(x_samples)

    x_samples = convolve_fft(z_samples, y_samples)
    draw_samples(x_samples)

    x_samples = correlate(y_samples, z_samples)
    draw_samples(x_samples)

    x_samples = correlate_fft(y_samples, z_samples)
    draw_samples(x_samples)


if __name__ == '__main__':
    main()
