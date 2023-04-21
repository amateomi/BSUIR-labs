import numpy as np


def signal(x):
    return np.sin(x) + np.cos(4 * x)


def discretize(func, n):
    return [func(i * 2 * np.pi / n) for i in range(n)]


def fwht(samples):
    n = len(samples)
    r = n // 2

    result = [0.0] * n

    for i in range(r):
        result[i] = samples[i] + samples[i + r]
        result[i + r] = -samples[i + r] + samples[i]

    samples = result
    if n > 2:
        samples[:r] = fwht(result[:r])
        samples[r:] = fwht(result[r:])

    return result


def main():
    n = 16
    samples = discretize(signal, n)
    transformation_result = np.array(fwht(samples))
    print(transformation_result)


if __name__ == "__main__":
    main()
