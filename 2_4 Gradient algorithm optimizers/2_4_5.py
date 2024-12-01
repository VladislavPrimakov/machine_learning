import numpy as np


def f(x):
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3*x)


def df(x):
    return 2 + 0.3 * x**2 - 6 * np.sin(3*x)


n = 0.5
x = 4.0
N = 200
a = 0.8
G = 0
e = 0.01

np.random.seed(0)

for i in range(N):
    _df = df(x)
    G = a * G + (1 - a) * _df * _df
    x = x - n * (_df / (np.sqrt(G) + e))

print(x)