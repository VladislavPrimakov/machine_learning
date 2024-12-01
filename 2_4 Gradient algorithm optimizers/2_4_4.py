import numpy as np


def f(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)


def df(x):
    return 0.4 + 0.2 * np.cos(2*x) - 0.6 * np.sin(3*x)


n = 1.0
x = 4.0
N = 500
g = 0.7
v = 0

np.random.seed(0)

for i in range(N):
    v = g * v + (1 - g) * n * df(x-g*v)
    x -= v

print(x)