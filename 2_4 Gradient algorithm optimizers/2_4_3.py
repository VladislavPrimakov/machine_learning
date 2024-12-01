import numpy as np


def f(x):
    return -0.5 * x + 0.2 * x**2 - 0.01 * x**3 - 0.3 * np.sin(4*x)


def df(x):
    return -0.5 + 0.4 * x - 0.03 * x**2 - 1.2 * np.cos(4*x)


n = 0.1
x = -3.5
N = 200
g = 0.8
v = 0

np.random.seed(0)

for i in range(N):
    v = g * v + (1 - g) * n * df(x)
    x -= v

print(x)