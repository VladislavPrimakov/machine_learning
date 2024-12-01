import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.1 * x ** 3


def df(x):
    return 0.5 + 0.4 * x - 0.3 * x**2


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс
coord_y = func(coord_x) # значения по оси ординат (значения функции)
x = -4
n = 0.01
N = 200
for i in range(N):
    x -= n * df(x)

print(x)