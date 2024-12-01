import numpy as np

def func(x):
    return 0.1 * x**2 - np.sin(x) + 0.1 * np.cos(x * 5) + 1.


def loss(y, y_real):
    return (y-y_real)**2

def model(w, x):
    return sum([w[0], *[w[i] * x**i for i in range(1, 5)]])
# здесь объявляйте дополнительные функции (если необходимо)


coord_x = np.arange(-5.0, 5.0, 0.1) # значения отсчетов по оси абсцисс
coord_y = func(coord_x) # значения функции по оси ординат
w = [1.11,-0.26, 0.061, 0.0226, 0.00178]
sz = len(coord_x) # общее число отсчетов

# здесь продолжайте программу
a = [model(w, x) for x in coord_x]
Q = sum([loss(a[i], coord_y[i]) for i in range(sz)]) / sz

print(Q)