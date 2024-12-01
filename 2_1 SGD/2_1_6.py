import numpy as np


def func(x):
    return 0.1 * x ** 2 - np.sin(x) + 5.


def dfQ(W, X, Y):
    s = 0
    for x, y in zip(X, Y):
        s += (x @ W - y) * x.T
    return 2 * s / sz


def QW(W, X, Y):
    return np.array((X @ W - Y) ** 2).mean()


coord_x = np.arange(-5.0, 5.0, 0.1)  # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x)  # значения функции по оси ординат

sz = len(coord_x)  # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001])  # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.])  # начальные значения параметров модели
N = 200  # число итераций градиентного алгоритма

X = np.column_stack([np.ones(sz), coord_x, coord_x ** 2, coord_x ** 3])
Y = np.array(coord_y)
for i in range(N):
    w -= eta * dfQ(w, X, Y)
Q = QW(w, X, Y)

print(w)
print(Q)