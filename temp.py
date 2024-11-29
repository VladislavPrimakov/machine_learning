import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

def loss(w, x, y):
     return (w.T @ x - y) ** 2

def df(w, x, y):
     return 2 * (w.T @ x - y) * x.T

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат
x_train = np.array([[1, i, i ** 2, np.cos(2 * i), np.sin(2 * i)] for i in coord_x])

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего

Qe = np.mean(loss(w, x_train.T, coord_y)) # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for _ in range(N):
     k = np.random.randint(0, sz - 1)
     w -= eta * df(w, x_train[k], coord_y[k])
     q = np.mean(loss(w, x_train[k], coord_y[k]))
     Qe = lm * q + (1 - lm) * Qe

Q = np.mean(loss(w, x_train.T, coord_y))


print(w)
print(Qe)
print(Q)
import matplotlib.pyplot as plt
plt.plot(coord_x, coord_y, c='g', label="origin")
plt.plot(coord_x, x_train @ w, c='r', label="sgd")
plt.legend()
plt.show()