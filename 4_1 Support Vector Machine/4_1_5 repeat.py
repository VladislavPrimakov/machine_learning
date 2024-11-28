import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.02 * np.exp(-x) - 0.2 * np.sin(3 * x) + 0.5 * np.cos(2 * x) - 7

def df(w, x, y):
    return 2 * (x @ w - y) * x.T

def loss(w, x, y):
    return (x @ w - y)**2

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 1e-3, 1e-4, 1e-5, 1e-6]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего

x = np.column_stack([np.ones(sz), coord_x, coord_x**2, coord_x**3, coord_x**4])
y = np.array(coord_y)
Qe = np.mean(loss(w, x, y))# начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

# здесь продолжайте программу
for i in range(N):
    k = np.random.randint(0, sz-1)
    lossK = loss(w, x[k],y[k])
    w -= eta * df(w, x[k], y[k])
    Qe = lm * lossK + (1-lm) * Qe
Q = np.mean(loss(w, x, y))
print(w)
print(Q)
print(Qe)