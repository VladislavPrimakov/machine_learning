import numpy as np

def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

def loss(y, y_real):
    return np.abs(y-y_real)
def model(w, x):
    return w[0] + w[1] * x + w[2] * x**2 + w[3] * np.cos(2*x) + w[4] * np.sin(2*x)

coord_x = np.arange(-5.0, 5.0, 0.1) # значения отсчетов по оси абсцисс
coord_y = func(coord_x) # значения функции по оси ординат
w=[-1.59,-0.69,0.278,0.497,-0.106]
sz = len(coord_x) # общее число отсчетов

# здесь продолжайте программу
predict = [model(w, x) for x in coord_x]
Q = sum([loss(predict[i], coord_y[i]) for i in range(sz)]) / sz


print(Q)