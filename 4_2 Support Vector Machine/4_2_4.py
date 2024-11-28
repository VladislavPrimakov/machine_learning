import numpy as np
from sklearn import svm

def func(x):
    return np.sin(0.5*x) + 0.2 * np.cos(2*x) - 0.1 * np.sin(4 * x) - 2.5


def a(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3 + w[4] * np.cos(x) + w[5] * np.sin(x)


# обучающая выборка
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

x_train = np.array([[1, x, x**2, x**3, np.cos(x), np.sin(x)] for x in coord_x])
y_train = coord_y

# здесь продолжайте программу
model = svm.SVR(kernel='linear')
model.fit(x_train, y_train)
w = np.array(model.coef_[0])
w[0] = model.intercept_[0]
Q = np.mean([(a(w, _x) - _y)**2 for _x, _y in zip(coord_x, y_train)] )

print(w)
print(Q)