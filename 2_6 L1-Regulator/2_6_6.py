import numpy as np


def func(x):
    return -0.5 * x ** 2 + 0.1 * x ** 3 + np.cos(3 * x) + 7


def loss(w, x, y):
    return (x @ w - y) ** 2


data_x = np.arange(-4.0, 6.0, 0.1)
data_y = func(data_x)

N = 5
lm_l1 = 2.0
sz = len(data_x)
w = np.zeros(N)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002])
n_iter = 500
lm = 0.02
batch_size = 20

np.random.seed(0)

x_train = np.array([[a ** n for n in range(N)] for a in data_x])
y_train = np.array(data_y)

Qe = np.mean(loss(w, x_train, y_train))
for i in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)
    kb = k + batch_size
    lossesK = np.mean(loss(w, x_train[k:kb], y_train[k:kb]))
    Qe = lm * lossesK + (1 - lm) * Qe
    wt = np.hstack([[0], w[1:]])
    w -= eta * (((2 / batch_size) * np.sum((x_train[k:kb] @ w - y_train[k:kb]) * x_train[k:kb].T, axis=1)) + lm_l1 * np.sign(wt))
Q = np.mean(loss(w, x_train, y_train))


print(w)
print(Q)
print(Qe)