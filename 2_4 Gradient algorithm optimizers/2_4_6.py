import numpy as np


def loss(W, X, Y):
    return (X @ W - Y) ** 2


def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2


data_x = np.arange(-4.0, 6.0, 0.1)
data_y = func(data_x)

sz = len(data_x)
W = np.array([0.0, 0.0, 0.0, 0.0])
eta = np.array([0.1, 0.01, 0.001, 0.0001])
lm = 0.02
N = 500
batch_size = 20
gamma = 0.8
v = np.zeros(len(W))

np.random.seed(0)

X = np.column_stack([np.ones(sz), data_x, data_x ** 2, data_x ** 3])
Y = np.array(data_y)
Qe = np.mean([loss(W, x, y) for x, y in zip(X, Y)])
for i in range(N):
    k = np.random.randint(0, sz - batch_size - 1)
    QK = np.mean([loss(W, X[i], Y[i]) for i in range(k, k + batch_size)])
    Qe = lm * QK + (1 - lm) * Qe
    t = (W - gamma * v)
    v = gamma * v + eta * (1 - gamma)  * (2 / batch_size) * np.sum([(t @ X[i]  - Y[i]) * X[i].T for i in range(k, k + batch_size)], axis=0)
    W -=  v
w = W
Q = np.mean([loss(W, x, y) for x, y in zip(X, Y)])
Q = round(Q, 17)


print(w)
print(Q)
print(Qe)