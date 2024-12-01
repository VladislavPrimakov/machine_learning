import numpy as np


def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

def loss(W, X, Y):
    return (X @ W - Y) ** 2


coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)

sz = len(coord_x)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01])
W = np.array([0., 0., 0., 0., 0.])
N = 500
lm = 0.02
np.random.seed(0)

X = np.column_stack([np.ones(sz), coord_x, coord_x ** 2, np.cos(2 * coord_x), np.sin(2 * coord_x)])
Y = np.array(coord_y)
Qe = np.mean([loss(W, x, y) for x, y in zip(X, Y)])
for i in range(N):
    k = np.random.randint(0, sz-1)
    lossK = loss(W, X[k], Y[k])
    W -= eta * (2 * ((X[k] @ W - Y[k]) * X[k].T))
    Qe = lm * lossK + (1 - lm) * Qe
w = W
Q = np.mean([loss(W, x, y) for x, y in zip(X, Y)])


print(w)
print(Q)
print(Qe)