import numpy as np


def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5

def loss(W, X, Y):
    return (X @ W - Y) ** 2

coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

sz = len(coord_x)
eta = np.array([0.1, 0.01, 0.001, 0.0001])
W = np.array([0., 0., 0., 0.])
N = 500
lm = 0.02
batch_size = 50
np.random.seed(0)

X = np.column_stack([np.ones(sz), coord_x, coord_x ** 2, coord_x ** 3])
Y = np.array(coord_y)
Qe = np.mean([loss(W, x, y) for x, y in zip(X, Y)])
for i in range(N):
    k = np.random.randint(0, sz-batch_size-1)
    lossesK = np.mean([loss(W, X[i], Y[i]) for i in range(k, k+batch_size)])
    W -= eta * ((2 / batch_size) * np.sum([(X[i] @ W - Y[i]) * X[i].T for i in range(k, k+batch_size)], axis=0))
    Qe = lm * lossesK + (1 - lm) * Qe
w = W
Q = np.mean([loss(W, x, y) for x, y in zip(X, Y)])


print(w)
print(Q)
print(Qe)