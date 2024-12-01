import numpy as np

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


def model(W, X):
    return X @ W


def loss(w, x, y):
    return (model(w, x) - y) ** 2


def dL(w, x, y):
    xv = np.array([x ** n for n in range(len(w))])
    return 2 * (model(w, x) - y) * xv



coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

N = 5
lm_l2 = 2
sz = len(coord_x)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002])
n_iter = 500
lm = 0.02
batch_size = 20

np.random.seed(0)

X = np.array([[a ** n for n in range(N)] for a in coord_x])
Y = np.array(coord_y)
W = np.zeros(N)
IL = lm_l2 * np.eye(N)
IL[0][0] = 0

Qe = np.mean(loss(W, X, Y))
for i in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)
    kb = k + batch_size
    lossesK = np.mean(loss(W, X[k:kb], Y[k:kb]))
    W -= eta * ((2 / batch_size) * np.sum((X[k:kb] @ W - Y[k:kb]) * X[k:kb].T, axis=1) + IL @ W)
    Qe = lm * lossesK + (1 - lm) * Qe
w = W
Q = np.mean(loss(W, X, Y))


print(w)
print(Q)
print(Qe)