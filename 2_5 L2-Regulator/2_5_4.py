import numpy as np


def func(x):
    return 0.1 * x + 0.1 * x ** 2 - 0.5 * np.sin(2*x) + 1 * np.cos(4*x) + 10

def model(W, X):
    return W[0] + np.sum([W[i] * X[i] for i in range(1, len(X))])

X = np.arange(-3.0, 4.1, 0.1)
Y = np.array(func(X))

N = 22
lm = 20

X = np.array([[a ** n for n in range(N)] for a in X])
IL = lm * np.eye(N)
IL[0][0] = 0

X_train = X[::2]
Y_train = Y[::2]

sz = len(X)
W = np.ones(N)

W = np.linalg.inv(X_train.T @ X_train + IL) @ X_train.T @ Y_train
Q = np.mean([(model(W, X[i]) - Y[i])**2 for i in range(0, len(X))])
w = W

print(w)
print(Q)