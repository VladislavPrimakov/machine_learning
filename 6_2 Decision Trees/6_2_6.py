import numpy as np

np.random.seed(0)
X = np.random.randint(0, 2, size=200)


def fIG(X, t):
    N = len(X)
    _, c0 = np.unique(X, return_counts=True)
    S0 = 1 - np.sum([(Ni / N) ** 2 for Ni in c0])
    _, c1 = np.unique(X[:t], return_counts=True)
    S1 = 1 - np.sum([(Ni / t) ** 2 for Ni in c1])
    _, c2 = np.unique(X[t:], return_counts=True)
    S2 = 1 - np.sum([(Ni / (N - t)) ** 2 for Ni in c2])
    return S0 - (t / N) * S1 - ((N - t) / N) * S2


# здесь продолжайте программу
t = 150
IG = fIG(X, t)


print(IG)
