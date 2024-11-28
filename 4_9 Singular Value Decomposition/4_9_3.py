import numpy as np

np.random.seed(0)

n_total = 1000 # число образов выборки
n_features = 200 # число признаков
lm = 0.01
table = np.zeros(shape=(n_total, n_features))

for _ in range(100):
    i, j = np.random.randint(0, n_total), np.random.randint(0, n_features)
    table[i, j] = np.random.randint(1, 10)

# матрицу table не менять

# здесь продолжайте программу
F = table.T @ table / table.shape[0]
L, W = np.linalg.eig(F)

idx = np.argsort(L)[::-1]
L = L[idx]
W = W[idx]

data_x = table @ W.T
data_x = data_x[:, L >= lm]

print(data_x.shape)
