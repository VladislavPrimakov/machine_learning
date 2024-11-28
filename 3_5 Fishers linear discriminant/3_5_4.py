import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2])
mean2 = np.array([-3, -1])
mean3 = np.array([1, 2])

r = 0.5
D = 1.0
V = [[D, D * r], [D*r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T
x3 = np.random.multivariate_normal(mean3, V, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2, x3

# параметры для линейного дискриминанта Фишера
Py = [0.2, 0.4, 0.4]
L = [1, 1, 1]

# здесь продолжайте программу
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)
mm = [mm1, mm2, mm3]

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T, (x3.T - mm3).T])
VV = np.array([[(a[0] @ a[0]) / (3*N), (a[0] @ a[1]) / (3*N)],
                [(a[1] @ a[0]) / (3*N), (a[1] @ a[1]) / (3*N)]])
inv_V = np.linalg.inv(VV)

alpha = [inv_V @ mm[i] for i in range(3)]
beta = [np.log(Py[i] * L[i]) - 0.5 * (mm[i].T @ inv_V @ mm[i]) for i in range(3)]

def a(x):
    return np.argmax([x.T @ alpha[i] + beta[i] for i in range(3)])

predict = [a(x) for x in x_train]
Q = np.sum(predict != y_train)

print(predict)
print(Q)