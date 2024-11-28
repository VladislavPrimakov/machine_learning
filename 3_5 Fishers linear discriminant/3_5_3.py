import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2, 0])
mean2 = np.array([1, 3, 1])
r = 0.7
D = 2.0
V = [[D, D * r, D*r*r], [D*r, D, D*r], [D*r*r, D*r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.zeros(N), np.ones(N)])

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2

# параметры для линейного дискриминанта Фишера
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

# здесь продолжайте программу
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
b = x1.T - mm1
VV = np.array([ [(a[0] @ a[0]) / (2*N), (a[0] @ a[1]) / (2*N), (a[0] @ a[2]) / (2*N)],
                [(a[1] @ a[0]) / (2*N), (a[1] @ a[1]) / (2*N), (a[1] @ a[2]) / (2*N)],
                [(a[2] @ a[0]) / (2*N), (a[2] @ a[1]) / (2*N), (a[2] @ a[2]) / (2*N)] ])
inv_V = np.linalg.inv(VV)

alpha1 = inv_V @ mm1
alpha2 = inv_V @ mm2
beta1 = np.log(Py1 * L1) - 0.5 * (mm1.T @ inv_V @ mm1)
beta2 = np.log(Py2 * L2) - 0.5 * (mm2.T @ inv_V @ mm2)

print(alpha1)
print(alpha2)
print(beta1)
print(beta2)