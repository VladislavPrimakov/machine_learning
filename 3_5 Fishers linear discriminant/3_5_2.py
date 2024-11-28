import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = [1, -2]
mean2 = [1, 3]
r = 0.7
D = 2.0
V = [[D, D * r], [D * r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# вычисление оценок МО и ковариационной матрицы

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
VV = np.array([[(a[0] @ a[0]) / (2*N), (a[0] @ a[1]) / (2*N)],
                [(a[1] @ a[0]) / (2*N), (a[1] @ a[1]) / (2*N)]])


inv_V = np.linalg.inv(VV)

def a(x, mm):
     return -0.5 * (mm.T @ inv_V @ mm) + (x.T @ inv_V @ mm)

predict = [np.argmax([a(x, mm1), a(x, mm2)]) * 2 - 1 for x in x_train]
Q = np.sum(y_train != predict)

print(predict)
print(Q)