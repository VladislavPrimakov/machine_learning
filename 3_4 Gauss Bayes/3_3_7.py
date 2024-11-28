import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [-1, -2, -1]
V1 = [[D1, D1 * r1, D1*r1*r1], [D1 * r1, D1, D1*r1], [D1*r1*r1, D1*r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [1, 2, 1]
V2 = [[D2, D2 * r2, D2*r2*r2], [D2 * r2, D2, D2*r2], [D2*r2*r2, D2*r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# здесь продолжайте программу
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[(a[0] @ a[0]) / N, (a[0] @ a[1]) / N, (a[0] @ a[2]) / N],
                [(a[1] @ a[0]) / N, (a[1] @ a[1]) / N, (a[1] @ a[2]) / N],
                [(a[2] @ a[0]) / N, (a[2] @ a[1]) / N, (a[2] @ a[2]) / N]])
a = (x2.T - mm2).T
VV2 = np.array([[(a[0] @ a[0]) / N, (a[0] @ a[1]) / N, (a[0] @ a[2]) / N],
                [(a[1] @ a[0]) / N, (a[1] @ a[1]) / N, (a[1] @ a[2]) / N],
                [(a[2] @ a[0]) / N, (a[2] @ a[1]) / N, (a[2] @ a[2]) / N]])

log_det_V1 = np.log(np.linalg.det(VV1))
log_det_V2 = np.log(np.linalg.det(VV2))

invV1 = np.linalg.inv(VV1)
invV2 = np.linalg.inv(VV2)

def a(x, log_det_V, invV, m):
    return -0.5 * (x-m) @ invV @ (x-m).T - 0.5 * log_det_V

predict = [np.argmax([np.log(Py1 * L1) + a(x, log_det_V1, invV1, mm1), np.log(Py2 * L2) + a(x, log_det_V2, invV2, mm2)]) * 2 - 1 for x in x_train]
Q = np.sum(y_train != predict)

print(predict)
print(Q)