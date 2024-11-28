import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = -0.5
D2 = 2.0
mean2 = [0, 2]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 500
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

# вычисление оценок МО и ковариационных матриц
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N1, np.dot(a[0], a[1]) / N1],
                [np.dot(a[1], a[0]) / N1, np.dot(a[1], a[1]) / N1]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N2, np.dot(a[0], a[1]) / N2],
                [np.dot(a[1], a[0]) / N2, np.dot(a[1], a[1]) / N2]])

# для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

# здесь продолжайте программу
log_det_V1 = np.log(np.linalg.det(VV1))
log_det_V2 = np.log(np.linalg.det(VV2))
invV1 = np.linalg.inv(VV1)
invV2 = np.linalg.inv(VV2)

def a(x, log_det_V, invV, m):
    return -0.5 * (x-m) @ invV @ (x-m).T - 0.5 * log_det_V

predict = np.array([np.argmax([np.log(Py1 * L1) + a(x, log_det_V1, invV1, mm1), np.log(Py2 * L2) + a(x, log_det_V2, invV2, mm2)]) * 2 - 1 for x in data_x])
TP = np.sum((predict == 1) & (data_y == 1))
TN = np.sum((predict == -1) & (data_y == -1))
FP = np.sum((predict == 1) & (data_y == -1))
FN = np.sum((predict == -1) & (data_y == 1))

print(predict)
print(TP)
print(TN)
print(FP)
print(FN)