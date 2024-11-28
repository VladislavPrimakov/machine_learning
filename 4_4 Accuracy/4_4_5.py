import numpy as np
from sklearn.model_selection import train_test_split


def loss(w, x, y):
    M = x @ w * y
    return np.exp(-M)


def df(w, x, y):
    M = x @ w * y
    return -np.exp(-M) * x.T * y


np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.4
D1 = 2.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 3.0
mean2 = [2, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.array([[1, x[0], x[1]] for x in np.hstack([x1, x2]).T])
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

sz = len(x_train)  # размер обучающей выборки
w = [0.0, 0.0, 0.0]  # начальные весовые коэффициенты
eta = np.array([0.5, 0.01, 0.01])  # шаг обучения для каждого параметра w0, w1, w2
lm = 0.01  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
N = 500  # число итераций алгоритма SGD
batch_size = 10 # размер мини-батча (величина K = 10)

# здесь продолжайте программу
for i in range(N):
    k = np.random.randint(0, sz - batch_size - 1)
    kb = k + batch_size
    w -= eta * (1 / batch_size) * np.sum([df(w, _x, _y) for _x,_y in zip(x_train[k:kb], y_train[k:kb])], axis=0)
a = np.array(x_test @ w * y_test).sort()
mrgs = np.array(x_test @ w * y_test)
mrgs.sort()
acc = np.mean(np.sign(x_test @ w) == y_test)


print(w)
print(acc)
