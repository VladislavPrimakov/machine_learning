import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


np.random.seed(0)
n_feature = 2

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 7]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [4, 2]
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N1, N2 = 1000, 1200
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

# здесь продолжайте программу
max_depth = 3
T = 10
alpha = []
algs = []
w = np.ones(len(x_train)) / len(x_train)
for i in range(T):
    algs.append(DecisionTreeClassifier(criterion='gini', max_depth=max_depth))
    algs[i].fit(x_train, y_train, sample_weight=w)
    predicted = algs[i].predict(x_train)
    N = np.sum(w * (predicted != y_train))
    if N == 0:
        N = 1e-10
    alpha.append(0.5 * np.log((1 - N) / N))
    w = w * np.exp(-alpha[i] * y_train * predicted)
    w = w / np.sum(w)

predict = np.sign(np.sum([a * b.predict(x_test) for a, b in zip(alpha, algs)], axis = 0))
Q = np.sum(predict != y_test)


print(Q)
import matplotlib.pyplot as plt
uniq = np.unique(predict)
for cl, c in zip(uniq, [('b', 'r'), ('orange', 'brown')]):
    indexes_true_predicted = (predict == cl) & (predict == y_test)
    plt.scatter(x_test[indexes_true_predicted, 0], x_test[indexes_true_predicted, 1], c=c[0], label=f"Class {cl}")
    indexes_false_predicted = (predict == cl) & (predict != y_test)
    plt.scatter(x_test[indexes_false_predicted, 0], x_test[indexes_false_predicted, 1], c=c[1], label=f"Wrong predicted for class {cl}")

plt.legend()
plt.show()
