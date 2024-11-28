import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.2
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 2500
N2 = 1500
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.4, shuffle=True)

# здесь продолжайте программу
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)
predict = np.array(model.predict(x_test))
w = np.array([model.intercept_[0], *model.coef_[0]])

TP = np.sum((predict == 1) & (y_test == 1))
TN = np.sum((predict == -1) & (y_test == -1))
FP = np.sum((predict == 1) & (y_test == -1))
FN = np.sum((predict == -1) & (y_test == 1))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F = (2*precision*recall) / (precision+recall)
Fb = ((1 + 0.5**2) * precision * recall) /(0.5**2 * precision + recall)

print(w)
print(precision)
print(recall)
print(F)
print(Fb)