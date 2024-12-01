import numpy as np

x_test = [(5, -3), (-3, 8), (3, 6), (0, 0), (5, 3), (-3, -1), (-3, 3)]

# здесь продолжайте программу
X = np.hstack([np.ones((len(x_test), 1)), np.array(x_test)])
w = [-33, 9, 13]
Y = np.array(w) @ X.T
predict = [-1 if y < 0 else 1 for y in Y]

print(predict)