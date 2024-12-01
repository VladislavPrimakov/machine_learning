import numpy as np

x_test = [(9, 6), (2, 4), (-3, -1), (3, -2), (-3, 6), (7, -3), (6, 2)]

# здесь продолжайте программу
w = [-14, 7 , -5]
X = np.hstack([np.ones([len(x_test), 1]), x_test])
predict = [1 if y < 0 else -1 for y in w @ X.T]

print(predict)