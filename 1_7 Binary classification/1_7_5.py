import numpy as np

x_test = np.array([(-5, 2), (-4, 6), (3, 2), (3, -3), (5, 5), (5, 2), (-1, 3)])
y_test = np.array([1, 1, 1, -1, -1, -1, -1])
w = np.array([-8/3, -2/3, 1])

# здесь продолжайте программу
x_test = np.column_stack((np.ones(len(x_test)), x_test))
Q = sum([1 if m < 0 else 0 for m in (y_test * (w @ x_test.T))])



print(Q)