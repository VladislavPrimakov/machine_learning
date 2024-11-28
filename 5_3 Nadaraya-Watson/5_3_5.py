import numpy as np


y = np.array([75, 76, 79, 82, 85, 81, 83, 86, 87, 85, 83, 80, 77, 79, 78, 81, 84])
x = np.arange(1, len(y) + 1)
predict_days = 10
x_est = np.arange(len(y) + 1, len(y) + 1 + predict_days)
y_est = []


# здесь продолжайте программу
def K(r):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * r ** 2)


def r(x1, x2):
    return np.sum(np.abs(x1 - x2))


h = 3
for i, x_e in enumerate(x_est):
    weights = np.array([K(r(xi, x_e) / h) for xi in [*x, *x_est[:len(y_est)]]])
    y_est.append(np.sum(weights * [*y, *y_est]) / np.sum(weights))
predict = y_est

print(predict)
import matplotlib.pyplot as plt

plt.plot(x, y, '-o', label="Origin")
plt.plot(x_est, y_est, '-o', label="Regression")
plt.grid()
plt.legend()
plt.show()
