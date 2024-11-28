import numpy as np


def func(x):
    return 0.1 * x - np.cos(x/2) + 0.4 * np.sin(3*x) + 5


np.random.seed(0)

x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
y = func(x) + np.random.normal(0, 0.2, len(x)) # значения функции по оси ординат

# здесь продолжайте программу
def K(r):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * r**2)

def r(x1, x2):
    return np.sum(np.abs(x1 - x2))

h = 0.5
x_est = x
y_est = np.zeros_like(x_est)

for i, x_e in enumerate(x_est):
    weights = np.array([K(r(xi, x_e) / h) for xi in x])
    y_est[i] = np.sum(weights * y) / np.sum(weights)

Q = np.mean((y_est - y)**2)


print(Q)
import matplotlib.pyplot as plt
plt.plot(x, y, 'ro', label = "Origin")
plt.plot(x_est, y_est, 'go', label = "Regression")
plt.legend()
plt.show()