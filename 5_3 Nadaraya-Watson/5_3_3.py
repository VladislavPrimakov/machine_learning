import numpy as np

# Data points
x = np.array([0, 1, 2, 3])
y = np.array([0.5, 0.8, 0.6, 0.2])

# Estimation points
x_est = np.arange(0, 3.1, 0.1)

def K(r, h):
    if 0 <= r <= h:
        return 1 - r / h
    else:
        return 0

def r(x1, x2):
    return np.sum(np.abs(x1 - x2))

h = 1
y_est = np.zeros_like(x_est)

for i, x_e in enumerate(x_est):
    weights = np.array([K(r(xi, x_e), h) for xi in x])
    y_est[i] = np.sum(weights * y) / np.sum(weights)

print(weights)
print(y_est)