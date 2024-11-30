import numpy as np

X = np.arange(-2, 3, 0.1)
Y = -X + 0.2 * X ** 2 - 0.5 * np.sin(4*X) + np.cos(2*X)

# здесь продолжайте программу
def fIG(X, Y, t):
    N = len(X)
    HR = np.sum((np.mean(Y) - Y)**2)
    leftY = Y[X <= t]
    HR1 = np.sum((np.mean(leftY) - leftY)**2)
    rightY = Y[X > t]
    HR2 = np.sum((np.mean(rightY) - rightY)**2)
    return [HR1, HR2, HR - (len(leftY) / N) * HR1 - (len(rightY) / N) * HR2]

HR1, HR2, IG = fIG(X, Y, 0)

print(HR1, HR2, IG)
