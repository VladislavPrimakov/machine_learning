import numpy as np
from sklearn.tree import DecisionTreeRegressor

x = np.arange(-3, 3, 0.1).reshape(-1, 1)
y = 2 * np.cos(x) + 0.5 * np.sin(2 * x) - 0.2 * np.sin(4 * x)

# здесь продолжайте программу
max_depth = 3
T = 6
algs = []
s = np.array(y.ravel())
for i in range(T):
    algs.append(DecisionTreeRegressor(max_depth=max_depth))
    algs[i].fit(x, s)
    s -= algs[i].predict(x)
predict = np.sum([b.predict(x) for b in  algs], axis=0)
QT = np.mean((predict - y.ravel())**2)

print(QT)
import matplotlib.pyplot as plt
plt.plot(x, y, c='g', label="origin")
plt.plot(x, predict, c='r', label=f"AdaBoost (T = {T}, Depth = {max_depth})")
plt.legend()
plt.show()
