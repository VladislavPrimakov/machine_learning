import numpy as np
from sklearn.cluster import DBSCAN

X = [(58, 138), (74, 96), (103, 82), (135, 76), (162, 79), (184, 97), (206, 111), (231, 118), (251, 118),  (275, 110), (298, 86), (320, 68), (344, 62), (376, 61), (403, 75), (414, 90), (430, 100), (254, 80), (219, 85), (288, 66), (260, 92), (201, 76), (162, 66), (127, 135), (97, 143), (83, 160), (82, 177), (88, 199), (105, 205), (135, 208), (151, 198), (157, 169), (153, 152), (117, 158), (106, 168), (106, 185), (123, 188), (125, 171), (139, 163), (139, 183), (358, 127), (328, 132), (313, 146), (300, 169), (300, 181), (308, 197), (326, 206), (339, 209), (370, 199), (380, 184), (380, 147), (343, 154), (329, 169), (332, 184), (345, 185), (363, 159), (361, 177), (344, 169), (311, 175), (351, 89), (134, 96)]
X = np.array(X)

# здесь продолжайте программу
stop = False
res = None
for _eps in range(1, 200):
    if stop:
        break
    for _samples in range(1, 30):
        clustering = DBSCAN(eps=_eps, min_samples=_samples, metric='euclidean')
        res = clustering.fit_predict(X)
        uniq = np.unique(res)
        if -1 not in uniq and len(uniq) == 3:
            print(_eps, _samples)
            stop = True
            break


unique = np.unique(res)
predict = [X[res == i] for i in unique]
import matplotlib.pyplot as plt
colors = ['r', 'g', 'b', 'y', 'g']
for i, x in enumerate(predict):
    _x = np.array(x)
    plt.scatter(_x[:, 0], _x[:, 1], c=colors[i], label=f"Cluster {i + 1}" if unique[i] != -1 else "Noise")
plt.grid()
plt.legend()
plt.show()
