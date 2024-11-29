import numpy as np
from sklearn.cluster import AgglomerativeClustering

X = [(189, 185), (172, 205), (156, 221), (154, 245), (164, 265), (183, 275), (204, 276), (227, 271), (241, 255), (250, 229), (240, 197), (217, 183), (194, 202), (179, 224), (179, 248), (199, 249), (197, 227), (211, 214), (211, 242), (210, 265), (226, 237), (218, 196), (79, 106), (97, 132), (117, 159), (138, 174), (148, 163), (140, 145), (121, 123), (112, 108), (89, 92), (282, 162), (298, 180), (344, 154), (344, 113), (362, 67), (397, 77), (412, 121), (379, 112), (377, 148), (312, 130)]

X = np.array(X)

# здесь продолжайте программу
K = 3
clustering = AgglomerativeClustering(n_clusters=K, linkage="ward", metric="euclidean")
res = clustering.fit_predict(X)
X1, X2, X3  = [X[res == i] for i in range(K)]


Clusters = [[_x] for _x in X]
distances = np.array([[((len(_c1) * len(_c2)) / (len(_c1) * len(_c2))) * np.sum((np.mean(_c1, axis=0) - np.mean(_c2, axis=0))**2) for _c2 in Clusters] for _c1 in Clusters])
while len(Clusters) > K:
    np.fill_diagonal(distances, np.inf)
    min_value = np.min(distances)
    min_indexes = np.unravel_index(np.argmin(distances), distances.shape)
    _cl1 = Clusters.pop(min(min_indexes))
    _cl2 = Clusters.pop(max(min_indexes)-1)
    Clusters.append([*_cl1, *_cl2])
    distances = np.array([[((len(_c1) * len(_c2)) / (len(_c1) * len(_c2))) * np.sum((np.mean(_c1, axis=0) - np.mean(_c2, axis=0)) ** 2) for _c2 in Clusters] for _c1 in Clusters])
X1, X2, X3  = map(np.array, Clusters)
res = np.empty(len(X))
for j, _x in enumerate(X):
    for i, cluster in enumerate(Clusters):
        if np.isin(_x, cluster).all():
            res[j] = i



import matplotlib.pyplot as plt
colors = ['r', 'g', 'b', 'y', 'o', 'br']
for i, x in enumerate(Clusters):
    _x = np.array(x)
    if len(_x):
            plt.scatter(_x[:, 0], _x[:, 1], c=colors[i], label=f"Cluster {i + 1}")
plt.grid()
plt.legend()
plt.show()
