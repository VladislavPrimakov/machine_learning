import time
st = time.time()
import numpy as np

T = [[(365, 200), (390, 180), (350, 172), (400, 171)], [(77, 150), (100, 200), (50, 130)],
     [(250, 100), (170, 88), (280, 102), (230, 108)]]

data_x = np.array(
    [(48, 118), (74, 96), (103, 82), (135, 76), (162, 79), (184, 97), (206, 111), (231, 118), (251, 118),
     (275, 110), (298, 86), (320, 68), (344, 62), (376, 61), (403, 75), (424, 95), (440, 114), (254, 80),
     (219, 85), (288, 66), (260, 92), (201, 76), (162, 66), (127, 135), (97, 143), (83, 160), (82, 177),
     (88, 199),
     (105, 205), (135, 208), (151, 198), (157, 169), (153, 152), (117, 158), (106, 168), (106, 185),
     (123, 188),
     (125, 171), (139, 163), (139, 183), (358, 127), (328, 132), (313, 146), (300, 169), (300, 181),
     (308, 197),
     (326, 206), (339, 209), (370, 199), (380, 184), (380, 147), (343, 154), (329, 169), (332, 184),
     (345, 185),
     (363, 159), (361, 177), (344, 169), (311, 175), (351, 89), (134, 96)])

K = 3
max_iterations = 10
ma = np.array([np.mean(cluster, axis=0) for cluster in T])

for _ in range(max_iterations):
    predict = np.array([np.argmin(np.mean((_x - ma) ** 2, axis=1)) for _x in data_x])
    ma = [np.mean([*cluster, *data_x[predict == i]], axis=0) for i, cluster in enumerate(T)]

X = [[*cluster, *map(tuple, data_x[predict == i])] for i, cluster in enumerate(T)]

print(time.time() - st)

import matplotlib.pyplot as plt

colors = ['r', 'g', 'b']
for i, x in enumerate(X):
    _x = np.array(x)
    plt.scatter(_x[:, 0], _x[:, 1], c=colors[i], label=f"Cluster {i + 1}")
plt.grid()
plt.legend()
plt.show()
