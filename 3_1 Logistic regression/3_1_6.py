import numpy as np
x = np.array([1,190,70,1,300])
y = 1
m = 1
w = (m / y) * (x / (x @ x))

print(w)