import numpy as np

De = 0.3
Dx = 0.8
x0 = 11
y1 = 10.6
y2 = 11.2
# pr1 = (-2x + y2 + y1) / De
# pr2 = (x0 - x) / Dx
x = (x0*De + y1*Dx + y2*Dx) / (De + 2*Dx)

print(x)