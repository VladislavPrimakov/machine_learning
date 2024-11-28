import numpy as np

De = 0.2
Dx = 0.5
x0 = 7
y1 = 6.5
# pr1 = (y1 - x) / De
# pr2 = (x0 - x) / Dx
x = (y1*Dx + x0*De) / (De + Dx)

print(x)