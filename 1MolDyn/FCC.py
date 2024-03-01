import numpy as np
from itertools import product

unit = np.array([[0,0,0],
                 [1,1,0],
                 [0,1,1],
                 [1,0,1]])

iterations = np.array(list(product(range(3), repeat=3)))
all_points = unit.copy()
for iteration in iterations[1:]:
    points     = unit + iteration
    all_points = np.vstack((all_points, points))
