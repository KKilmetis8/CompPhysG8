import numpy as np
from itertools import product

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colors = ['r', 'g', 'b', 'gold', 'purple', 'darkorange']
    
unit = np.array([[0,0,0],
                 [1,1,0],
                 [0,1,1],
                 [1,0,1]]) * 0.5
ax.scatter(unit.T[0], unit.T[1], unit.T[2], c='k', alpha = 0.5)
iterations = np.array(list(product(range(3), repeat=3)))
all_points = unit.copy()

for i, iteration in enumerate(iterations[1:5]):
    points     = np.add(unit, iteration)
    ax.scatter(points.T[0], points.T[1], points.T[2], c = colors[i])
    all_points = np.vstack((all_points, points))
    
counter = 0
for i, point in enumerate(all_points):
    for j, point2 in enumerate(all_points):
        diff = np.subtract(point, point2)
        if np.linalg.norm(diff) == 0 and i !=j:
            print(i, point)
            print(j, point2)
            print('---')
            counter +=1
print('No of duplicates:', counter // 2)
