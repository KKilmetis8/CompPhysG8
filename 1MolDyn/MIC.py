#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:16:59 2024

@author: konstantinos

TESTING MIC
"""

import numpy as np
import prelude as c
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
plt.rcParams['figure.figsize'] = [6 , 6]

nbodies = 10
L = 1
xs = np.random.random(nbodies) * L
ys = np.random.random(nbodies) * L

# We had the idea to create all the phantom poins, calculate all the drs
# sort them and keep the first N. We saw the wikipedia solution and
# it's so much better, it's obscene.
Tiles = np.tile( np.array([xs, ys]), 3**c.dims)
# pretend we implemented the permutations
rs = np.sqrt(xs**2 + ys**2)
np.sort(np.abs(rs[1] - rs))[:c.Nbodies]


# Stuff we copied over from wikipedia, but now understand
inv_L = 1/L
dx = xs[1] - xs
print( np.round(dx * inv_L,0) )
dx -= L * np.round(dx * inv_L,0)
dy = ys[1] - ys
dy -= L * np.round(dy * inv_L,0)
dr = np.sqrt(dx**2 + dy**2)


# Plotting code
# Sorry
fig, ax = plt.subplots()
# Tile 
xs2 = xs + 1
xs3 = xs - 1
ys2 = ys + 1
ys3 = ys - 1
ax.scatter(xs, ys, c= 'k')
ax.scatter(xs, ys2, c= 'k')
ax.scatter(xs, ys3, c= 'k')
ax.scatter(xs2, ys, c= 'k')
ax.scatter(xs2, ys2, c= 'k')
ax.scatter(xs2, ys3, c= 'k')
ax.scatter(xs3, ys, c= 'k')
ax.scatter(xs3, ys2, c= 'k')
ax.scatter(xs3, ys3, c= 'k')
ax.scatter(xs[1] , ys[1], c='r', s = 65)

# Square
plt.axvline(0, color = 'k', linestyle = '--')
plt.axvline(1, color = 'k', linestyle = '--')
plt.axhline(0, color = 'k', linestyle = '--')
plt.axhline(1, color = 'k', linestyle = '--')

# Color lines
circles = []
for i in range(nbodies):    
    plt.plot( [xs[1], xs[1] - dx[i]] , [ys[1], ys[1] - dy[i]] )
    # a = Circle((xs[1], ys[1]), dr[i], alpha = 1, color='g', fill = False, 
    #    transform = ax.transData)
    # ax.add_patch(a)


# Window
b = Rectangle((xs[1] - 1/2, ys[1] - 1/2), 1, 1, alpha = 0.3, color='b', fill = 'b', 
        transform = ax.transData)
ax.add_patch(b)

# Relevant range
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)
