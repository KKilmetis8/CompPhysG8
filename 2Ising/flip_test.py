#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:01:42 2024

@author: konstantinos
"""
# Vanilla 
import numpy as np

# Choc
import prelude as c
import plotters as plot
import ising

# Grid initialization
rng  = np.random.default_rng(seed=c.rngseed)
grid = np.sign(rng.random((c.Nsize, c.Nsize)) - 0.5)

# random spin-flip
flip_row, flip_col = rng.integers(c.Nsize, size=2)
flipped = grid.copy()
flipped[flip_row][flip_col] *= -1

plot.grid(grid, title='Initial grid')
ax, _ = plot.grid(flipped, title=f"{flip_row},{flip_col} flipped")

flipped_coords_x = [flip_col-0.5, flip_col+0.5, flip_col+0.5, flip_col-0.5, 
                    flip_col-0.5]
flipped_coords_y = [flip_row+0.5, flip_row+0.5, flip_row-0.5, flip_row-0.5, 
                    flip_row+0.5]

ax.plot(flipped_coords_x, flipped_coords_y, "yellow", lw=1)

plot.grid(ising.neighbor_sum(grid), title='Neighbors summed')