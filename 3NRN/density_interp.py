#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:37:47 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

Rsol_to_cm = 6.957e10 # [cm]
Msol_to_g = 2e33 # 1.989e33 # [g]

known_m = np.array([0.162, 1, 20])
known_r = np.array([0.196, 1, 12])
den = known_m/known_r * Msol_to_g/Rsol_to_cm**3
known_core_T = np.array([7,15,30]) * 1e6 # MK

den_of_T = CubicSpline(known_core_T, den, 
                       extrapolate = True)
T_range_1 = np.arange(8, 16+1)
T_range_2 = np.array([18, 20, 25, 30])
T_range_3 = np.arange(40, 140, step = 10)

T_range = list(T_range_1) + list(T_range_2) + list(T_range_3)
T_range = np.array(T_range)*1e6

interpolation = den_of_T(T_range)

if __name__ == '__main__':
    plt.plot(T_range, interpolation, c = 'darkorange', marker = 'h', ls = '--')
    plt.plot(known_core_T, den, c = 'k', ls = ':', marker = 'o')
