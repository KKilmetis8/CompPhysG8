#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:04:38 2024

@author: konstantinos
"""
# Vanilla
import numpy as np

# Choc
import prelude as c
import ising

def avg_magnetization(grid: np.ndarray) -> float:
    '''
    Calculates the average magnetization per cell.

    Parameters
    ----------
    grid: np.ndarray, the (Nsize, Nsize) grid.

    Returns
    -------
    avg_magnetization: float, average magnetization per cell.
    '''
    return ising.total_magnetization(grid)/(c.Nsize**2)