"""
Created on Thu May 30 13:22:12 2024

@authors: diederick & konstantinos
"""

import config as c
import time

year = 365*24*60*60

# initial abundances
if c.init_abunds == 'ism':
    init_abunds = None
elif c.metallicity != 1:
    init_abunds = c.metallicity
else:
    init_abunds = c.init_abunds

# simname
simname = c.simname
if simname is None:
    time_of_creation = time.strftime('%d-%h-%H%M%S', time.localtime())
    simname = f'{c.init_grid}_{c.kind}_at_{time_of_creation}'

# plotting stuff
import matplotlib.pyplot as plt
plt.rcParams['text.usetex']     = False # False makes it fast, True makes it slow.
plt.rcParams['figure.dpi']      = 300
plt.rcParams['font.family']     = 'Times New Roman'
plt.rcParams['figure.figsize']  = [6 , 5]
plt.rcParams['axes.facecolor']  = 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
AEK = '#F1C410'