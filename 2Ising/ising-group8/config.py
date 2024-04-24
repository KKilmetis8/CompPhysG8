import numpy as np
"""
Documentation goes here
"""

temperature = 3
Nsize       = 50
MC_steps    = 10_000
buffer      = 100

J_coupling = 1
H_ext      = 0

rngseed = None
loud    = True

kind      = 'sweep' # single or sweep
init_grid = '75% positive' # '75% positive' or 'random'
simname   = None # If None: set to {init_grid}_{kind}_at_{time_of_initialization}

# Used in sweep
temperatures = np.arange(1,4+0.3, 0.3)