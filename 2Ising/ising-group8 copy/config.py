"""
Created on Thu Apr 25 12:30:41 2024

@author: diederick, konstantinos

Ising Model Simulation Config.  
This is the only file any user should touch.
Specifies parameters of simulation.

    temperature, float | int: A positive number giving the
        temperature of the Ising grid. This parameter is only 
        used when running a simulation for a single grid, make 
        sure to set kind = 'single' when doing so.
        Default: 3
    
    Nsize, int: A positive integer giving the side-length
        of the Ising grid. 
        Default: 30

    MC_steps, int: A positive integer giving for how many 
        Monte-Carlo steps to run the simulation for. Where 1
        Monte-Carlo step equals Nsize**2 steps in the 
        Metropolis-Hastings algorithm. Chosen this way as to
        give every spin in the Nsize-by-Nsize grid an equal
        number of chances to flip. 
        Default: 10_000

    buffer, int: A positive integer giving the number of 
        Monte-Carlo steps to run after the Metropolis-Hastings
        algorithm has equilibrated. 
        Default: 100

    J_coupling, float: Non-negative number, the coupling 
        constant J appearing in the Hamiltonian. 
        Default: 1

    H_ext, float: A real number, the strength (and direction)
        of the external magnetic field H appearing in the 
        Hamiltonian. 
        Default: 0

    rngseed, int | None: A positive integer or None, the
        seed for grid generation. Setting this to 
        None will result in a different grid each 
        initialization. Note: rngseed has no effect on 
        determining if a grid transitions to some higher energy
        grid within the Metropolis-Hastings algorithm.
        Default: None

    loud, bool: Boolean, wether to print progressbars in 
        the terminal or not during the simulation.
        Default: True

    kind, str: String, either 'single' or 'sweep'. Determines
        what kind of simulation to run. If set to 'single' will
        run a single simulation for the temperature set in the
        'temperature' variable. If set to 'sweep' will run
        multiple simulations, one for each temperature set in
        the 'temperatures' variable.
        Default: 'sweep'

    init_grid, str: String, either '75% positive' or 'random',
        determines which kind of initial grid to use. 
        If set to '75% positive' will generate a Nsize-by-Nsize 
        grid of ones, where randomly 25% of the spins are changed
        to -1. If set to 'random', will generate a Nsize-by-Nsize
        grid where each spin has a 50/50 chance of being 1 or -1.
        Note: If running a sweep-simulation, each simulation will
        use the same initial grid.
        Default: '75% positive'

    simname, str | None: String or None, determines the name
        of the folder where results of the simulation(s) are saved.
        If set to None will generate a name with the structure
        '/{init_grid}_{kind}_at_{time_of_initialization}/'.
        Default: None

    temperatures, list | np.ndarray: List or numpy array of 
        positive floats giving the temperatures for which to
        run simulations if kind = 'single'.
        Default: numpy.arange(1, 4 + 0.2, 0.2)
"""

temperature = 3
Nsize       = 30
MC_steps    = 10_000
buffer      = 100

J_coupling = 1
H_ext      = 0

rngseed = None
loud    = True

kind      = 'sweep'
init_grid = '75% positive'
simname   = None

from numpy import arange
temperatures = arange(2,20+1, 1)