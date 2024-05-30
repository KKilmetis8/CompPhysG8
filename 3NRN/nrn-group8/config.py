"""
Created on Thu May 30 13:10:01 2024

@authors: diederick

Nuclear Reaction Network Simulation Config.  
This is the only file any user should touch.
Specifies parameters of simulation.

    temperature, float | int: A positive number giving the
        core temperature of the simulated star. This parameter 
        is only used when running a simulation for a single cycle,
        make sure to set kind = 'single' when doing so.

        In units of Giga-Kelvins
        Default: 0.015 GK
    
    metallicity, float | int: A positive number giving the
        core metallicity of the simulated star. This parameter 
        is only used when running a simulation for a single cycle,
        make sure to set kind = 'single' when doing so.

        With respect to the ISM metallicity
        Default: 1

    cycle, str: Which network of nuclear reactions to use 
        during the simulation, can either be 'pp' for the
        pp-chain or 'cno' for the CNO-cycle.

        Default: 'pp'

    init_abunds = str | np.ndarray: Initial abundances of the species 
        in the simulated stellar core. Can either be 'ism' for ISM
        abundances, or a custom array.
        The length of `init_abunds` depends on which network will be run:
        For `cycle = 'pp'`, it must have a length of 4 and for 
        `cycle = 'cno'`, it must have a length of 8.

        The species are ordered as follows:
        `cycle = 'pp'` : [1H, 2H, 3He, 4He]
        `cycle = 'cno'`: [1H, 12C, 13N, 13C, 14N, 15O, 15N, 4He]

        This parameter is only used when running a simulation for 
        a single cycle, make sure to set kind = 'single' when doing so.
    
    loud, bool: Boolean, wether to print progressbars in 
        the terminal or not during the simulation.
        Default: True

    simname, str | None: String or None, determines the name
        of the folder where results of the simulation(s) are saved.
        If set to None will generate a name with the structure
        '/{init_grid}_{kind}_at_{time_of_initialization}/'.
        Default: None

    kind, str: What kind of simulation to run.
        If `kind = 'single'`: will run a single simulation for the 
        nuclear reaction network given by `cycle`, with temperature
        `temperature` and initial abundances `init_abund`.

        If `kind = 'equality time'`: will run a series of simulations
        for both pp-chain and CNO-cycle, over the temperatures given 
        in `temperatures`. Calculates the equality time for each simulation.

        If `kind = 'dominance'`: will run a series of simulations
        for both pp-chain and CNO-cycle, over each temperature-metallicity 
        combination given in `temperatures` and `metallicities`. Finds which
        networks dominates at which temperature and metallicity.

    temperatures, list | np.ndarray: List or numpy array of 
        positive floats giving the temperatures for which to
        run simulations if kind = 'equality time' or 'dominance'.
        
        In units of Giga-Kelvins
        Default: numpy.linspace(12, 25, 10)*1e-3

    metallicities, list | np.ndarray: List or numpy array of 
        positive floats giving the metallicities for which to
        run simulations if kind = 'dominance'.
        
        With respect to the ISM metallicity
        Default: numpy.linspace(1, 1e2, 10)
"""

temperature = 0.015 #GK
metallicity = 1 #Z_ism
cycle       = 'pp'
init_abunds = 'ism'

loud    = True
simname = None
kind    = 'single' # 'equality time', 'dominance'

from numpy import linspace
temperatures  = linspace(12, 25, 10)*1e-3
metallicities = linspace(1, 1e2, 10)