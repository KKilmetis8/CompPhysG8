"""
Created on Tue Apr 23 17:27:36 2024

@authors: diederick & konstantinos
"""

import numpy as np
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
import numba
from tqdm import tqdm

import config as c
import prelude as p

kernel = 0.5 * np.array([[0, 1, 0], 
                         [1, 0, 1], 
                         [0, 1, 0]])

def Hamiltonian(grid: np.ndarray, J: float = c.J_coupling, H: float = c.H_ext) -> float:
    '''
    Calculates the Hamiltonian of the grid.

    Hamiltonian = -J * sum(s_i, s_j) - H * sum(s_j)
    
    Parameters
    ----------
    grid: np.ndarray, the (Nsize, Nsize) grid.

    J: float, coupling constant between spins,
       default to J = k_B = 1.

    H: float, external magnetic field.

    Returns
    -------
    fancy_H: Calculated Hamiltonian of the grid
    '''
    conv = convolve2d(grid, kernel, boundary='wrap', mode='same')
    coupling_term      = -J * (grid*conv).sum()
    ext_mag_field_term = -H * grid.sum()
    return coupling_term + ext_mag_field_term

@numba.njit(nopython=True, nogil=True)
def metropolis(init_grid: np.ndarray[int], steps: int, temperature: float, init_energy: float, 
               J: float = c.J_coupling, H: float = c.H_ext, rtol: float=1e-3, atol: float=1e-4,
               chunk_size: int = 20, buffer: int = 100, equilibrate: bool = True) \
               -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], bool]:
    '''
    Performs the Metropolis algorithm.

    Includes an equilibration process, where the energies are compared
    between chunk_size * MC_step windows and the algorithm is terminated
    if the energies differ within preset tolerances. The simulation
    is considered to be equilibrated at this point.

    Parameters
    ----------
    init_grid: np.ndarray, the starting (Nsize, Nsize) grid. (x_0)

    steps: int, how many steps to perform the Metropolis algorithm,
           usually given as a multiple of MC-steps.

    temperature: float, the temperature of grid.

    init_energy: float, the energy of the initial grid `init_grid`,
                 calculated using the `Hamiltonian` function.
    
    J: float, the coupling constant. Defaulted to the one given in `config.py`
       Default: config.J_coupling

    H: float, the external magnetic field. Defaulted to the one given in `config.py`
       Default: config.H_ext
    
    rtol: float, the accepted relative tolerance used in the equilibration condition.
          Default: 1e-3

    atol: float, the accepted absolute tolerance used in the equilibration condition.
          Default: 1e-4
    
    chunk_size: int, the window size in MC_steps between which energies are compared
                for the equilibration.
                Default: 20

    buffer: int, how many MC_steps to continue running after the equilibration is done.
            Useful for determining the autocorrelation time.
            Default: 100
            
    equilibrate: bool, wether to check if the system is equilibrated or not. 
                 Usually set to True, only really set to False for measuring observables.
                 Default: True

    Returns
    -------
    all_energies[:i_eq]: array of floats, the energy at each simulation step,
                         up to the point of equilibration.
    
    all_avg_mags[:i_eq]: array of floats, the average spin magnitude at each simulation step,
                         up to the point of equilibration.
    
    last_grid: array of floats, the final grid generated in algorithm.
    
    converged: bool, wether the simulation equilibrated or not.
    '''
    beta = 1/temperature
    trial = chunk_size*p.MC_step
    
    # Initialize holders
    all_energies = np.zeros(steps+1)
    all_avg_mags = np.zeros(steps+1)
    all_energies[0] = init_energy
    all_avg_mags[0] = init_grid.mean()
    
    last_grid = init_grid.copy()
    i_eq = steps
    converged = False
    for i in range(steps):
        # Equilibration Check ----------------------------------------------->
        if equilibrate:
            if i > i_eq + buffer*p.MC_step:
                # If equilibrated: run for 'buffer' extra MC-steps
                break
            
            if i > (2*trial) and np.mod(i,2*trial)==0:
                old_energy = np.abs(np.mean(all_energies[i-2*trial:i - trial]))
                new_energy = np.abs(np.mean(all_energies[i-trial:i]))
                abs_diff = np.abs(old_energy - new_energy)
                rel_diff = abs_diff/old_energy

                #print(rel_diff, rtol, abs_diff, atol)
                
                if not converged and (rel_diff < rtol or abs_diff < atol):
                    converged = True
                    i_eq = i
                    # if rel_diff < rtol: print('rel-eq:', rel_diff, rtol)
                    # if abs_diff < atol: print('abs-eq:', abs_diff, atol)
                    
            
        # Generate single-flip grid ------------------------------------------>
        flip_row = np.random.randint(0,c.Nsize)
        flip_col = np.random.randint(0,c.Nsize)
        new_grid = last_grid.copy()
        new_grid[flip_row,flip_col] *= -1
        
        # Calc new energy  --------------------------------------------------->
        current_spin = last_grid[flip_row,flip_col]

        top    = last_grid[(flip_row-1)%len(last_grid), flip_col]
        left   = last_grid[flip_row, (flip_col-1)%len(last_grid)]
        right  = last_grid[flip_row, (flip_col+1)%len(last_grid)]
        bottom = last_grid[(flip_row+1)%len(last_grid), flip_col]

        neighbors_sum = top + left + right + bottom
        delta_E = 2*(J*neighbors_sum + H)*current_spin

        # Accept or Deny ----------------------------------------------------->
        prob_ratio = np.exp(-delta_E * beta)
        if delta_E < 0 or np.random.random() < prob_ratio:
            all_energies[i+1] = all_energies[i] + delta_E
            all_avg_mags[i+1] = new_grid.mean()
            last_grid = new_grid.copy()
            continue
        
        # Store
        all_energies[i+1] = all_energies[i]
        all_avg_mags[i+1] = last_grid.mean()

    return all_energies[:i], all_avg_mags[:i], last_grid, converged


@numba.njit(parallel=True)
def chi(steps: np.ndarray[float], arr: np.ndarray[float]) -> np.ndarray[float]:
    """
    Calculates the parameter chi for each step in steps over the array arr

    Parameters
    ----------
    steps : np.ndarray[float]
        The steps over which to calculate chi.
    arr : np.ndarray[float]
        The array to use to calculate chi.

    Returns
    -------
    chis : np.ndarray[float]
        The calculated chi values.
    """    
    chis = np.zeros(len(steps))
    for i in numba.prange(len(steps)):
        t_diff  = len(arr) - steps[i]
        term1   = (1/t_diff)*(arr[:t_diff] * arr[steps[i]:]).sum()
        term2   = (1/t_diff**2) * (arr[:t_diff].sum()*arr[steps[i]:].sum())
        chis[i] = term1 - term2
    return chis

def exp_model(t: float, chi0: float, tau: float) -> float:
    """
    Exponential modelling fitting function for finding the
    autocorrelation time.

    Parameters
    ----------
    t : float
        The time steps at which to calculate chi.
    chi0 : float
        The value of chi at t=0.
    tau : float
        The autocorrelation time.

    Returns
    -------
    chi: float
        The calculated chi values
    """    
    return chi0 * np.exp(-t/tau)

def calc_correlation_times(eq_mags: list[np.ndarray[float]]) -> list[float]:
    """
    Calculates the autocorrelation time for each set of 
    equilibrated mean spins in eq_mags.

    Parameters
    ----------
    eq_mags : list[np.ndarray[float]]
        Equilibrated mean spins as resulted from the simulations.

    Returns
    -------
    list[float]
        Autocorrelation times corresponding to the data in each array of eq_mags.
    """    
    taus = []
    tau_errs = []
    for ms in tqdm(eq_mags, desc='Correlation times') if c.loud else eq_mags:
        ts = np.arange(len(ms))
        chis = chi(ts, ms)
        up_to = np.where(chis <= 1e-7)[0]
        
        if len(up_to) != 0:
            up_to = up_to[0]
        else:
            up_to = -1

        popt, pcov = curve_fit(exp_model,ts[:up_to], chis[:up_to])
        tau = popt[-1]
        taus.append(tau)
        tau_errs.append(np.sqrt(pcov[-1,-1]))
    return taus, tau_errs

def obs_mean_abs_spin(eq_mags: list[np.ndarray[float]], taus: np.ndarray[float]) \
    -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Calculates the observable mean absolute spin for each set of 
    equilibrated mean spins in eq_mags.

    Parameters
    ----------
    eq_mags : list[np.ndarray[float]]
        Equilibrated mean spins as resulted from the simulations.
    taus : np.ndarray[float]
        Autocorrelation times corresponding to the data in each array of eq_mags.

    Returns
    -------
    mean_abs_spins, sigmas : tuple[np.ndarray[float], np.ndarray[float]]
        The calculated mean absolute spins and the corresponding standard deviations.
    """    
    mean_abs_spins = np.zeros(len(eq_mags))
    sigmas = np.zeros(len(eq_mags))
    for i, m in enumerate(eq_mags):
        mean_abs_spins[i] = np.abs(m.mean())
        var = np.abs( np.mean(m**2) - np.mean(m)**2)
        sigmas[i] = np.sqrt(2*taus[i]/len(m) * var)
    
    return mean_abs_spins, sigmas

def obs_energy_per_spin(eq_mags: list[np.ndarray[float]], eq_energies: list[np.ndarray[float]], taus: np.ndarray[float]) \
    -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Calculates the mean energy per spin for each set of 
    equilibrated mean spins in eq_mags and 
    equilibrated energies in eq_energies.

    Parameters
    ----------
    eq_mags : list[np.ndarray[float]]
        Equilibrated mean spins as resulted from the simulations.
    eq_energies : list[np.ndarray[float]]
        Equilibrated energies as resulted from the simulations.
    taus : np.ndarray[float]
        Autocorrelation times corresponding to the data in each array of eq_mags and eq_energies.

    Returns
    -------
    energy_per_spins, sigmas : tuple[np.ndarray[float], np.ndarray[float]]
        The calculated energy per spin and the corresponding standard deviations.
    """    
    
    energy_per_spins = np.zeros(len(eq_mags))
    sigmas = np.zeros(len(eq_mags))
    for i in range(len(eq_energies)):
        mean_e = np.array(eq_energies[i]) / c.Nsize**2  
        energy_per_spins[i] = mean_e.mean()
        var = np.abs( np.mean(mean_e**2) - np.mean(mean_e)**2)
        sigmas[i] = np.sqrt(2*taus[i]/len(mean_e) * var)
    
    return energy_per_spins, sigmas

def obs_mag_susceptibility(eq_mags: list[np.ndarray[float]], taus: np.ndarray[float]) \
    -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Calculates the magnetic susceptibility for each set of 
    equilibrated mean spins in eq_mags.

    Parameters
    ----------
    eq_mags : list[np.ndarray[float]]
        Equilibrated mean spins as resulted from the simulations.
    taus : np.ndarray[float]
        Autocorrelation times corresponding to the data in each array of eq_mags.

    Returns
    -------
    mag_suss, sigmas : tuple[np.ndarray[float], np.ndarray[float]]
        The calculated magnetic susceptibilities and the corresponding standard deviations.
    """    

    mag_suss = np.zeros(len(eq_mags))
    sigmas = np.zeros(len(eq_mags))
    for i, m in enumerate(eq_mags): # loops over runs
        chunk_size = int(16*taus[i])
        chunk_chi_ms = []

        if c.kind == 'sweep':
            temp = c.temperatures[i]
        elif c.kind == 'single':
            temp = c.temperature

        prefactor = c.Nsize**2/temp
        for j in range(chunk_size, len(m), chunk_size): # loops over tau chunks
            chunk_m = m[j-chunk_size:j]
            chunk_chi_ms.append(prefactor*(np.mean(chunk_m**2) - np.mean(chunk_m)**2))
        
        sigmas[i] = np.std(chunk_chi_ms)
        mag_suss[i] = np.mean(chunk_chi_ms)
    return mag_suss, sigmas

def obs_spec_heat_susceptibility(eq_mags: list[np.ndarray[float]], eq_energies: list[np.ndarray[float]], taus: np.ndarray[float]) \
    -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Calculates the specific heat susceptibility for each set of 
    equilibrated mean spins in eq_mags and 
    equilibrated energies in eq_energies.

    Parameters
    ----------
    eq_mags : list[np.ndarray[float]]
        Equilibrated mean spins as resulted from the simulations.
    eq_energies : list[np.ndarray[float]]
        Equilibrated energies as resulted from the simulations.
    taus : np.ndarray[float]
        Autocorrelation times corresponding to the data in each array of eq_mags and eq_energies.

    Returns
    -------
    spec_heat_suss, sigmas : tuple[np.ndarray[float], np.ndarray[float]]
        The calculated specific heat susceptibilities and the corresponding standard deviations.
    """    
    
    spec_heat_suss = np.zeros(len(eq_mags))
    sigmas = np.zeros(len(eq_mags))

    for i, energies in enumerate(eq_energies): # loops over runs
        chunk_size = int(16*taus[i])
        chunk_cs = []

        if c.kind == 'sweep':
            temp = c.temperatures[i]
        elif c.kind == 'single':
            temp = c.temperature

        prefactor = 1/temp**2 * 1/c.Nsize**2
        for j in range(chunk_size, len(energies), chunk_size): # loops over tau chunks
            chunk_c = energies[j-chunk_size:j]
            chunk_cs.append(prefactor*(np.mean(chunk_c**2) - np.mean(chunk_c)**2))
        
        sigmas[i] = np.std(chunk_cs)
        spec_heat_suss[i] = np.mean(chunk_cs)
    
    return spec_heat_suss, sigmas