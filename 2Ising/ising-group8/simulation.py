from tqdm import tqdm
from importlib import reload
import numpy as np
from os import makedirs

import config as c
import prelude as p
import functions as f
import plotters as plot
reload(c), reload(p), reload(f), reload(plot)


init_energy = f.Hamiltonian(p.init_grid, c.J_coupling, c.H_ext)
makedirs(f"sims/{p.simname}/", exist_ok=True)

if c.kind == 'single':
    energies, avg_mags, last_grid, converged = f.metropolis(p.init_grid, c.MC_steps * p.MC_step, c.temperature, init_energy,
                                                  c.J_coupling, c.H_ext, buffer=c.buffer)
    
    np.savetxt(f'sims/{p.simname}/mean_spins_and_energies_after_eq.csv', np.reshape([energies, avg_mags], (2, len(energies))),
                delimiter=",", header='Mean spins and energies after equilibration for each step')
    
    avg_mags_after_eq = avg_mags[-c.buffer*p.MC_step:]
    energies_after_eq = energies[-c.buffer*p.MC_step:]

    tau, tau_err = f.calc_correlation_times([avg_mags_after_eq])

    steps = int(160*tau[0])
    measured_energies, measured_mags, _, _ = f.metropolis(last_grid, steps, c.temperature, energies[-1],
                                                      c.J_coupling, c.H_ext, equilibrate = False)

    mean_abs_spin  , mean_abs_spins_sigma   = f.obs_mean_abs_spin(           [measured_mags],                      tau)
    energy_per_spin, energy_per_spins_sigma = f.obs_energy_per_spin(         [measured_mags], [measured_energies], tau)
    mag_sus        , mag_sus_sigma          = f.obs_mag_susceptibility(      [measured_mags],                      tau)
    spec_heat_sus  , spec_heat_sus_sigma    = f.obs_spec_heat_susceptibility([measured_mags], [measured_energies], tau)

    observables = [tau    , mean_abs_spin       , energy_per_spin       , mag_sus      , spec_heat_sus      ]
    sigmas      = [tau_err, mean_abs_spins_sigma, energy_per_spins_sigma, mag_sus_sigma, spec_heat_sus_sigma]
    
    obs_rounded    = []
    sigmas_rounded = []
    # Rounding mean and standard deviation appropriately
    for i, (obs, sigma) in enumerate(zip(observables, sigmas)):
        j = 0
        while np.round(sigma, j) == 0:
            j += 1

        obs = np.round(obs,j)[0]
        sigma  = np.round(sigma,j)[0]

        if j == 0:
            obs = int(obs)
            sigma  = int(sigma)

        obs_rounded.append(obs)
        sigmas_rounded.append(sigma)
        

    main_result_string = ('# Inputted parameters\n'
    f'temperature = {c.temperature}\n'
    f'Nsize       = {c.Nsize}\n'
    f'MC_steps    = {c.MC_steps}\n'
    f'buffer      = {c.buffer}\n'
    f'J_coupling  = {c.J_coupling}\n'
    f'H_ext       = {c.H_ext}\n'
    '\n'
    '# Results\n'
    f'Auto-correlation time (tau): {obs_rounded[0]} ± {sigmas_rounded[0]}\n'
    f'mean absolute spin: {obs_rounded[1]} ± {sigmas_rounded[1]}\n'
    f'energy per spin: {obs_rounded[2]} ± {sigmas_rounded[2]}\n'
    f'magnetic susceptibility: {obs_rounded[3]} ± {sigmas_rounded[3]}\n'
    f'specific heat susceptibility: {obs_rounded[4]} ± {sigmas_rounded[4]}')

    with open(f'sims/{p.simname}/results.txt', 'w') as file: file.write(main_result_string)

elif c.kind == 'sweep':
    final_mags  = []
    convergence = []
    eq_mags     = []
    eq_energies = []
    last_grids  = []

    for temp in tqdm(c.temperatures, desc='Metropolis simulations') if c.loud else c.temperatures:
        energies, avg_mags, last_grid, converged = f.metropolis(p.init_grid, c.MC_steps * p.MC_step, temp, init_energy,
                                                      c.J_coupling, c.H_ext, buffer = c.buffer)
        final_mags.append( avg_mags[-c.buffer*p.MC_step:].mean() )
        convergence.append(converged)
        eq_mags.append(avg_mags[-c.buffer*p.MC_step:])
        eq_energies.append(energies[-c.buffer*p.MC_step:])
        last_grids.append(last_grid)

    np.savetxt(f'sims/{p.simname}/final_mags_and_temps.csv', np.reshape([c.temperatures, final_mags, convergence], (3, len(final_mags))).T,
                delimiter=",", header='Temperature [sim units], final average spin, convergence')
    np.savetxt(f'sims/{p.simname}/mean_spins_after_eq.csv', np.reshape(eq_mags, (len(eq_mags), len(eq_mags[0]))).T,
                delimiter=",", header='Mean spins after equilibration for each step and temperature')
    np.savetxt(f'sims/{p.simname}/energies_after_eq.csv', np.reshape(eq_energies, (len(eq_energies), len(eq_energies[0]))).T,
                delimiter=",", header='Energies after equilibration for each step and temperature')

    plot.eq_mags(final_mags, convergence)

    taus, taus_err = f.calc_correlation_times(eq_mags)

    measured_mags = []
    measured_energies = []
    for i in tqdm(range(len(c.temperatures)), desc='160*tau Metropolis simulations') if c.loud else range(len(c.temperatures)):
        steps = int(160*taus[i])
        temp = c.temperatures[i]
        energies, avg_mags, last_grid, converged = f.metropolis(last_grids[i], steps, temp, eq_energies[i][-1],
                                                      c.J_coupling, c.H_ext, equilibrate = False)
        measured_mags.append(avg_mags)
        measured_energies.append(energies)

    mean_abs_spins,   mean_abs_spins_sigmas   = f.obs_mean_abs_spin(           measured_mags,                    taus)
    energy_per_spins, energy_per_spins_sigmas = f.obs_energy_per_spin(         measured_mags, measured_energies, taus)
    mag_suss,         mag_suss_sigmas         = f.obs_mag_susceptibility(      measured_mags,                    taus)
    spec_heat_suss,   spec_heat_suss_sigmas   = f.obs_spec_heat_susceptibility(measured_mags, measured_energies, taus)

    observables = [taus    , mean_abs_spins       , energy_per_spins       , mag_suss       , spec_heat_suss       ]
    sigmas      = [taus_err, mean_abs_spins_sigmas, energy_per_spins_sigmas, mag_suss_sigmas, spec_heat_suss_sigmas]

    table_for_saving = np.zeros((len(mean_abs_spins), 2*len(observables)))
    for i, (obs, sigma) in enumerate(zip(observables, sigmas)):
        table_for_saving[:,2*i:2*(i+1)] = np.reshape([obs, sigma], (2,len(obs))).T

    np.savetxt(f'sims/{p.simname}/Observables.csv', table_for_saving, delimiter=',', 
                header=('Autocorrelation time, Mean absolute spin, energy per spin, magnetic susceptibility,'
                        'specific heat susceptibility, and their standard deviations per temperature.'))

    plot.main_result(observables[1:], sigmas[1:])

else:
    raise ValueError(f"Unrecognized option '{c.kind}' for the variable 'kind'")