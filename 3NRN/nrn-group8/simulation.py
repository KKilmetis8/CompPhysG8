"""
Created on Thu May 30 13:15:19 2024

@authors: diederick & konstantinos
"""

from tqdm import tqdm
import numpy as np
from os import makedirs

import config  as c
import prelude as p
from functions.sim import run_network
import functions.plotter as plot

makedirs(f"sims/{p.simname}/", exist_ok=True)

if c.kind == 'single':
    Ys, t_eq, savetimes = run_network(c.cycle, c.temperature, p.init_abunds)
    
    if c.cycle == 'pp':
        np.savetxt(f'sims/{p.simname}/pp_evol.csv', np.hstack((savetimes.reshape((len(Ys),1)), Ys)),
                delimiter=",", header=('Abundance evolution pp-chain (t_eq = {t_eq} Gyr)\n'
                'time [s], 1H, 2H, 3He, 4He'))
        plot.pp_evol(Ys, t_eq, savetimes)
    if c.cycle == 'cno':
        plot.cno_evol(Ys, t_eq, savetimes)
        np.savetxt(f'sims/{p.simname}/cno_evol.csv', np.hstack((savetimes.reshape((len(Ys),1)), Ys)),
                delimiter=",", header=('Abundance evolution CNO-cycle (t_eq = {t_eq} Gyr)\n'
                'time [s], 1H, 12C, 13N, 13C, 14N, 15O, 15N, 4He'))

elif c.kind == 'equality time':
    cno_eqs = []
    cno_eqs_bad = []
    cno_T9 = []
    cno_T9s_bad = []
    pp_eqs = []
    for T9 in tqdm(c.temperatures) if c.loud else c.temperatures:
        _, eqpp, _ = run_network('pp', T9, init_step = c.init_step,
                                 max_step = c.max_step, max_time=c.max_time)
        pp_eqs.append(eqpp)
        try:
            _, eqcno, _ = run_network('cno', T9, initY = c.metallicity, 
                                      init_step = c.init_step,
                                      max_step = c.max_step, max_time=c.max_time)
            if eqcno >= 1e12/1e9:
                cno_T9s_bad.append(T9)
                cno_eqs_bad.append(1100)
            else:
                cno_eqs.append(eqcno)
                cno_T9.append(T9)

        except np.linalg.LinAlgError:
            continue
    
    cno_T9 = np.array(cno_T9) * 1e3
    cno_T9s_bad = np.array(cno_T9s_bad) * 1e3

    plot.t_eq_diagram(c.temperatures, pp_eqs, cno_T9, cno_eqs, cno_T9s_bad, cno_eqs_bad)

elif c.kind == 'dominance':
    eq_pps  = -np.ones((len(c.temperatures), len(c.metallicities)))
    eq_cnos = -np.ones_like(eq_pps)

    for i,metallicity in tqdm(enumerate(c.metallicities)) if c.loud else enumerate(c.metallicities):
        for j,T9 in enumerate(c.temperatures):
            _, eq_pps[i,j], _ = run_network('pp', T9, max_step = c.max_step,
                                            init_step = c.init_step,
                                            max_time=c.max_time)
            try:
                _, eq_cnos[i,j], _ = run_network('cno', T9, 
                                                 initY = float(metallicity), 
                                                 init_step = c.init_step,
                                                 max_step = c.max_step, 
                                                 max_time = c.max_time)
            except:
                continue
    
    np.savetxt(f'sims/{p.simname}/pp_equality_times.csv', np.hstack((c.metallicities.reshape((len(eq_pps),1)), eq_pps)),
                delimiter=",", header=('1H-4He equality times in Gyr for the pp-chain \n'
                'metallicity,'+ ', '.join([str(T9) for T9 in c.temperatures])))
    np.savetxt(f'sims/{p.simname}/cno_equality_times.csv', np.hstack((c.metallicities.reshape((len(eq_cnos),1)), eq_cnos)),
                delimiter=",", header=('1H-4He equality times in Gyr for the CNO-cycle \n'
                'metallicity,'+ ', '.join([str(T9) for T9 in c.temperatures])))
    plot.dominance_diagram(c.temperatures, c.metallicities, eq_pps, eq_cnos)

else:
    raise ValueError(f"Unrecognized option '{c.kind}' for the variable 'kind'")