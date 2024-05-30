"""
Created on Thu May 30 13:43:26 2024

@author: diederick & konstantinos

Plotting functions
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from prelude import AEK, simname, year
from numpy import where, argmin, abs

def pp_evol(Ys, equality_time, savetimes):
    """Generates a plot showing the 
    evolution of the abundances of the species
    in the pp-chain.

    Parameters
    ----------
    Ys : array of floats
        Abundances of the species over time
    equality_time : float
        The time at which 1H and 4He approximately
        had the same abundance
    savetimes : array of floats
        Timesteps at which the abundances were saved
    """    
    labels = ["H", "D", "$^{3}$He", "$^{4}$He"]
    colors = ["dodgerblue", "dodgerblue", AEK, AEK]
    linestyles = ["-","--","--","-"]
    fig = plt.figure(tight_layout=True)

    try:
        stop = where(Ys.T[0] == 0)[0][0]
    except:
        stop = -1

    unit = 1 / (year * 1e9)
    for i,abundances in enumerate(Ys.T):
        plt.plot(savetimes[:stop]*unit, abundances[:stop], 
                label = labels[i], ls=linestyles[i], color=colors[i], marker='', linewidth=2.5)

    eq_idx =  argmin(abs( equality_time - savetimes*unit))
    plt.scatter(equality_time, Ys.T[-1][eq_idx]
                , marker = 'h', c = 'gold', ec = 'dodgerblue', 
                linewidth = 2, 
                s = 200, zorder = 4)


    plt.grid()
    plt.ylim(1e-8,10)
    plt.xlim(1e-3,1e3)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Abundance', fontsize = 14)
    plt.xlabel('time [Gyrs]', fontsize = 14)
    plt.savefig(f'sims/{simname}/pp_evol.pdf')
    plt.close(fig)
    

def cno_evol(Ys, equality_time, savetimes):
    """Generates a plot showing the 
    evolution of the abundances of the species
    in the CNO-cycle.

    Parameters
    ----------
    Ys : array of floats
        Abundances of the species over time
    equality_time : float
        The time at which 1H and 4He approximately
        had the same abundance
    savetimes : array of floats
        Timesteps at which the abundances were saved
    """    
    labels = ["H", "$^{12}$C", "$^{13}$N", "$^{13}$C", "$^{14}$N", "$^{15}$O", "$^{15}$N", "$^{4}$He"]
    colors = ['dodgerblue', 'dimgrey', 'yellowgreen', 'dimgrey', 'yellowgreen', 'tomato', 'yellowgreen', AEK ]
    lines = ['-', '-', '--', '-.', '-', '--', '-.', '-']
    fig = plt.figure(tight_layout=True)

    unit = year * 1e9
    for i,abundances in enumerate(Ys.T):
        plt.plot(savetimes / unit, 
                abundances, color = colors[i], ls = lines[i],
                label = labels[i], marker='', linewidth = 2.5)

    eq_idx =  argmin(abs( equality_time - savetimes/unit))
    plt.scatter(equality_time, Ys.T[-1][eq_idx]
                , marker = 'h', c = 'gold', ec = 'dodgerblue', 
                linewidth = 2, 
                s = 200, zorder = 4)

    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-17,2)
    plt.ylabel('Abundance', fontsize = 14)
    plt.xlabel('time [Gyr]', fontsize = 14)
    plt.savefig(f'sims/{simname}/cno_evol.pdf')
    plt.close(fig)

def t_eq_diagram(T9s, pp_eqs, cno_T9, cno_eqs, cno_T9s_bad, cno_eqs_bad):
    """Generates a plot showing the 1H-4He equality time for each
    temperature and for both the pp-chain and CNO-cycle.

    Parameters
    ----------
    T9s : array of floats
        The temperatures at which the simulation
        was performed in Giga-Kelvins.
    pp_eqs : array of floats
        The 1H-4He equality times for the pp-chain.
    cno_T9 : array of floats
        The temperatures in Giga-Kelvins for the 
        CNO-cycle simulations which converged.
    cno_eqs : array of floats
        The 1H-4He equality times for the 
        CNO-cycle simulations which converged.
    cno_T9s_bad : array of floats
        The temperatures in Giga-Kelvins for the 
        CNO-cycle simulations which did not converge.
    cno_eqs_bad : array of floats
        The 1H-4He equality times for the 
        CNO-cycle simulations which did not converge.
    """    
    fig, ax = plt.subplots(tight_layout=True)

    ax.plot(T9s * 1e3, pp_eqs, c = 'k', marker = 'h', ls=':')
    ax.plot(cno_T9, cno_eqs, c = AEK, marker = 'h', ls=':', 
            markeredgecolor = 'k')
    ax.plot(cno_T9s_bad, cno_eqs_bad, c = AEK, marker = '^', ls=':', 
            markeredgecolor = 'r')
    ax.plot( [cno_T9s_bad[-1], cno_T9[0]], [cno_eqs_bad[-1], cno_eqs[0]], 
            c = AEK, marker ='', ls =':')
    ax.axvline(18, color = 'maroon', ls = '--')
    ax.set_xscale('log')
    ax.set_xlabel('Temperature [MK]')
    ax.set_ylabel('$t_\\mathrm{eq}$ [Gyrs]')

    # Inset
    left, bottom, width, height = [0.55, 0.4, 0.35, 0.5]
    ax2 = fig.add_axes([left, bottom, width, height])
    start1 = 11
    start2 = 4 
    ax2.plot(T9s[start1:] * 1e3, pp_eqs[start1:], 
            c = 'k', marker = 'h', ls=':')
    ax2.plot(cno_T9[start2:], cno_eqs[start2:], 
            c = AEK, marker = 'h', ls=':', markeredgecolor='k')
    ax2.set_yscale('log')
    ax2.set_facecolor('snow')
    ax.indicate_inset_zoom(ax2, edgecolor="#3d3c3c")
    plt.savefig(f'sims/{simname}/t_eq_diagram.pdf')
    plt.close(fig)

def dominance_diagram(T9s, Zs, eq_pps, eq_cnos):
    """Creates a dominance diagram, showing which
    nuclear reaction network (pp-chain or CNO-cycle)
    dominates at which (temperature, metallicity)

    Parameters
    ----------
    T9s : array of floats
        The temperatures in Giga-Kelvins for
        which the simulations were run.
    Zs : array of floats
        The metallicities for
        which the simulations were run.
    eq_pps : array of floats
        The 1H-4He equality times of the pp-chain.
    eq_cnos : array of floats
        The 1H-4He equality times of the CNO-cycle.
    """    
    pp_dominance = (eq_pps <= eq_cnos)
    fig = plt.figure()

    custom_cmap = ListedColormap(['firebrick', plt.get_cmap('cividis')(255)])
    plt.pcolormesh(T9s*1e3, Zs, pp_dominance, cmap=custom_cmap, ec='k', lw=0.1)
    plt.ylabel('Metallicity $Z/Z_\\mathrm{ISM}$', fontsize = 14)
    plt.xlabel('Temperature [MK]', fontsize = 14)
    plt.savefig(f'sims/{simname}/dominance_diagram.pdf')
    plt.close(fig)