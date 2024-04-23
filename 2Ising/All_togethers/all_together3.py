import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba
from matplotlib.backends.backend_pdf import PdfPages
#%%
kernel = 0.5 * np.array([[0, 1, 0], 
                         [1, 0, 1], 
                         [0, 1, 0]])

rng  = np.random.default_rng(None)

def Hamiltonian(grid: np.ndarray[float]) -> float:
    Conv = convolve2d(grid, kernel, boundary='wrap', mode='same')
    coupling_term      = -J * (grid*Conv).sum()
    ext_mag_field_term = -H * grid.sum()
    return coupling_term + ext_mag_field_term

@numba.njit( nopython=True, nogil=True)
def metropolis(init_grid, steps, T, init_energy, J, H, rtol=1e-3, atol=1e-4, chunk_size = 20, buffer=20):
    beta = 1/T
    N = len(init_grid)
    MC_step = N**2
    trial = chunk_size*MC_step
    
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
        if i > i_eq + buffer*MC_step: 
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
                eq_grid = last_grid.copy()
                # if rel_diff < rtol: print('rel-eq:', rel_diff, rtol)
                # if abs_diff < atol: print('abs-eq:', abs_diff, atol)
                
            
        # Generate single-flip grid ------------------------------------------>
        flip_row = np.random.randint(0,N)
        flip_col = np.random.randint(0,N)
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
    
    if not converged: eq_grid = last_grid
    return all_energies[:i], all_avg_mags[:i], converged, eq_grid


#%%
N = 30
MC_step = N**2
steps = 10_000*MC_step
J = 1
H = 0
chunk_size = 20
x_array = np.arange(chunk_size*MC_step)
Ts = np.arange(1, 4, 0.2)
# Ts = np.array([1, 1.5, 2.1, 2.2, 2.3, 2.4, 3, 3.5])
mags = []
convergence = []

# ----- 75% Positive ----- #
flip = rng.integers(N, size = (2, N**2//4))
init_grid = np.ones((N, N))
init_grid[flip[0], flip[1]] = -1

# ----- Random ----- #
# init_grid = np.sign(rng.random((N, N)) - 0.5)

init_energy = Hamiltonian(init_grid)

ms = []
es = []
buffer = 200
# last_grids = PdfPages('last_grids.pdf')
for T in tqdm(Ts):
    e, m, c, eq_grid = metropolis(init_grid, steps, T, init_energy, J, H,
                                  rtol=1e-3, atol=1e-4, buffer=buffer)
    
    # fig = plt.figure()
    # plt.imshow(eq_grid, origin='lower', cmap='coolwarm')
    # last_grids.savefig(fig)
    # plt.close(fig)

    mags.append( m[-buffer*N**2:].mean() )
    convergence.append(c)
    ms.append(m[-buffer*N**2:])
    es.append(e[-buffer*N**2:])
# last_grids.close()

#%%
# black = converged, red = did not converge
colors = ["k" if converged else "r" for converged in convergence]

# plt.plot(Ts, mags, 
#          c = 'k', marker = 'h', ls = '-', markersize = 10, zorder=3)

plt.scatter(Ts, mags, 
         c = colors, marker = 'h', s = 100, zorder=3)
# plt.plot(Ts, mags2, c = '#F1C410', 
#           marker = 'h', ls = ':', markeredgecolor = 'k', markersize = 10)
plt.axhline(0, c = 'maroon', ls ='--', zorder=2)
plt.axvline(2.269, c = 'maroon', ls ='--', zorder=2)
plt.xlabel('T', fontsize = 14)
plt.ylabel('Avg mag', fontsize = 14)
plt.title(f'2D Ising model for N:{N}', fontsize = 15)
plt.ylim(-1.2,1.2)
plt.xlim(Ts.min()*0.8, Ts.max()*1.2)

#%%
@numba.njit(parallel=True)
def chi(steps, arr):
    chis = np.zeros(len(steps))
    for i in numba.prange(len(steps)):
        t_diff  = len(arr) - steps[i]
        term1   = (1/t_diff)*(arr[:t_diff] * arr[steps[i]:]).sum()
        term2   = (1/t_diff**2) * (arr[:t_diff].sum()*arr[steps[i]:].sum())
        chis[i] = term1 - term2
    return chis

from scipy.optimize import curve_fit
def exponential(t, x0, tau):
    return x0 * np.exp(-t/tau)

# chi_figs = PdfPages('chi_figs.pdf')
taus = []
for i,m in tqdm(enumerate(ms)):
    ts = np.arange(len(m))
    chis = chi(ts, m)
    up_to = np.where(chis <= 1e-7)[0] # needs to be dynamic
    
    if len(up_to) != 0:
        up_to = up_to[0]
    else:
        up_to = -1

    popt, _ = curve_fit(exponential,ts[:up_to], chis[:up_to])
    tau = popt[-1]
    taus.append(tau)
    continue
    chi0 = popt[0]
    # avg_rel_error = (np.abs(chis[:up_to] - chi0*np.exp(-ts[:up_to]/tau))/chis[:up_to]).mean()
    # print(f"({i+1:{len(str(len(ms)))}d}/{int(len(ms))}): tau = {tau:6.3f}, avg_rel_error = {avg_rel_error:6.3f}", end= '\r')
    
    ts_fit = np.linspace(0, ts[-1], 10000)
    fig = plt.figure()
    plt.plot(ts/MC_step, chis)
    plt.plot(ts_fit/MC_step, chi0*np.exp(-ts_fit/tau))
    plt.axhline(y=0, c='k', ls='--')
    plt.axvline(x=up_to/MC_step, c='k', ls='--')

    # plt.xlabel('Monte Carlo steps per lattice site $\\left(\\mathrm{steps}/N^2 \\right)$')
    # plt.ylabel('$\\chi$')

    # plt.title(f'T = {Ts[i]:.2f}: $\\tau = {tau:.2f},\\;\\varepsilon = {avg_rel_error:.2f}$')
   # chi_figs.savefig(fig)

    plt.close(fig)
# chi_figs.close()
#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Ts, mags, ls = ':', marker = '.', c='k')
ax.set_ylabel('Avg Mags')
ax.set_ylim(-1.1, 1.1)
ax2 = ax.twinx()
ax2.plot(Ts, np.array(taus)/50**2, ls = '-', marker = '.', c='maroon')
ax2.set_ylabel('Auto-corr time')
plt.xlabel('Temperature')


#%% Observables

fig, axs = plt.subplots(2,2, figsize = (8,6), tight_layout = True)

# Mean absolute spin
obs1 = np.zeros(len(ms))
sigma1 = np.zeros(len(ms))
for i in range(len(ms)):
    m = np.array(ms[i])
    obs1[i] = np.abs(m.mean())
    var1 = np.abs( np.mean(m**2) - np.mean(m)**2)
    sigma1[i] = np.sqrt(2*taus[i]/len(m) * var1)
axs[0,0].errorbar(Ts, obs1, yerr = sigma1, 
             ls=':', marker='h', c='k', capsize = 4)
axs[0,0].set_xlabel('Temperature', fontsize = 14)
axs[0,0].set_ylabel('Mean absolute spin', fontsize = 14)

# Energy per spin
obs2 = np.zeros(len(ms))
sigma2 = np.zeros(len(ms))
for i in range(len(es)):
    e = np.array(es[i]) / N**2  
    obs2[i] = e.mean()
    var2 = np.abs( np.mean(e**2) - np.mean(e)**2)
    sigma2[i] = np.sqrt(2*taus[i]/len(e) * var2)
axs[0,1].errorbar(Ts, obs2, yerr = sigma2, 
             ls=':', marker='h', c='k', capsize = 4)
axs[0,1].set_xlabel('Temperature', fontsize = 14)
axs[0,1].set_ylabel('Energy per spin', fontsize = 14)

# Magnetic susceptibility
obs3 = np.zeros(len(ms))
sigma3 = np.zeros(len(ms))
for i in range(len(ms)): # loops over runs
    m = np.array(ms[i])
    chunk_size = int(16*taus[i])
    chunk_chi_ms = 0
    total_chunks = 0
    prefactor = N**2/Ts[i] 
    for j in range(chunk_size, len(m), chunk_size): # loops over tau chunks
        chunk_m = m[j-chunk_size:j]
        chunk_chi_ms += np.mean(chunk_m**2) - np.mean(chunk_m)**2
        total_chunks += 1
    sigma3[i] = prefactor * chunk_chi_ms / total_chunks
    obs3[i] = prefactor * (np.mean(m**2) - np.mean(m)**2)

axs[1,0].errorbar(Ts, obs3, yerr = sigma3, 
             ls=':', marker='h', c='k', capsize = 4)
axs[1,0].set_xlabel('Temperature', fontsize = 14)
axs[1,0].set_ylabel('Magnetic susceptibility', fontsize = 14)

# Specific heat per spin susceptibility
obs4 = np.zeros(len(ms))
sigma4 = np.zeros(len(ms))
for i in range(len(es)): # loops over runs
    e = np.array(es[i])
    chunk_size = int(16*taus[i])
    chunk_cs = 0
    total_chunks = 0
    prefactor = 1/Ts[i]**2 * 1/N**2
    for j in range(chunk_size, len(e), chunk_size): # loops over tau chunks
        chunk_c = e[j-chunk_size:j]
        chunk_cs += np.mean(chunk_c**2) - np.mean(chunk_c)**2
        total_chunks += 1
    sigma4[i] = prefactor * chunk_cs / total_chunks
    obs4[i] = prefactor * (np.mean(e**2) - np.mean(e)**2)

axs[1,1].errorbar(Ts, obs4, yerr = sigma4, 
             ls=':', marker='h', c='k', capsize = 4)
axs[1,1].set_xlabel('Temperature', fontsize = 14)
axs[1,1].set_ylabel('Specific heat per spin', fontsize = 14)
# %%
