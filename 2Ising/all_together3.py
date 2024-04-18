import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from plotters import grid as plot_grid
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
N = 20
MC_step = N**2
steps = 10_000*MC_step
J = 1
H = 0
chunk_size = 20
x_array = np.arange(chunk_size*MC_step)
Ts = np.arange(1, 5, step = 0.1)
mags = []
convergence = []

# ----- 75% Positive ----- #
#flip = rng.integers(N, size = (2, N**2//4))
# init_grid = np.ones((N, N))
# init_grid[flip[0], flip[1]] = -1

# ----- Random ----- #
init_grid = np.sign(rng.random((N, N)) - 0.5)

init_energy = Hamiltonian(init_grid)


ms = []
buffer = 100
last_grids = PdfPages('last_grids.pdf')
for T in tqdm(Ts):
    e, m, c, eq_grid = metropolis(init_grid, steps, T, init_energy, J, H,
                                  rtol=1e-3, atol=1e-4, buffer=buffer)
    
    fig = plt.figure()
    plt.imshow(eq_grid, origin='lower', cmap='coolwarm')
    last_grids.savefig(fig)
    plt.close(fig)

    mags.append( m[-buffer*N**2:].mean() )
    convergence.append(c)
    ms.append(m[-buffer*N**2:])
last_grids.close()

#plt.plot(convergence)
# mags2 = []
# init_grid = np.ones((N, N))*(-1)
# init_grid[flip[0], flip[1]] = 1
# init_energy = Hamiltonian(init_grid)
# print(init_grid.mean())
# for T in tqdm(Ts):
#     e, m = metropolis(init_grid, steps, T, init_energy, H)
#     mags2.append( m[-50*MC_step:].mean() )
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
def chi(step, arr):
    t_diff = len(arr) - step
    term1  = (1/t_diff)*(arr[:t_diff] * arr[step:]).sum()
    term2  = (1/t_diff**2) * (arr[:t_diff].sum()*arr[step:].sum())
    return term1 - term2


chi_figs = PdfPages('chi_figs.pdf')
for i,m in enumerate(ms):
    ts = np.arange(len(m))
    chis = np.array([chi(t, np.abs(m)) for t in ts])
    up_to = np.where(chis <= 0)[0]
    
    if len(up_to) != 0:
        up_to = up_to[0]
    else:
        up_to = -1

    a, b = np.polyfit(ts[:up_to], np.log(chis[:up_to]), deg=1)
    tau  = -1/a
    chi0 = np.exp(b)

    avg_rel_error = (np.abs(chis[:up_to] - chi0*np.exp(-ts[:up_to]/tau))/chis[:up_to]).mean()
    print(f"({i+1:{len(str(len(ms)))}d}/{int(len(ms))}): tau = {tau:6.3f}, avg_rel_error = {avg_rel_error:6.3f}", end= '\r')

    ts_fit = np.linspace(0, ts[-1], 10000)
    fig = plt.figure()
    plt.plot(ts/MC_step, chis)
    plt.plot(ts_fit/MC_step, chi0*np.exp(-ts_fit/tau))
    plt.axhline(y=0, c='k', ls='--')
    plt.axvline(x=up_to/MC_step, c='k', ls='--')

    plt.xlabel('Monte Carlo steps per lattice site $\\left(\\mathrm{steps}/N^2 \\right)$')
    plt.ylabel('$\\chi$')

    plt.title(f'T = {Ts[i]:.2f}: $\\tau = {tau:.2f},\\;\\varepsilon = {avg_rel_error:.2f}$')
    chi_figs.savefig(fig)
    plt.close(fig)

chi_figs.close()

# %%
