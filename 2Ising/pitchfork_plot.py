import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex']     = False # False makes it fast, True makes it slow.
plt.rcParams['figure.dpi']      = 300
plt.rcParams['font.family']     = 'Times New Roman'
plt.rcParams['figure.figsize']  = [6 , 5]
plt.rcParams['axes.facecolor']  = 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

data_pos = np.loadtxt("2Ising/sims/75% positive_sweep_at_25-Apr-16:21:26/final_mags_and_temps.csv", delimiter=',')
data_neg = np.loadtxt("2Ising/sims/75% negative_sweep_at_25-Apr-17:07:55/final_mags_and_temps.csv", delimiter=',')

temps = data_pos[:,0]
mags_pos = data_pos[:,1]
mags_neg = data_neg[:,1]

plt.figure(figsize=(4,2.75), tight_layout=True)
plt.plot(temps, mags_pos, marker='h', c='k',       ls='--', label='75% Positive', zorder=3)
plt.plot(temps, mags_neg, marker='h', c='#F1C410', ls='--', label='75% Negative', zorder=3)
plt.axhline(0, c = 'maroon', ls ='--', zorder=2)
plt.axvline(2.269, c = 'maroon', ls ='--', zorder=2)

plt.xlabel("Temperature", fontsize=14), plt.ylabel("Mean Spin", fontsize=14)
plt.legend()
plt.ylim(-1.2, 1.2)
#plt.xlim(np.min(temps)*0.8, np.max(temps)*1.2)
plt.xticks(np.arange(1, 4+0.5, 0.5))

plt.show()