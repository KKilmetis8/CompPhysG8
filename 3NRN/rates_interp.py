import numpy as np
import matplotlib.pyplot as plt

rates_table = np.loadtxt("NRN_Rates.csv", skiprows=1, delimiter=',')

T9s = rates_table[:,0]
rates_pp = rates_table[:,1:4]
rates_cno = rates_table[:,4:]

x = np.log10(T9s)
x_fit = np.linspace(x[0], x[-1], 20)

fig, axs = plt.subplots(1, 2, figsize=(6,3.5), tight_layout=True, sharey=True)
for i, reac_rates in enumerate([rates_pp, rates_cno]):
    ax = axs[i]
    #ax.grid()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Temperature [MK]')
    if i == 0: ax.set_ylabel('Reaction rate $\\left[\\mathrm{cm^{3}\\;s^{-1}\\;mol^{-1}}\\right]$')
    ax.set_title(['pp-chain', 'CNO-cycle'][i])
    for j, rates in enumerate(reac_rates.T):

        y = np.log10(rates)
        coeff = np.polyfit(x, y, 6)
        y_fit = np.poly1d(coeff)(x_fit)

        ax.plot(10**x*1e3, 10**y, ls='-', marker='', c=f'C{j}', label=f"Reaction {j+1}")
        ax.scatter(10**x_fit*1e3, 10**y_fit, c=f"C{j}")
    if i == 1: ax.legend(loc='upper left', bbox_to_anchor = (1.01, 1))

plt.show()