import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True # be FAST
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [5 , 5]
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

from os import listdir

AEK = '#F1C410'
lstyles = ['-', '--', ':']

Zs = np.sort([float(file[file.find('=')+1:file.find(')')]) for file in listdir('Metallicities') if 'y (Z=' in file])
#Zs = [int(Z) if int(Z) == Z else Z for Z in Zs ]

eq_times = [np.load(f'/data2/vroom/CompPhysG8/3NRN/Metallicities/eq_time (Z={Z}).npy')[0] for Z in Zs]

plt.figure(tight_layout=True)
plt.plot(Zs, eq_times, 'k:', marker='h')
plt.grid()
plt.xscale('log')
plt.yscale('log')
# plt.xlim(1e-4,1e-1)
plt.xlabel('Metallicity $Z/Z_\\mathrm{ISM}$', fontsize = 14)
plt.ylabel('$t_\\mathrm{eq}$ [Gyr]', fontsize = 14)
plt.show()
