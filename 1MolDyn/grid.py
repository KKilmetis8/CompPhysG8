import config
import prelude
from importlib import reload
import simulation


Ts   = [0.5, 0.75, 1, 1.25, 1.5]
rhos = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

for T in Ts:
    for rho in rhos:
        config.temperature = T
        config.density = rho
        reload(prelude)
        print(prelude.density, prelude.temperature)
        reload(simulation)