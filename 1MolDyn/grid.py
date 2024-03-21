import config
import prelude
from importlib import reload


configs = [(2.0, 0.5), (1.5, 0.5), (0.25, 1), (0.25, 1.125), (2.5, 1)]

first = True
for T, rho in configs:
    config.temperature = T
    config.density = rho
    reload(prelude)
    print(prelude.density, prelude.temperature)
    if first:
        import simulation
        first = False
    else:
        reload(simulation)