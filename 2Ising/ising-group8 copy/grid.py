import config
from importlib import reload


mag_fields = [-2, -1, 0, 1, 2]
init_grids = ['75% positive', '75% negative']

first = True
for H in mag_fields:
    config.H_ext = H
    for init_grid in init_grids:
        config.init_grid = init_grid
        config.simname = f'non-zero H/H = {H}/{init_grid}'
        if first:
            import simulation
            first = False
        else:
            reload(simulation)