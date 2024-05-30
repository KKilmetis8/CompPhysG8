# Computational Physics: Nuclear-Reaction Network

**Authors**: Konstantinos Kilmetis and Diederick Vroom

This code was made to simulate nuclear processes involving Hydrogen-burning (H-burning) occurring in the cores of stars, resulting in the production of Helium (He). In particular, we focussed on simulating the proton-proton chain (pp-chain) and the Carbon-Nitrogen-Oxygen cycle (CNO-cycle). Our main resource for going about this was the 2017 paper by Lippuner and Roberts, titled "SkyNet: A Modular Nuclear Reaction Network Library". 

At its core, our code solves a series of coupled non-linear differential equations, which are set by the reactions occurring in the pp-chain and CNO-cycle. These equations encode the change in abundances for each element occurring in the reaction-network, over each reaction in the network. These equations were solved using the implicit Euler method for integration and the Newton-Raphson method for root finding.

This code was created for Project 3: Nuclear-Reaction Network of the MSc course "Computational Physics" at Leiden University, year 2024.

## How to Use

### Running a simulation

Running a simulation consists mainly of two steps:

* Setting the initial conditions in `config.py`
* Running `simulation.py`

#### Setting initial conditions

`config.py` is the file where all the initial conditions are set. In this file one can set the following base parameters:

* `temperature`: The core temperature of the star at which the simulation is run.

* `metallicity`: The metallicity with respect to the InterStellar Medium (ISM) value at which the simulation is run.

* `cycle`: Which nuclear reaction network to use, either the pp-chain or CNO-cycle.

* `init_abunds`: The initial abundances to use in the simulation. Can be preset ISM abundances, ISM abundances with a different metallicity, or custom abundances.

* `max_time`: The maximum number of steps the simulation is run for.

* `max_step`: The maximum size of the timestep.

`config.py` also contains the following Quality of Life parameters:

* `loud`: Wether to print progress in the terminal or not.

* `kind`: What kind of simulation you're running; for a single temperature and metallicity, a range of temperatures and constant metallicity, or a range of temperatures and metallicities.

* `simname`: The name of the simulation, used for saving results. Defaulted to `/{kind}_at_{time_of_initialization}/`.

Lastly, `config.py` contains the `temperatures` and `metallicities` (plural) parameters. These are used when running simulations for ranges of temperatures and/or metallicities. Make sure to set `kind = 'equality time'` or `kind = 'dominance'` when desired to do so.

For more details, see the documentation in `config.py`.

#### Running the simulation

Running the simulation is fairly straightforward. After having set the parameters within config.py simply run simulation.py to run the simulation.

```bash
python3 'simulation.py'
```

If `loud = True`, progressbars from the `tqdm` package will appear in the terminal at time-intensive steps of the simulation; namely when simulating over ranges of temperatures and/or metallicities. (`kind = 'equality time'` or `kind = 'dominance'`).

### Outputs

Depending on the kind (`kind = 'single'`, `'equality time'`, or `'dominance'`) of simulation run, different outputs will be produced.

#### Outputs for `kind = 'single'`

For a simulation run for a single temperature and metallicity, 2 outputs will be produced:

* `pp_evol.csv` or `cno_evol.csv`: Contains an array storing the times at which the abundances were saved in the first column, in seconds, and the abundances themselves in the other columns. Number of columns is dependent on which reaction network was run; 4+1 columns for the pp-chain, 8+1 for the CNO-cycle. 

* `pp_evol.pdf` or `cno_evol.pdf`:

#### Outputs for `kind = 'equality time'`

#### Outputs for `kind = 'dominance'`

#### Example Outputs

## Other Files

While the user only has to interact with `config.py`, the rest of the constituent parts of the code are summarized here.

1. `pp.py` contains functions used during simulations of the pp-chain: the vector-function to find the root for, its Jacobian, and the Newton-Raphson method.
2. `cno.py` contains functions used during simulations of the CNO-chain: the vector-function to find the root for, its Jacobian, and the Newton-Raphson method.
3. `ISM_abundances.py` contains the default values used for abundances for each element/isotope found in the pp-chain and CNO-cycle, as if they are in the Inter-Stellar Medium (ISM).
4. `NRN_rates.csv` contains a table giving the reaction-rates for the pp-chain and CNO-cycle, for a series of temperatures normalized to $10^9$ K. These rates and temperatures were taken from Angulo et al. (1999).
5. `density_interp.py` calculates the mean core density for given temperatures, which are used to scale the nuclear reaction-rates appropriately.