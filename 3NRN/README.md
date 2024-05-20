# Computational Physics: Nuclear-Reaction Network

**Authors**: Konstantinos Kilmetis and Diederick Vroom

This code was made to simulate nuclear processes involving Hydrogen-burning (H-burning) occurring in the cores of stars, resulting in the production of Helium (He). In particular, we focussed on simulating the proton-proton chain (pp-chain) and the Carbon-Nitrogen-Oxygen cycle (CNO-cycle). Our main resource for going about this was the 2017 paper by Lippuner and Roberts, titled "SkyNet: A Modular Nuclear Reaction Network Library". 

At its core, our code solves a series of coupled non-linear differential equations, which are set by the reactions occurring in the pp-chain and CNO-cycle. These equations encode the change in abundances for each element occurring in the reaction-network, over each reaction in the network. Using these equations, a vector function was created, the roots of which gives the new abundances of each element. The roots of this function were determined using the Newton-Raphson method for root finding.

This code was created for Project 3: Nuclear-Reaction Network of the MSc course "Computational Physics" at Leiden University, year 2024.

## How to Use

### Running a simulation

Running a simulation consists mainly of two steps:

* Setting the initial conditions in `config.py`
* Running `simulation.py`

#### Setting initial conditions

`config.py` is the file where all the initial conditions are set. In this file one can set the following base parameters:

For more details, see the documentation in `config.py`.

#### Running the simulation

### Outputs

#### Example Outputs

## Other Files

While the user only has to interact with `config.py`, the rest of the constituent parts of the code are summarized here.

1. `pp.py` contains functions used during simulations of the pp-chain: the vector-function to find the root for, its Jacobian, and the Newton-Raphson method.
2. `cno.py` contains functions used during simulations of the CNO-chain: the vector-function to find the root for, its Jacobian, and the Newton-Raphson method.
3. `ISM_abundances.py` contains the default values used for abundances for each element/isotope found in the pp-chain and CNO-cycle, as if they are in the Inter-Stellar Medium (ISM).
4. `NRN_rates.csv` contains a table giving the reaction-rates for the pp-chain and CNO-cycle, for a series of temperatures normalized to $10^9$ K. These rates and temperatures were taken from Angulo et al. (1999).
5. `density_interp.py` calculates the mean core density for given temperatures, which are used to scale the nuclear reaction-rates appropriately.