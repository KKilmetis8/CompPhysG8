# Computational Physics: The Ising Model

**Authors**: Konstantinos Kilmetis and Diederick Vroom

This code was made as a Monte-Carlo Markov-Chain implementation of the 2D Ising Model for different initial temperatures, grid sizes, spin-coupling constant $J$, and external magnetic field $H$. The 2D grid of spins is simulated to be of pseudo-infinite size by periodic boundary conditions.

Throughout the simulation, the grid is guided towards a state of minimum energy, as per the Hamiltonian $\mathcal{H} = -J\sum_{\langle i,j\rangle} s_is_j - H\sum_is_i$, via the Metropolis-Hastings algorithm.

This code was created for Project 2: The Ising Model of the MSc course "Computational Physics" at Leiden University, year 2024.

## How to Use

### Running a simulation

Running a simulation consists mainly of two steps:

* Setting the initial conditions in `prelude.py`
* Running `simulation.py`

#### Setting initial conditions

`prelude.py` is the file where all the initial conditions are set. In this file one can set the temperature at which the simulation is run (`temperature`), the grid-size of the simulation (`Nsize`), the time period (in picoseconds) over which the simulation is run (  `time `), the size of the timestep (`timestep`) (in picoseconds), wether to print progress in the terminal (`loud`), and wether to plot the current configuration during the simulation.

#### Running the simulation


### Ouputs


#### Example Outputs


## Other Files

While the user only has to interact with `config.py`, the rest of the constituent parts of the code are summarized here.
