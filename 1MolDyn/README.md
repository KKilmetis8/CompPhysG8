# Computational Physics: Molecular Dynamics

**Authors**: Konstantinos Kilmetis and Diederick Vroom

This code was made to simulate Argon atoms at different states of matter by changing initial densities and temperatures. The atoms are simulated to be in a pseudo-infinite box by implementing the Minimal Image Convention, which creates a periodic-boundary effect similar to the classic arcade game "Pacman", where an atom reapears on the opposite edge when it crosses the box boundary. Distances between the atoms are measured as effective distances, where they take the Minimal Image Convention into account.
Forces are calculated using the Lennard-Jones potential: $V(r)=4\varepsilon\left[\left(\frac{r}{\sigma}\right)^{12} - \left(\frac{r}{\sigma}\right)^{6} \right]$, where $\varepsilon$ and $\sigma$ are constants, dependent on the used element. For Argon $\varepsilon = 119.8\;\mathrm{K}$ and $\sigma = 3.405\;\mathrm{\AA}$. 

## How to Use

### Running different states of matter

### Running a custom configuration

Run

```bash
python3 'simulation.py'
```

Don't touch anything else

### Outputs
