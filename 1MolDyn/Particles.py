import numpy as np
import prelude as c
import config
from Atom import Atom
from itertools import product

class Particles:
#### --------------------------------------------------------------------------
# Initialization methods
#### --------------------------------------------------------------------------   
    def __init__(self, particles: np.ndarray | int, mode = 'FCC', 
                 seed: int =c.rngseed):
        ''' Initializes a collection of particles.
            
            Parameters
            ----------
            particles: An array of Atom objects that contains all the 
                       particles in the simulation.
                       Alternatively, particles can be an int, then 
                       class will initialize a list of particles of 
                       that size.
            
            mode: string, accepts FCC and random. Chooses how to 
                  initialize positions.

            seed: Which seed to use for radomization. If set to None,
                  changes seed during each initialization.
            '''
        self.particles = particles
        self.rng = np.random.default_rng(seed=seed)
        
        if type(particles) not in [list,np.ndarray]:
            # Random Initialization
            if mode == 'random':
                    self.particles = []
                    for i in range(int(particles)):
                        pos = self.rng.random(c.dims)*c.boxL
                        vel = self.rng.random(c.dims)*2 - 1 # -1 to 1
                        temp = Atom(pos, vel, c.colors[i])
                        self.particles.append(temp)
        
            # FCC Positions
            elif mode == 'FCC':
                self.particles = []
                init_poss = self.init_position_FCC()
                init_vels = self.init_velocities()
                for i in range(int(particles)):
                    temp = Atom(init_poss[i], init_vels[i], 'purple')
                    self.particles.append(temp)
                    
        # First step for Verlet
        for me, particle in enumerate(self.particles):
            particle.first_step(self.particles, me)

        self.all_positions  = [self.positions]
        self.all_velocities = [self.velocities]
        self.all_energies = [self.energies]
        self.n_hists = []
        self.bin_edges = None
        self.pressure_sum_parts = []
        
    def init_position_FCC(self) -> np.ndarray:
        ''' 
        Initializes the positions of the particles via
        Face-Centred Cubic configuration.

        Returns
        -------
        pos: array of shape (108, 3) with positions of the
             FCC-lattice.
        '''
        unit_fcc = np.array([[0,0,0],
                             [1,1,0],
                             [0,1,1],
                             [1,0,1]]) * 0.5 # Ensures no overlap
        
        iterations = np.array(list(product(range(3), repeat=3)))

        pos = unit_fcc.copy()
        for iteration in iterations[1:]:
            points = unit_fcc + iteration
            pos = np.vstack((pos, points))
            
        small_box = 0.9 * c.boxL # Rescale to box, 0.9 to stay inside
        pos = np.multiply(pos, small_box / pos.max()) # pos.max() ensures
                                                      # correct normalization
        return pos

    def init_velocities(self) -> np.ndarray:
        '''
        Generates the initial velocities via a Maxwell 
        distribution with standard deviation equal to
        temperatures.

        Returns
        ------
        vels: Array of floats of shape (c.Nbodies, c.dims).
        '''
        vels = np.random.normal(0, c.temperature, size = (c.Nbodies, c.dims))
        return vels
#### --------------------------------------------------------------------------
# Main simulation loop
#### --------------------------------------------------------------------------       
    def update(self, step: str = 'leapfrog'):
        """
        Updates the positions/velocities for each particle, 
        taking the forces from the other particles into account.

        Parameters
        ----------
        step: String, which integration method to use. 
              "leapfrog" for Verlet, "euler" for Euler.
        """
        if step == 'euler':
            for particle in self.particles:
                particle.euler_update_pos(self.particles)
            for particle in self.particles:
                particle.euler_update_vel(self.particles)
                
        elif step == 'leapfrog':
            for particle in self.particles:
                particle.vel_verlet_update_pos()
                
            for me, particle in enumerate(self.particles):
                particle.vel_verlet_update_vel(self.particles, me)
                particle.energy_update()
        else:
            raise ValueError('There is no ' + step + 'in Ba Sing Se')
            
        # Save positions and velocities
        self.all_positions.append(self.positions)
        self.all_velocities.append(self.velocities)
        self.all_energies.append(self.energies)
#### --------------------------------------------------------------------------
# Equilibriation methods
#### --------------------------------------------------------------------------      
    def relax_run(self, nsteps: int = 10):
        '''
        Runs the simulation for a certain number of steps.

        Parameters
        ----------
        nsteps: integer, number of steps to run the simulation for.
        '''
        for i in range(nsteps):
            self.update(step = 'leapfrog')

    def rescale_vels(self, target: float):
        ''' 
        Rescales velocities to better match the expected Maxwellian 
        distribution.
        
        Parameters
        ----------
        target: The target kinetic energy.
        '''
        total_kinetic = np.sum(self.energies[0])
        rescale_lambda = np.sqrt(target/total_kinetic)
        rescaled_vels = np.multiply(rescale_lambda, self.velocities)
        self.velocities = rescaled_vels
        
        # Printing
        if config.loud:
            print(total_kinetic,"|", target)
            print('lamda', rescale_lambda)
            print(0.5 * np.sum(self.velocities)**2,"|", target)
        
    def equilibriate(self, tolerance: float = 0.05):
        '''
        Equilibrate the simulation so it better matches reality.

        Parameters
        ----------
        tolerance: float, the tolerance on the kinetic energy,
                   with respect to the target kinetic energy,
                   in which the simulation is in equilibrium.

        '''
        our_kinetic = np.sum(self.energies[0])
        target = (len(self.particles) - 1) * (3/2) * c.temperature
        print('Our Kinetic energy | Target')
        print('---------------------------')
        print(our_kinetic,"|", target)
        while np.abs(our_kinetic - target)/target > tolerance: 

            # Run for 10 steps
            self.relax_run(10)
            
            # Check for equilibriation
            our_kinetic = np.sum(self.energies[0])
            if np.abs(our_kinetic - target)/target < tolerance:
                print('System in Equilibrium')
                print(our_kinetic,'|', target)
                break
            else:
                print('Not in Equilibrium')
                #print(our_kinetic, target)
                self.rescale_vels(target)
#### --------------------------------------------------------------------------
# Observables
#### --------------------------------------------------------------------------    
    def pressure_sum_part(self):
        '''
        Calculate the sum part of the pressure equation for a single snapshot.
        '''
        # np.nans to avoid errors in lj_pot_primes calculation.
        friends_matrix = np.nan * np.ones((c.Nbodies, c.Nbodies-1))
        for i, particle in enumerate(self.particles):
            friends_matrix[i,i:] = particle.friends[i:]

        lj_pot_primes = self.particles[0].lj_pot_prime(friends_matrix)
        sum_part = np.nansum(friends_matrix * lj_pot_primes)
        self.pressure_sum_parts.append(sum_part)

    def pressure(self) -> tuple[float, float]:
        '''
        Calculate the pressure, averaging all pressure_sum_parts
        and including coefficients.

        Also calculates the standard deviation

        Returns
        -------
        out: (pressure, stdev), calculated pressure and associated 
              standard deviation.
        '''

        # Ideal gas
        ig_part = c.temperature * c.density
        averaged = np.mean(self.pressure_sum_parts)
        pressure = ig_part - c.density * averaged/(6 * c.Nbodies )
        
        # sigma_y = |dy/dx * sigma_x|
        # -> sigma_P = |dP/d<sum_part> * sigma_<sum_part>|
        # dP/d<sum_part> = - rho/(6 * N)
        pressure_stdev = c.density/(6 * c.Nbodies) * np.std(self.pressure_sum_parts)

        return pressure, pressure_stdev
    
    def n_pair_correlation(self, deltar: float = 0.01):
        ''' 
        Calculate n(r) for a given snapshot 
        
        Parameters
        ----------
        deltar: float, the bin size.
        '''
        # Ensure same r range is used
        if self.bin_edges is None:
            self.bin_edges = np.arange(deltar, np.sqrt(3) * c.boxL / 2, deltar) 
        n_corr = np.zeros(len(self.bin_edges[1:]))
        for i, particle in enumerate(self.particles):
            temp_n_corr, _ = np.histogram(particle.friends[i:], # dont overcount
                                          bins = self.bin_edges,)
            n_corr = np.add(n_corr, temp_n_corr)
        self.n_hists.append(n_corr)
    
    def g_pair_correlation(self) -> tuple[np.ndarray, np.ndarray]:
        ''' 
        Average all n(r), include coefficients for g(r).
        
        Returns
        -------
        out: (bin_r, g_corr), the radii of the bins and the associated
             correlation function value.
        '''
        # Average
        n_corr = np.zeros(len(self.n_hists[0]))
        for hist in self.n_hists:
            n_corr = np.add(n_corr, hist)
        n_corr = np.divide(n_corr, len(self.n_hists))
        
        # Ensure same r range is used
        bin_r = self.bin_edges[1:] 
        deltar = bin_r[1] - bin_r[0]
        coeff  = 2 * c.boxL**3 / (4 * np.pi * c.Nbodies * (c.Nbodies - 1))
        g_corr = coeff * n_corr / (bin_r**2 * deltar) 
        return bin_r, g_corr
#### --------------------------------------------------------------------------
# Quality of Life
#### --------------------------------------------------------------------------
    @property
    def positions(self) -> np.ndarray:
        """
        Get positions of all particles.
        Returns an (c.dims, number of particles) array,
        with each row corresponding to velocity-coordinates.
        """
        positions = np.zeros((c.dims, len(self.particles)))
        for i, particle in enumerate(self.particles):
            positions[:,i] = particle.pos
        return positions

    @positions.setter
    def positions(self, new_pos : np.ndarray):
        """
        Change positions of all particles.
        Requires an (c.dims, number of particles) array as input,
        with each row corresponding to position-coordinates.
        """
        for i, particle in enumerate(self.particles):
            particle.pos = new_pos[i]

    @property
    def velocities(self) -> np.ndarray:
        """
        Get velocities of all particles.
        Returns an (c.dims, number of particles) array,
        with each row corresponding to velocity-coordinates.
        """
        velocities = np.zeros((c.dims, len(self.particles)))
        for i, particle in enumerate(self.particles):
            velocities[:,i] = particle.vel
        return velocities
    
    @velocities.setter
    def velocities(self, new_vels : np.ndarray):
        """
        Change velocities of all particles.
        Requires an (c.dims, number of particles) array as input,
        with each row corresponding to velocity-coordinates.
        """
        for i, particle in enumerate(self.particles):
            particle.vel = new_vels[:,i]
            
    @property
    def energies(self) -> np.ndarray:
        """
        Get energies of all particles.
        Returns an (3, number of particles) array,
        with kinetic on row 0, potential on row 1, and the sum on row 2
        """
        energies = np.zeros((3, len(self.particles)))
        for i, particle in enumerate(self.particles):
            energies[:,i] = [particle.kinetic, particle.potential, particle.total]
        return energies
    
    @energies.setter
    def energies(self, new_ene : np.ndarray):
        """
        No such thing as an energy setter
        """
        raise ValueError('We never set energies explicitly; you vile, Noether-defying creature')
            
    @property
    def colors(self) -> list:
        """
        Get colors of all particles.
        Returns a list of length equal to the number of particles,
        with a color for each particle.
        """
        colors = []
        for particle in self.particles:
            colors.append(particle.color)
        return colors
    
    @colors.setter
    def colors(self, new_colors: list):
        """
        Change colors of all particles.
        Requires a list of length equal to the number of particles,
        with a new color for each particle.
        """
        for i, particle in enumerate(self.particles):
            particle.color(new_colors[i])



 
         