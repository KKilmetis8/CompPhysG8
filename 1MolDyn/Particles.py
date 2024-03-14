import numpy as np
import prelude as c
from Atom import Atom
from itertools import product

class Particles:
#### --------------------------------------------------------------------------
# Initialization methods
#### --------------------------------------------------------------------------   
    def __init__(self, particles: np.ndarray | int, mode = 'FCC', seed=c.rngseed):
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
       
    def init_position_FCC(self):
        unit_fcc = np.array([[0,0,0],
                             [1,1,0],
                             [0,1,1],
                             [1,0,1]]) * 0.5 # Kon: ensures no overlap
        
        iterations = np.array(list(product(range(3), repeat=3)))

        pos = unit_fcc.copy()
        for iteration in iterations[1:]:
            points = unit_fcc + iteration
            pos = np.vstack((pos, points))
            
        # Rescale to box
        small_box = 0.9 * c.boxL
        pos = np.multiply(pos, small_box / pos.max()) # using pos.max() ensures right normalization
        return pos

    def init_velocities(self):
        # Maxwell
        # velocity_std = np.sqrt(c.EPSILON / c.M_ARGON) * np.sqrt(c.temperature / c.m_argon)
        # vels = np.random.normal(scale=velocity_std, size=(c.Nbodies, c.dims))
        vels = np.random.normal(0,c.temperature,
                                size = (c.Nbodies, c.dims))
        return vels
#### --------------------------------------------------------------------------
# Main simulation loop
#### --------------------------------------------------------------------------       
    def update(self, step = 'leapfrog'):
        """
        Updates the positions/velocities for each particle, 
        taking the forces from the other particles into account.
        """
        if step == 'euler':
            for particle in self.particles:
                particle.euler_update_pos(self.particles)
            for particle in self.particles:
                particle.euler_update_vel(self.particles)

                
        elif step == 'leapfrog':
            # Any way to write this in a more elegant way?
            for particle in self.particles:
                particle.vel_verlet_update_pos()
                
            for me, particle in enumerate(self.particles):
                particle.vel_verlet_update_vel(self.particles, me)
                particle.energy_update()
                #print(np.sum(self.energies[0]))
        else:
            raise ValueError('There is no ' + step + 'in Ba Sing Se')
        # save positions and velocities to a big list containing past positions and velocities of all particles
        self.all_positions.append(self.positions)
        self.all_velocities.append(self.velocities)
        self.all_energies.append(self.energies)
        # self.all_energies.
#### --------------------------------------------------------------------------
# Equilibriation methods
#### --------------------------------------------------------------------------      
    def relax_run(self, trelax: float):
        '''
        Parameters
        ----------
        trelax : float
            Relaxation time of the particle set.

        Returns
        -------
            Runs the simulation for 1 relaxation time
        '''
        ## Maybe instead of that let's do run until change in kinetic energy
        ## is small?
        # window = 20
        # roll = 5
        # tol = 0.05
        # # Run for a little while
        # for i in range(window):
        #     self.update(step = 'leapfrog')

        # max_steps = 500 # Don't get stuck
        # rolling_kinetic_energy = np.sum(self.all_energies[-window:][0]) / window
        # for i in range(max_steps):
        #     self.update(step = 'leapfrog')
            
        #     # Every roll steps, check for convergance in kin
        #     if not i % roll:
        #         kinetic = np.sum(self.all_energies[-roll:][0]) / roll
        #         if np.abs(kinetic - rolling_kinetic_energy) < tol:
        #             break
        #         else:
        #             rolling_kinetic_energy = np.sum(self.all_energies[-window:][0]) / window
        #             print('Eq Try:', i // roll)
        #             print(rolling_kinetic_energy, kinetic)
        #             print(len(self.all_energies[0]))
        for i in range(50):
            self.update(step = 'leapfrog')

                    
            
    def rescale_vels(self, target):
        total_kinetic = np.sum(self.energies[0])
        print(target, total_kinetic)
        rescale_lambda = np.sqrt(target/total_kinetic)
        print('lamda', rescale_lambda)
        rescaled_vels = np.multiply(rescale_lambda, self.velocities)
        self.velocities = rescaled_vels
        print(target, 0.5 * np.sum(self.velocities)**2)
        
    def equilibriate(self, tolerance = 0.1):
        
        our_kinetic = np.sum(self.energies[0])
        target = (len(self.particles) - 1) * (3/2) * c.temperature
        print(our_kinetic, target)
        while np.abs(our_kinetic - target)/target > tolerance: 
            # Calc Relax time
            # mean_vel = np.abs(np.mean(self.velocities))
            # number_density = c.density * c.inv_m_argon
            # cross_section = np.pi * (2*c.R_ARGON / c.SIGMA)**2 #πd^2
            # trelax = 1 /  (cross_section * mean_vel * np.sqrt(2) * number_density)
            
            # Run for one tenth of the relax time
            self.relax_run(1)
            
            # Check for equilibriation
            our_kinetic = np.sum(self.energies[0])
            if np.abs(our_kinetic - target)/target < tolerance:
                print('System in Equilibrium')
                print(our_kinetic, target)
                break
            else:
                print('Not in Equilibrium')
                #print(our_kinetic, target)
                self.rescale_vels(target)
#### --------------------------------------------------------------------------
# Observables
#### --------------------------------------------------------------------------    
    def pressure(self):
        # Old version of getting friends,
        # only used latest positions.
        # Friends doesn't include self (j=i)
        # distance_pairs = np.zeros((c.Nbodies, c.Nbodies-1))
        # for i, particle in enumerate(self.particles):
        #     particle.get_friends(self.particles, i)
        #     distance_pairs[i] = particle.friends
        
        # Big 3D matrix, which will contain the pairwise distances
        # for each particle for each timestep.
        friends_matrices = np.zeros((len(self.all_positions), c.Nbodies, c.Nbodies-1))
        # Loop over all positions for each timestep.
        for t, pos_at_tstep in enumerate(self.all_positions):
            # Pick reference particle.
            for i in range(c.Nbodies):
                # Reference particle coordinates.
                ref_particle = pos_at_tstep[:,i].reshape((c.dims,1))
                # Calculate distance to all particles from reference particle (includes itself)
                friends = np.sqrt(np.sum((pos_at_tstep - ref_particle)**2, axis=0))
                # Insert friends into 3D matrix, remove self (0).
                friends_matrices[t,i] = friends[np.arange(len(friends)) != i]

        # Calculate LJ-pot-primes for each pairwise distance.
        # Not removing zeros previously would give many warnings here.
        lj_pot_primes = self.particles[0].lj_pot_prime(friends_matrices)

        # Only care about j>i pairs, otherwise counting twice (j<i)
        # Hence use lower triangular part (tril)
        sum_part  = np.tril(friends_matrices * lj_pot_primes, -1).sum(axis=(1,2)).mean()
        sum_part *= c.density/(6 * c.Nbodies)

        # ideal gas
        ig_part = c.temperature * c.density

        pressure = ig_part - sum_part
        
        return pressure
    
    def pair_correlation(self, bins = c.Nbodies // 3):
        n_corr = np.zeros(bins)
        for particle in self.particles:
            # Ensure same r range is used
            temp_n_corr, r = np.histogram(particle.friends, bins = bins,
                                          range=(0.3, c.boxL / 2))
            n_corr = np.add(n_corr, temp_n_corr)
        n_corr = np.divide(n_corr, c.Nbodies) # Average
        deltar = r[1] - r[0]
        coeff  = 2 * c.boxL**3 / (4 * np.pi * c.Nbodies * (c.Nbodies - 1))
        g_corr = coeff * n_corr / (r[1:]**2 * deltar) 
        return r[1:], g_corr
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



 
         