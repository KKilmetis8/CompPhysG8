import numpy as np
import prelude as c
from Atom import Atom
from itertools import product

class Particles:
    def __init__(self, particles: np.ndarray | int, seed=c.rngseed):
        self.particles = particles
        
        self.rng = np.random.default_rng(seed=seed)
        
        # Random Initialization
        # if type(particles) not in [list,np.ndarray]:
        #     self.particles = []
        #     for i in range(int(particles)):
        #         pos = self.rng.random(c.dims)*c.boxL
        #         vel = self.rng.random(c.dims)*2 - 1 # -1 to 1
        #         temp = Atom(pos, vel, c.colors[i])
        #         self.particles.append(temp)
        
        # FCC Positions
        if type(particles) not in [list,np.ndarray]:
            self.particles = []
            init_poss = self.init_position_FCC()
            init_vels = self.init_velocities()
            for i in range(int(particles)):
                temp = Atom(init_poss[i], init_vels[i], 'purple')
                self.particles.append(temp)
                
        for particle in self.particles:
            particle.first_step(self.particles)

        self.all_positions  = [self.positions]
        self.all_velocities = [self.velocities]
        self.all_energies = [self.energies]
     
    def init_position_FCC(self):
        unit_fcc = np.array([[0,0,0],
                             [1,1,0],
                             [0,1,1],
                             [1,0,1]])
        
        iterations = np.array(list(product(range(3), repeat=3)))

        pos = unit_fcc.copy()
        for iteration in iterations[1:]:
            points = unit_fcc + iteration
            pos = np.vstack((pos, points))
            
        # Rescale to box
        small_box = 0.9 * c.boxL
        pos = np.multiply(pos, small_box / 3)
        return pos

    def init_velocities(self):
        # Maxwell
        velocity_std = np.sqrt(c.EPSILON / c.M_ARGON) * np.sqrt(c.temperature / c.m_argon)
        vels = np.random.normal(scale=velocity_std, size=(c.Nbodies, c.dims))
        return vels

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
            particle.vel = new_vels[i]
            
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
                
            for particle in self.particles:
                particle.vel_verlet_update_vel(self.particles)
                particle.energy_update()
        else:
            raise ValueError('There is no ' + step + 'in Ba Sing Se')
        # save positions and velocities to a big list containing past positions and velocities of all particles
        self.all_positions.append(self.positions)
        self.all_velocities.append(self.velocities)
        self.all_energies.append(self.energies)
        # self.all_energies.

    def rescale_vels(self):
        target = (len(self.particles) - 1) * (3/2) * c.temperature
        total_kinetic = np.sum(self.energies[0])
        rescale_lambda = np.sqrt(target/total_kinetic)
         