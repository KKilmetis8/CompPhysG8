#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:10:39 2024

@authors: konstantinos & diederick
"""

import numpy as np
import prelude as c

class Atom:
    def __init__(self, pos: np.ndarray, vel: np.ndarray,
                 color: str, markersize: float | int = 10):
        '''
        Each Argon Atom is an object of this class.
        
        Parameters
        ----------
        pos: array of length equal to c.dims, initial positions.

        vel: array of length equal to c.dims, initial velocities.

        color: string, color of the marker associated to this atom,
               used during plotting.

        markersize: float or int, size of the marker during plotting.
        '''


        # Tuples
        self.pos = np.array(pos)
        self.oldpos = self.pos # update at the same time
        self.vel = np.array(vel)

        # Energies
        self.kinetic = 0.5 * c.m_argon * np.linalg.norm(self.vel)**2
        self.potential = 0
        self.total = self.kinetic + self.potential
        
        # Other bodies that it cares about, also class atom. Added with
        # get_friends() method
        self.friends = []
        self.directions = []
        self.old_force = 0
        
        # Plotting
        self.color = color
        self.markersize = markersize
        
        if c.dims == 2:
            self.past_x = []
            self.past_y = []
        if c.dims == 3:
            self.past_x = []
            self.past_y = []
            self.past_z = []
            
    def first_step(self, particles: list | np.ndarray, me: int):
        '''
        Perform the first simulation step
        Needed for appropiate behaviour

        Parameters
        ----------
        particles: A list or numpy array of Atom objects that
                   contains all the particles in the simulation.
        
        me: Integer, index of this particle in the particles list.
        '''
        # Apply the minimum imaging convention
        self.get_friends(particles, me)

        self.old_force = self.force()
        
    def get_friends(self, particles: np.ndarray | list, me: int, step: str = 'leapfrog'):
        
        '''
        Returns the distances to the rest of the atoms,
        Appllying the minimum imaging convention
        
        Parameters
        ----------
        particles: A list or numpy array of Atom objects that
                   contains all the particles in the simulation.
        
        me: Integer, index of this particle in the particles list.
        
        step: String, which integration method to use. 
              "leapfrog" for Verlet, "euler" for Euler.

        '''
        rs = np.zeros(len(particles))
        unitaries = np.zeros( (len(particles), c.dims))
        for i, particle in enumerate(particles):
            if i != me:
                temp_r = 0            
                temp_unitaries = np.zeros(c.dims)
                # MIC
                for j in range(c.dims):
                    if step == 'euler':
                        diff = self.oldpos[j] - particle.oldpos[j]
                    elif step == 'leapfrog':
                        diff = self.pos[j] - particle.pos[j]
                    else:
                        raise ValueError('There is no ' + step + 'in Ba Sing Se')
                    diff -= c.boxL * np.round(diff * c.inv_boxL,0) # 0 in, 1 out
                    
                    temp_unitaries[j] = diff
                    temp_r += diff**2
                # Don't come too close
                if temp_r < 1:
                    temp_r =  1
                rs[i] = np.sqrt(temp_r) 
                unitary = temp_unitaries / rs[i]
                unitaries[i] = unitary    
            else:
                rs[i] = 0
                unitaries[i] = [np.NaN, np.NaN, np.NaN]
                
        # Remove yourself NOTE: May be slow
        self.friends = rs[ rs > 0]
        unitaries = unitaries[~np.isnan(unitaries)]
        self.directions = np.reshape(unitaries, (len(particles) - 1, c.dims))

    def lj_pot(self, r: float | np.ndarray) -> float | np.ndarray:
        ''' 
        Calculates the Lennard-Jones potential 
        
        Parameters
        ----------
        r: float or an array of floats, distances in Angstrom.

        Returns
        -------
        lennard_jones: float or an array of floats, calculated potential values.
        '''
        prefactor = 4 * c.epsilon
        pauli = c.sigma**12 / r**12
        vdWaals = c.sigma**6 / r**6
        lennard_jones = prefactor * ( pauli - vdWaals )
        return lennard_jones

    def lj_pot_prime(self, r: float | np.ndarray) -> float | np.ndarray:
        ''' 
        Calculates the dU/dr for the Lennard-Jones potential 
        
        Parameters
        ----------
        r: float or an array of floats, distances in Angstrom.

        Returns
        -------
        U_prime: float or an array of floats, calculated potential derivatives.
        '''
        prefactor = 4 * c.epsilon
        pauli = - 12 * c.sigma**12 / r**13
        vdWaals = - 6 * c.sigma**6 / r**7
        U_prime = prefactor * ( pauli - vdWaals )
        return U_prime
        
    def force(self) -> np.ndarray:
        ''' 
        Calculate the force of each particle, with the Lennard Jones 
        Potential.
        Friends must not be empty for this to work.

        Returns
        -------
        force: array of floats, size equal to c.dims.
               Net force from all other particles. 
        '''
        force = np.zeros(c.dims)
        for distance, direction in zip(self.friends, self.directions):
            force -= self.lj_pot_prime(distance) * direction # F=-\nabla U
        return force
    
    def am_i_in_the_box(self):
        ''' 
        Applies the Pacman condition. 
        If something moves outside the box, it appears on the other side.
        '''
        for d in range(c.dims):
            if self.pos[d] > c.boxL or self.pos[d] < 0:
                self.pos[d] -= c.boxL * np.floor(self.pos[d] * c.inv_boxL)
        
    def vel_verlet_update_pos(self):
        '''
        Energy perserving Verlet step.
        Applies one integration step for positions.
        '''
        timestep = c.h_sim_units #* c.t_tilde # time units!!
        self.pos += self.vel * timestep + self.old_force * timestep**2 * 0.5 * c.inv_m_argon
        self.am_i_in_the_box() 

    def vel_verlet_update_vel(self, particles: list | np.ndarray, me: int):
        '''
        Energy perserving Verlet step.
        Applies one integration step for velocities.
        
        Parameters
        ----------
        particles: A list or numpy array of Atom objects that
                   contains all the particles in the simulation.
        
        me: Integer, index of this particle in the particles list.
        '''

        # Update oldpos
        self.get_friends(particles, me)
        new_force = self.force()
        self.vel += 0.5 * c.h_sim_units * (self.old_force + new_force) * c.inv_m_argon
        self.old_force = new_force
        
    def energy_update(self):
        ''' 
        Updates the energy attributes to new values.
        '''
        self.kinetic = 0.5 * c.m_argon * np.linalg.norm(self.vel)**2
        potential = 0
        for dist in self.friends:
            potential += self.lj_pot(dist)
        self.potential = potential
        self.total = self.kinetic + potential

    def euler_update_vel(self, particles: list | np.ndarray):
        '''
        Naive Euler step.
        Applies one integration step for velocities.
        
        Parameters
        ----------
        particles: A list or numpy array that contains all the 
            particles in the simulation.
        '''
        self.get_friends(particles) # Apply the minimum imaging convention
        self.vel += self.force() * c.h_sim_units * c.inv_m_argon
        
    def euler_update_pos(self, particles: np.ndarray | list):
        '''
        Naive Euler step.
        Applies one integration step for positionsß.
        
        Parameters
        ----------
        particles: A list or numpy array that contains all the 
            particles in the simulation.
        '''
        self.pos += self.vel * c.h_sim_units
        self.am_i_in_the_box()
        self.energy_update(particles)
