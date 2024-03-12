#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:10:39 2024

@authors: konstantinos & diederick
"""

import numpy as np
import prelude as c

class Atom:
    def __init__(self, pos, vel,
                 color, markersize = 10):
        
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
            
    def first_step(self, particles):
        self.get_friends(particles) # Apply the minimum imaging convention
        self.old_force = self.force()
        
    def get_friends(self, particles, step = 'leapfrog'):
        
        '''
        Returns the distances to the rest of the atoms,
        Appllying the minimum imaging convention
        
        '''
        rs = np.zeros(len(particles))
        unitaries = np.zeros( (len(particles), c.dims))
        for i, particle in enumerate(particles):
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
            rs[i] = np.sqrt(temp_r) 
            unitary = temp_unitaries / rs[i]
            unitaries[i] = unitary    
            
        # Remove yourself NOTE: May be slow
        self.friends = rs[ rs > 0]
        unitaries = unitaries[~np.isnan(unitaries)]
        self.directions = np.reshape(unitaries, (len(particles) - 1, c.dims))
        
        
    def lj_pot(self, r):
        ''' Calculates the Lennard-Jones potential '''
        prefactor = 4 * c.epsilon
        pauli = c.sigma**12 / r**12
        vdWaals = c.sigma**6 / r**6
        lennard_jones = prefactor * ( pauli - vdWaals )
        return lennard_jones

    def lj_pot_prime(self, r):
        ''' Calculates the dU/dr for the Lennard-Jones potential '''
        prefactor = 4 * c.epsilon
        pauli = - 12 * c.sigma**12 / r**13
        vdWaals = - 6 * c.sigma**6 / r**7
        U_prime = prefactor * ( pauli - vdWaals )
        return U_prime
        
    def force(self):
        ''' 
        Calculate the force of each particle, with the Lennard Jones 
        Potential.
        Friends must not be empty for this to work 
        '''
        force = np.zeros(c.dims)
        for distance, direction in zip(self.friends, self.directions):
            force -= self.lj_pot_prime(distance) * direction # F=-\nabla U
        return force
    
    def am_i_in_the_box(self):
        # NOTE: Maybe this can be phrased more succintly
        for d in range(c.dims):
            if self.pos[d] > c.boxL or self.pos[d] < 0:
                self.pos[d] -= c.boxL * np.floor(self.pos[d] * c.inv_boxL)
        
    def vel_verlet_update_pos(self):
        '''Much better verlet step '''
        timestep = c.h_sim_units #* c.t_tilde # time units!!
        self.pos += self.vel * timestep + self.old_force * timestep**2 * 0.5 * c.inv_m_argon
        self.am_i_in_the_box() 

    def vel_verlet_update_vel(self, particles):
        timestep = c.h_sim_units # * c.t_tilde # time units!!
        # Update oldpos
        self.get_friends(particles)
        new_force = self.force()
        self.vel += 0.5 * timestep * (self.old_force + new_force) * c.inv_m_argon
        self.old_force = new_force
        
    def energy_update(self):
        ''' Naive Euler Step that does not conserve energy'''
        # NOTE: Unit problem.
        self.kinetic = 0.5 * c.m_argon * np.linalg.norm(self.vel)**2#  * \
                                          # c.vel_to_cgs)**2
        potential = 0
        for r in self.friends:
            potential += self.lj_pot(r)
        self.potential = potential
        self.total = self.kinetic + potential

    def euler_update_vel(self, particles):
        self.get_friends(particles) # Apply the minimum imaging convention
        self.vel += self.force() * c.h_sim_units * c.inv_m_argon #* c.t_tilde
        
    def euler_update_pos(self, particles):
        self.pos += self.vel * c.h_sim_units #* c.t_tilde # time units!!
        self.am_i_in_the_box()
        self.energy_update(particles)
