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

        # Other bodies that it cares about, also class atom. Added with
        # get_friends() method
        self.friends = []
        
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
        
    def get_friends(self, particles):
        
        '''
        Returns the distances to the rest of the atoms,
        Applling the minimum imaging convention
        
        '''
        rs = np.zeros(len(particles))
        for i, particle in enumerate(particles):
            temp_r = 0            
            for j in range(c.dims):
                # MIC
                diff = self.oldpos[j] - particle.oldpos[j]
                diff -= c.boxL * np.round(diff * c.inv_boxL,0) # 0 in, 1 out
                temp_r += diff**2
            
            rs[i] = np.sqrt(temp_r)
                
        # NOTE: May be slow
        self.friends = rs[ rs > 0]
                    
    def lj_pot_prime(self, r):
        ''' Calculates the dU/dr for the Lennard-Jones potential '''
        prefactor = 4 * c.epsilon
        pauli = - 12 * c.sigma**12 / r**13
        vdWaals = - 6 * c.sigma**6 / r**7
        U_prime = prefactor * ( pauli - vdWaals )
        return U_prime
        
    def accleration(self):
        ''' 
        Calculate the acceleration of each particle, with the Lennard Jones 
        Potential.
        Friends must not be empty for this to work 
        '''
        acc = np.zeros(c.dims)
        for r in self.friends:
            acc -= self.lj_pot_prime(r) * self.pos / r # F=-\nabla U
        acc *= c.inv_m_argon
        return acc
    
    def am_i_in_the_box(self):
        # NOTE: Maybe this can be phrased more succintly
        for d in range(c.dims):
            if self.pos[d] > c.boxL or self.pos[d] < 0:
                self.pos[d] -= c.boxL * np.floor(self.pos[d] * c.inv_boxL)
        self.oldpos = self.pos
                
    def euler_step(self, particles):
        ''' Naive Euler Step that does not conserve energy'''
        self.get_friends(particles) # Apply the minimum imaging convention
        # self.am_i_in_the_box()
        self.oldpos = self.pos
        self.pos += self.vel * c.h
        self.vel += self.accleration() * c.h
        self.am_i_in_the_box()


        
    
    # def leapfrog_step(self, h = half_day):
    #     if self.friends != None:
            
    #         # self.pos += 0.5 * h * self.vel # Drift 
    #         self.vel +=  h * self.gravity(self.pos) # Kick
    #         self.pos +=  h * self.vel # Drift 
    #         # self.vel +=  0.5 * h * self.gravity(self.pos) # Kick