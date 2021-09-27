import illustris_python as il
import numpy as np
import pandas as pd
from nbodykit.cosmology import Planck15
from utils import *

class Halo:

    """
    The halo class interfaces with Illustris-3-Dark and supplies key halo/subhalo information.
    This class is meant to interact with a single halo which means obtaining the group pos
    and the subgroup positions as well as the key group characteristics in group_df must be
    obtained first. 

    The Halo class automaticly computes a density profile at spherical shells in logspace
    from 3 * grav_softening to 10 r_200 of the halo with spacing determined by the resolution
    kwd. This density can be fit with the nfw fitting routine outlined here. 
    """

    def __init__(self, halo_num, group_df, groupPos, subgroupPos, 
                 particle_mass=4.8e-2, resolution=50, basePath='./Illustris-3-Dark/output', 
                 snap_num=135, grav_softening=5.7):

        self.halo_num    = halo_num
        self.group_df    = group_df
        self.groupPos    = groupPos
        self.subgroupPos = subgroupPos
        self.rho_s = 0
        self.c     = 0

        self.particle_mass = particle_mass
        self.resolution = resolution
        self.basePath = basePath
        self.snap_num = snap_num
        self.grav_softening = grav_softening

        self.g_COM       = groupPos[halo_num]
        self.sub = group_df.GroupFirstSub[halo_num]
        self.sg_COM      = subgroupPos[self.sub]
        self.first_sub   = group_df.GroupFirstSub[halo_num]

        self.run_density()

    def _get_halos(self):

        """
        extract m_200 and r_200 from the data frame and return the dm particles
        """
        halo_dm = il.snapshot.loadHalo(self.basePath,self.snap_num, self.halo_num,'dm')
        self.r_200 = self.group_df.Group_R_Crit200[halo_num]
        self.m_200 = self.group_df.Group_M_Crit200[halo_num]
        return halo_dm

    def _get_subhalos(self):
        """
        extract the dark matter particles for a subhalo and convert the positions to
        radius
        """
        subhalo_dm = il.snapshot.loadSubhalo(self.basePath,self.snap_num,self.sub,'dm')
        subhalo_dm_r = make_r(subhalo_dm['Coordinates'], self.sg_COM)
        return subhalo_dm, subhalo_dm_r

    def _build_density(self, subhalo_dm_r, mult=10):
        """
        Build the density profile for the subhalo. This function computes the poison error, 
        the mass of the halo at each radius, and the density itself. 
        """
        ri = np.logspace(np.log10(self.grav_softening*3), np.log10(mult*self.r_200), self.resolution)
        halo_density = []
        errors = []
        N = []
        for rii in ri:
            particle_count = np.sum(subhalo_dm_r<rii)
            N.append(particle_count*self.particle_mass)
            d = density(rii, self.particle_mass * particle_count)
            halo_density.append(d)
            errors.append(d**2/particle_count)
        return np.array(halo_density), ri, np.array(errors), np.array(N)

    def run_density(self, mult):
        """
        This function runs the density function and stores dark matter particles
        """
        halo_dm = self._get_halos()
        halo_dm_r = make_r(halo_dm['Coordinates'], self.g_COM)


        self.subhalo_dm, subhalo_dm_r_over = self._get_subhalos()
        subhalo_dm_r = subhalo_dm_r_over[subhalo_dm_r_over<self.r_200]

        self.m_200 = self.particle_mass * len(subhalo_dm_r)

        self.halo_density, self.ri, self.errors, self.masses = self._build_density(subhalo_dm_r, mult=mult)

    def clip(self, rho, r):
        """
        Clip densities that are above r_200
        """
        try:
            rho[r>self.r_200] = 0
        except:
            if r>self.r_200:
                rho=0
        return rho

    def nfw_density(self, r, c=None, rho_s=None, clipped=False):
        """
        Compute the NFW predicted density for a given radius concentration factor
        and density factor. clipping the density past the r_200 mark is optional
        """
        if c== None:
            c = self.c
        if rho_s==None:
            rho_s = self.rho_s

        rs = self.r_200/c
        x = r/rs
        rho = rho_s *x**-1 * (1+x)**-2
        if clipped:
            return self.clip(rho, r)
        return rho

    def nfw_fit(self):
        """
        fit NFW profile to density profile
        """
        self.c,self.rho_s = minimize(self.minimize_func, (3, 1e-4), args=(self.halo_density, np.sqrt(self.errors), self.ri, self.r_200))['x']

    def minimize_func(self, ps, obs, sig, r, r_200):
        c, rho_s = ps
        model = self.nfw_density(r, c, rho_s)
        return chi2(model, obs, sig)

    def params(self):
        return self.c, self.rho_s
