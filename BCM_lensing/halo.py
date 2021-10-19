import illustris_python as il
import numpy as np
from BCM_lensing.utils import *

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
                 particle_mass=4.8e-2, resolution=20, basePath='../Illustris-3-Dark/output', 
                 snap_num=135, grav_softening=5.7):

        self.halo_num    = halo_num
        self.group_df    = group_df
        self.groupPos    = groupPos
        self.subgroupPos = subgroupPos
        self.rho_s = 0
        self.c     = 0
        self.r_200 = group_df.Group_R_Crit200[self.halo_num]
        self.m_200 = group_df.Group_M_Crit200[self.halo_num]

        self.particle_mass = particle_mass
        self.resolution = resolution
        self.basePath = basePath
        self.snap_num = snap_num
        self.grav_softening = grav_softening

        self.g_COM       = groupPos[halo_num]
        self.sub = group_df.GroupFirstSub[halo_num]
        self.sg_COM      = subgroupPos[self.sub]
        self.first_sub   = group_df.GroupFirstSub[halo_num]

        self.dm = il.snapshot.loadHalo(self.basePath,self.snap_num, self.halo_num,'dm')
        self.r_200 = self.group_df.Group_R_Crit200[self.halo_num]
        self.m_200 = self.group_df.Group_M_Crit200[self.halo_num]


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
        ri = np.logspace(np.log10(self.grav_softening*3), np.log10(self.r_200*mult), self.resolution)
        p_count = self.counter(ri)
        halo_density = den(ri[1:], np.diff(ri), p_count*self.particle_mass)
        return np.array(halo_density), ri, p_count


    def run_density(self, mult):
        """
        This function runs the density function and stores dark matter particles
        """
        self.halo_dm_r = make_r(self.dm['Coordinates'], self.g_COM)
        self.subhalo_dm, self.subhalo_dm_r_over = self._get_subhalos()

        self.subhalo_dm_r = self.subhalo_dm_r_over[self.subhalo_dm_r_over<self.r_200]
        self.m_200 = self.particle_mass * len(self.subhalo_dm_r)
        self.halo_density, self.ri, self.p_count = self._build_density(self.subhalo_dm_r, mult=mult)
        self.masses = np.sum(self.subhalo_dm_r[:, np.newaxis]<self.ri, axis=0) * self.particle_mass

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

    def build_rho_s(self, c):
        return c**3 * self.m_200 / (4*np.pi * self.r_200**3) * (np.log(1+c) - c/(1+c)) ** -1

    def nfw_density(self, r, c=1, rho_s=1e-3, clipped=False):
        """
        Compute the NFW predicted density for a given radius concentration factor
        and density factor. clipping the density past the r_200 mark is optional
        """
        rs = self.r_200/c

        try:
            x = r/rs
        except:
            x = np.einsum('i, j->ij', r, 1/rs)

        rho = rho_s *x**-1 * (1+x)**-2
        if clipped:
            return self.clip(rho, r)
        return rho

    def counter(self, ri):
        dri = np.diff(ri)
        p_count = []
        for i, rii in enumerate(ri[1:]):
            count = np.sum((ri[i]<self.subhalo_dm_r) & (self.subhalo_dm_r<=rii))
            p_count.append(count)
        p_count = np.array(p_count)
        return p_count

    def nfw_fit(self):
        """
        fit NFW profile to density profile
        """
        # Make sure that we only count up to r_200!
        ri = np.logspace(np.log10(self.grav_softening*3), np.log10(self.r_200), self.resolution)
        p_count = self.counter(ri)
        
        # Generate table of cs and corresponding rhos
        cs = np.logspace(.1, 1, 10)
        rho_s = self.build_rho_s(cs)

        # This should be ~1 for each c... If not we got problems!
        tbl = np.array([integrate_shells(ri ,(self.nfw_density, [c, rho])) for c, rho in zip(cs, rho_s)])/self.m_200
        assert np.isclose(len(cs), np.sum(tbl), rtol=1e-1)

        p_percent = p_count/len(self.subhalo_dm_r)
        error = tbl**2  / len(self.subhalo_dm_r)

        # Compute chi2 and find best parameters
        chi2 = np.sum((tbl - p_percent)**2/error, axis=-1)
        self.c = cs[np.argmin(chi2)]
        self.rho_s = rho_s[np.argmin(chi2)]
        self.error = error[np.argmin(chi2)]

    def params(self):
        return self.c, self.rho_s
