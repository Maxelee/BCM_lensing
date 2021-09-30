from BCM_lensing.halo import Halo
from BCM_lensing.component import CG, EG, BG, RDM
import numpy as np
from BCM_lensing.utils import *
from scipy.interpolate import interp1d
import warnings; warnings.simplefilter('ignore')

def BCM_pos(halo_num, group_df, groupPos, subgroupPos, resolution=20):
    halo = Halo(halo_num, group_df, groupPos, subgroupPos, resolution=resolution)

    halo.run_density(mult=10)

    # Fit NFW parameters
    halo.nfw_fit()

    # Generate the BCM components
    cg = CG(halo.r_200, halo.m_200, halo.c, halo.rho_s)
    bg = BG(halo.r_200, halo.m_200, halo.c, halo.rho_s)
    eg = EG(halo.r_200, halo.m_200, halo.c, halo.rho_s)
    rdm = RDM(cg, bg, eg)

    # Compute the relaxation parameter xi for each r
    xi = rdm.run_xi(halo.ri, halo.masses)

    # Calculate the BCM mass profile
    M_BCM = cg.Mass(halo.ri)\
          + bg.Mass(halo.ri)\
          + eg.Mass(halo.ri)\
          + rdm.Mass(halo.ri, halo.masses, xi)
    
    ############################### RADIAL CORRECTIONS ###############################
    # Remember the order of r
    i = np.argsort(halo.subhalo_dm_r_over)
    r_sorted = halo.subhalo_dm_r_over[i]

    # Interpolate the BCM to each of the particle r's
    M_BCM_interp = interp1d(halo.ri, M_BCM, fill_value='extrapolate')

    # Compute the true mass profile
    true_masses = (np.arange(len(r_sorted))+1) * halo.particle_mass

    # Invert to find r_BCM
    r_BCM = M_BCM_interp(true_masses)

    # Compute the difference in the radii
    dr = r_BCM - r_sorted
    dr[r_sorted>halo.r_200] = 0
    dr[r_sorted<halo.ri[0]] = 0

    # Return to original order
    res = np.empty(dr.shape)
    res[i] = dr


    # Convert back to cartesian
    new_cartesian = ((halo.subhalo_dm_r_over + res)/halo.subhalo_dm_r_over )[:, np.newaxis] * halo.subhalo_dm['Coordinates']

    return new_cartesian
