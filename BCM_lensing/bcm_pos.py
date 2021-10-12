from BCM_lensing.halo import Halo
from BCM_lensing.component import CG, EG, BG, RDM
import numpy as np
from BCM_lensing.utils import *
from scipy.interpolate import interp1d
import warnings; warnings.simplefilter('ignore')

def BCM_POS( group_df, halo_num, groupPos, subgroupPos, 
        basePath='./', constraint='BCM', resolution=50,  snapNum=135, 
        M1=86.3, MC=3.3e3, eta=.54, beta=.12):
    
    
    halo = Halo(halo_num, group_df, groupPos, subgroupPos, resolution=resolution, basePath=basePath)

    halo.run_density(mult=10)

    # Fit NFW parameters
    halo.nfw_fit()

    # Remember the order of r
    i = np.argsort(halo.subhalo_dm_r_over)
    r_sorted = halo.subhalo_dm_r_over[i]

    # Compute the true mass profile
    true_masses = (np.arange(len(r_sorted))+1) * halo.particle_mass

    # Generate the BCM components
    if constraint=='CG':
        cg = CG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        M_BCM = cg.Mass(r_sorted) + (1 - cg.f_CG()) * true_masses

    elif constraint=='BG':
        bg = BG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        M_BCM = bg.Mass(r_sorted) + (1 - bg.f_BG()) * true_masses

    elif constraint=='EG':
        true_masses[r_sorted>halo.r_200] = halo.m_200 * (1-eg.f_EG()) 
        eg = EG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        M_BCM = eg.Mass(r_sorted) + (1-eg.f_EG()) * true_masses

    elif constraint=='RDM':
        cg = CG(halo.r_200, halo.m_200, halo.c, halo.rho_s)
        bg = BG(halo.r_200, halo.m_200, halo.c, halo.rho_s)
        eg = EG(halo.r_200, halo.m_200, halo.c, halo.rho_s)
        rdm = RDM(cg, bg, eg, M1=M1, M_c=MC, eta=eta, beta=beta)

        # Compute the relaxation parameter xi for each r
        _ = rdm.run_xi(halo.ri, halo.masses)
        xi = rdm.xi_func(r_sorted)
        rdm.build_MassFunc(r_sorted, true_masses)
        
        M_BCM = rdm.Mass(r_sorted, true_masses, xi) + (1 - rdm.f_RDM()) * true_masses

    elif constraint=='BCM':
        cg = CG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        bg = BG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        eg = EG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        rdm = RDM(cg, bg, eg, M1=M1, M_c=MC, eta=eta, beta=beta)

        # Compute the relaxation parameter xi for each r
        _ = rdm.run_xi(halo.ri, halo.masses)
        xi = rdm.xi_func(r_sorted)
        rdm.build_MassFunc(r_sorted, true_masses)

        # Calculate the BCM mass profile
        M_BCM = cg.Mass(r_sorted)+ bg.Mass(r_sorted)+ rdm.Mass(r_sorted, true_masses, xi)+eg.Mass(r_sorted)
        M_BCM[r_sorted>halo.r_200] =M_BCM[r_sorted<halo.r_200][-1]
        M_BCM += eg.Mass(r_sorted)

    else:
        raise ValueError(f'Must use CG, BG, EG, RDM, or BCM as constraint not {constraint}')

    # Interpolate the BCM to each of the particle r's
    M_BCM_interp = interp1d(M_BCM, r_sorted, fill_value='extrapolate', assume_sorted=True)


    # Invert to find r_BCM
    r_BCM = M_BCM_interp(true_masses)

    # Compute the difference in the radii
    dr = r_BCM - r_sorted

    # Dont adjust the particles outside of r_200 or inside of 3epsilon
    dr[r_sorted > halo.r_200] = 0
    dr[r_sorted<halo.ri[0]] = 0

    # Return to original order
    res = np.empty(dr.shape)
    res[i] = dr

    # Apply corrections to full dm particle positions
    dm = halo.dm['Coordinates']
    delta = np.zeros(len(dm))
    delta[:len(res)] = res
    correction = ((halo.halo_dm_r + delta)/halo.halo_dm_r )[:, np.newaxis]

    # Convert back to cartesian
    new_cartesian =  correction * (dm - halo.sg_COM) + halo.sg_COM

    return new_cartesian, halo.dm['ParticleIDs']
