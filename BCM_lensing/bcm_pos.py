from BCM_lensing.halo import Halo
from BCM_lensing.component import CG, EG, BG, RDM
import numpy as np
from BCM_lensing.utils import *
from scipy.interpolate import interp1d
import warnings; warnings.simplefilter('ignore')

def BCM_POS( group_df, halo_num, groupPos, subgroupPos, 
        basePath='./', constraint='BCM', resolution=20,  snapNum=135, 
        M1=86.3, MC=3.3e3, eta=.54, beta=.12):
    
    
    halo = Halo(halo_num, group_df, groupPos, subgroupPos, resolution=resolution, basePath=basePath)

    halo.run_density(mult=10)

    # Fit NFW parameters
    halo.nfw_fit()

    # Create extra bins near truncation site
    br = halo.ri[halo.ri<halo.r_200]
    ar = halo.ri[halo.ri>halo.r_200]
    inbetween = np.logspace(np.log10(br[-2]), np.log10(halo.r_200+1), 10)
    halo.ri = np.hstack((br, inbetween, ar))

    # Update the halo masses at the new bin counts
    halo.masses =       np.sum(halo.subhalo_dm_r_over[:, np.newaxis]<halo.ri, axis=0) * halo.particle_mass


    # Generate the BCM components
    if constraint=='CG':
        cg = CG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        M_BCM = cg.Mass(halo.ri) + (1 - cg.f_CG()) * halo.masses

    elif constraint=='BG':
        bg = BG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        M_BCM = bg.Mass(halo.ri) + (1 - bg.f_BG()) * halo.masses

    elif constraint=='EG':
        eg = EG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        M_BCM = eg.Mass(halo.ri) + (1 - eg.f_EG()) * halo.masses

    elif constraint=='RDM':
        cg = CG(halo.r_200, halo.m_200, halo.c, halo.rho_s)
        bg = BG(halo.r_200, halo.m_200, halo.c, halo.rho_s)
        eg = EG(halo.r_200, halo.m_200, halo.c, halo.rho_s)
        rdm = RDM(cg, bg, eg, M1=M1, M_c=MC, eta=eta, beta=beta)
        
        # Compute the relaxation parameter xi for each r
        xi = rdm.run_xi(halo.ri, halo.masses)
        M_BCM = rdm.Mass(halo.ri, halo.masses, xi) + (1 - rdm.f_RDM()) * halo.masses

    elif constraint=='BCM':
        cg = CG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        bg = BG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        eg = EG(halo.r_200, halo.m_200, halo.c, halo.rho_s, M1=M1, M_c=MC, eta=eta, beta=beta)
        rdm = RDM(cg, bg, eg, M1=M1, M_c=MC, eta=eta, beta=beta)
        # Compute the relaxation parameter xi for each r
        xi = rdm.run_xi(halo.ri, halo.masses)

        # Calculate the BCM mass profile
        M_BCM = cg.Mass(halo.ri)+ bg.Mass(halo.ri)+ eg.Mass(halo.ri)+ rdm.Mass(halo.ri, halo.masses, xi)
    else:
        raise ValueError(f'Must use CG, BG, EG, RDM, or BCM as constraint not {constraint}')

    # Remember the order of r
    i = np.argsort(halo.subhalo_dm_r_over)
    r_sorted = halo.subhalo_dm_r_over[i]

    # Interpolate the BCM to each of the particle r's
    M_BCM_interp = interp1d(M_BCM, halo.ri, fill_value='extrapolate')

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


    dm = halo.dm['Coordinates']
    delta = np.zeros(len(dm))
    delta[:len(res)] = res
    correction = ((halo.halo_dm_r + delta)/halo.halo_dm_r )[:, np.newaxis]

    # Convert back to cartesian
    new_cartesian =  correction * (dm - halo.sg_COM) + halo.sg_COM
    return new_cartesian, halo.dm['ParticleIDs']
