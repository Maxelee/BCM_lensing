import sys
from BCM_lensing.halo import Halo
from BCM_lensing.component import CG, EG, BG, RDM
import numpy as np
from BCM_lensing.utils import *
from scipy.interpolate import interp1d
import warnings; warnings.simplefilter('ignore')
from absl import app, flags
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_string("subhalo_dm_path",'./Data/subhalo_dm', "Path to the subhalo_dm pickle file")
flags.DEFINE_string("subhalo_dm_r_path", './Data/subhalo_dm_r', "Path to the subhalo_dm_r pickle files")


def main(argv):
    del argv  # Unused.

    infile = open(FLAGS.subhalo_dm_path, 'rb')
    subhalo_dm   = pickle.load(infile)
    infile.close()
    infile = open(FLAGS.subhalo_dm_r_path, 'rb')
    subhalo_dm_r   = pickle.load(infile)
    infile.close()

    particle_mass = 4.8e-2

    # Build a halo object
    halo = Halo(subhalo_dm, subhalo_dm_r, resolution=20)

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
    # Convert particle positions in a subgroup to spherical coordinates
    spherical = get_spherical(halo.subhalo_dm['Coordinates'], halo.sg_COM)
    r = spherical[0, :]

    # Remember the order of r
    i = np.argsort(r)
    r_sorted = r[i]

    # Interpolate the BCM to each of the particle r's
    M_BCM_interp = interp1d(halo.ri, M_BCM, fill_value='extrapolate')
    M_BCM_total = M_BCM_interp(r_sorted)

    # Compute the true mass profile
    true_masses = (np.arange(len(r_sorted))+1) * particle_mass

    # build_r interpolateion to get r(M)
    r_interp = interp1d(true_masses, r_sorted, fill_value='extrapolate')

    # Plug M_BCM into r(M) to get r_BCM
    r_BCM = r_interp(M_BCM_total)

    # Compute the difference in the radii
    dr = r_sorted- r_BCM
    dr[r_sorted>halo.r_200] = 0
    dr[r_sorted<halo.ri[0]] = 0

    # Return to original order
    res = np.empty(dr.shape)
    res[i] = dr

    # Apply correction
    spherical_new = spherical.copy()
    spherical_new[0, :] = spherical[0, :] + res


    # Convert back to cartesian
    new_cartesian = make_cartesian(spherical_new, halo.sg_COM)


if __name__ == "__main__":
    app.run(main)

