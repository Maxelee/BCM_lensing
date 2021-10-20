import os
from BCM_lensing.component import CG, BG, EG, RDM
from BCM_lensing.bcm_pos import BCM_POS
from BCM_lensing.halo import Halo
from BCM_lensing.utils import *
import time
import numpy as np
import pandas as pd
from nbodykit.lab import *
from scipy.interpolate import interp1d
from absl import app, flags
from mpi4py import MPI
import illustris_python as il
from pmesh import ParticleMesh
import glob
import h5py

FLAGS = flags.FLAGS

flags.DEFINE_string('basePath', '/burg/astro/users/mel2260/Illustris-3-Dark/output', 'Path to Illustris Data')
flags.DEFINE_string('outPath', '/burg/astro/users/mel2260/BCM_results/', 'Path to save Data')
flags.DEFINE_string('constraint', 'BCM', 'constraint for identifiying outputs')
flags.DEFINE_integer('snapNum', 135, 'Snapshot corresponding to redshift')
flags.DEFINE_integer('num_halos', 100, 'number of halos to use') # We should turn this into a mass threshold
flags.DEFINE_float('M1', 86.3,   'M1 bcm parameter in 10^10 M_sun h^-1')
flags.DEFINE_float('MC', 3300.0, 'MC bcm parameter in 10^10 M_sun h^-1')
flags.DEFINE_float('eta', 0.54, 'eta bcm parameter')
flags.DEFINE_float('beta', 0.12, 'beta bcm parameter')
flags.DEFINE_boolean('power', True, 'compute power of correction or not')
flags.DEFINE_boolean('DMO', False, 'compute power of DMO')


def main(argv):
    del argv

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size= comm.Get_size()

    # Illustris Data
    subgroupFields = ['SubhaloPos']
    groupFields    = ['GroupFirstSub', 'GroupLen',  'Group_M_Crit200','Group_R_Crit200', 'GroupPos']

    subgroupPos = il.groupcat.loadSubhalos(FLAGS.basePath, FLAGS.snapNum, fields=subgroupFields)
    groups    = il.groupcat.loadHalos(   FLAGS.basePath, FLAGS.snapNum, fields=groupFields)

    groupPos = groups.pop('GroupPos')
    group_df = pd.DataFrame(groups)
    group_df['Group_M_Crit200']  = group_df['Group_M_Crit200']
    group_df['Group_R_Crit200']  = group_df['Group_R_Crit200']

    """ --------------------------------------------CORRECTION----------------------------"""
    bcm_pos = np.ones((1, 3))
    counter = 0
    perrank = FLAGS.num_halos // size
    comm.Barrier()
    bcm_start = time.time()
    # Do the BCM on each of halos, parallelized
    for halo_num in range(rank * perrank, (rank + 1) * perrank):
        h_coords= BCM_POS(group_df, halo_num,groupPos, subgroupPos, 
                FLAGS.basePath, constraint=FLAGS.constraint, 
                M1=FLAGS.M1, MC=FLAGS.MC, eta=FLAGS.eta, beta=FLAGS.beta)
        bcm_pos = np.concatenate((bcm_pos, h_coords))
        counter += len(h_coords)

    comm.Barrier()
    # Now replace the snapshots halos with BCM corrected halos
    bcm_pos  = bcm_pos[1:]
    bcm_pos = comm.gather(bcm_pos, root=0)
    counter = comm.reduce(counter, root=0, op=MPI.SUM)
    
    if rank ==0:
        print(f'\nBCM finished in {time.time()-bcm_start} seconds')
        bcm_pos = np.vstack(bcm_pos)

        # First pull up all of the hdf5 files into a list
        files = sorted(glob.glob(FLAGS.basePath+'_BCM/snapdir_135/*.hdf5'))

        for f_name in files:
            # Open a file and check that the length is shorter than the length of BCM
            f = h5py.File(f_name, 'r+')
            d = f['PartType1']['Coordinates']
            l_d = len(d)
            if counter > l_d:
                start = len(bcm_pos) - counter
                end   = start + l_d
                d[...] = bcm_pos[start:end]
                counter -=l_d
            else:
                start = len(bcm_pos) - counter
                d = d[:counter]
                d[...] = bcm_pos[start:]
                f.close()
                break
            f.close()

    comm.Barrier()
    if FLAGS.power:
        if rank == 0:
            dat = il.snapshot.loadSubset(FLAGS.basePath + '_BCM', FLAGS.snapNum, 'dm',  'Coordinates')
        else:
            dat = None

        dat = comm.bcast(dat, root=0)
        # Put in box coordinates (Mpc/h)
        dat /= 1000
        dat %= 75
        dat[np.isnan(dat)] =0

        # Put into a particle mesh object for mapping and power spectrum
        pm = ParticleMesh(Nmesh=[455]*3, BoxSize=[75, 75, 75], resampler='cic')
        mass1 =4.8e-2# 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(dat), op=MPI.SUM) * 4.8e-2

        IllustrisMap = pm.create(type="real")
        IllustrisMap.paint(dat, mass=mass1, layout=None, hold=False)
        del mass1, dat


        address  = '%s/snapdir_%i/BCM_coordinates_M1-%2.1f_MC-%3.1f_eta-%1.3f_beta-%1.3f/'%(FLAGS.outPath, FLAGS.snapNum, FLAGS.M1, FLAGS.MC, FLAGS.eta, FLAGS.beta)
        if not os.path.exists(address):
            os.makedirs(address)
        address_BCM = address + 'power/' + FLAGS.constraint + 'map_Nmesh' + str(pm.Nmesh[0])

        # Dont need to save map rightnow, so save memory and dont output
        #FieldMesh(IllustrisMap).save(address_BCM)

        P = FFTPower(IllustrisMap, mode='1d').save(address_BCM +'_p.json')

    """-------------------------------------------DMO-----------------------------"""
    if FLAGS.DMO:

        address_DMO = address + 'power/' +'DMOmap_Nmesh' + str(pm.Nmesh[0])

        try:
            IllustrisMap_DMO = BigFileMesh(address_DMO, dataset='Field').to_real_field()
        except:
            if rank == 0:
                dat = il.snapshot.loadSubset(FLAGS.basePath, FLAGS.snapNum, 'dm',  'Coordinates')
                dat /= 1000
                dat %= 75
            else:
                dat = None

            dat = comm.bcast(dat, root=0)
            mass1 = 4.8e-2#1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(dat), op=MPI.SUM) * 4.8e-2

            IllustrisMap_DMO = pm.create(type="real")
            IllustrisMap_DMO.paint(dat, mass=mass1, layout=None, hold=False)
            del dat, mass1,

            #FieldMesh(IllustrisMap_DMO).save(address_DMO)
            P_DMO = FFTPower(IllustrisMap_DMO, mode='1d').save(address_DMO +'_p.json')

        # Cross correlation coefficient incase this is a useful statistic
        r_cc = FFTPower(IllustrisMap, second=IllustrisMap_DMO, mode='1d').save(address_BCM +'_rcc.json')


if __name__ == "__main__":
    app.run(main)
