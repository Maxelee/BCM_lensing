import os
import pickle
from BCM_lensing.component import CG, BG, EG, RDM
from BCM_lensing.bcm_pos import BCM_POS
from BCM_lensing.halo import Halo
from BCM_lensing.utils import *
import time
import numpy as np
import pandas as pd
from nbodykit.lab import *
from scipy.interpolate import interp1d
from mpi4py import MPI
import illustris_python as il
from pmesh import ParticleMesh
import glob
import h5py
import warnings
from read import loadSubset
import sys
warnings.filterwarnings("ignore")
import re
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('basePath', '/home1/08434/tg877334/scratch/', 'Path to Illustris Data')
flags.DEFINE_string('sim_name', 'L205n2500TNG_DM', 'sim name')
flags.DEFINE_string('outPath', '/home1/08434/tg877334/work/BCM_results/', 'Path to save Data')
flags.DEFINE_string('constraint', 'BCM', 'constraint for identifiying outputs')
flags.DEFINE_integer('snapNum', 99, 'Snapshot corresponding to redshift')
flags.DEFINE_integer('batches', 2000, 'Snapshot corresponding to redshift')
flags.DEFINE_float('Mass_thresh', 2e12, 'number of halos to use') # We should turn this into a mass threshold
flags.DEFINE_float('M1', 2.2,   'M1 bcm parameter in 10^10 M_sun h^-1')
flags.DEFINE_float('MC', .23e4, 'MC bcm parameter in 10^10 M_sun h^-1')
flags.DEFINE_float('eta', 0.14, 'eta bcm parameter')
flags.DEFINE_float('beta', 4.09, 'beta bcm parameter')
flags.DEFINE_boolean('power', False, 'compute power of correction or not')
flags.DEFINE_boolean('DMO', False, 'compute power of DMO')
flags.DEFINE_boolean('overwrite', True, 'reperform BCM with same setup')

# Read in the position files
def read_file(filename, start, stop, Np):
    with open(filename, 'rb') as f:
        f.seek(4 * start, os.SEEK_SET)
        particle = np.fromfile(f, dtype=np.float32, count=stop-start)
    return particle
def sort_nicely( l ): 
  """ Sort the given list in the way that humans expect. 
  """ 
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  l.sort( key=alphanum_key ) 


def main(argv):
    del argv
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size= comm.Get_size()
    params = FLAGS.flag_values_dict()

    params['outPath']  = '%s%s/snapdir_%i/BCM_coordinates_M1-%2.1f_MC-%3.1f_eta-%1.3f_beta-%1.3f/'%(FLAGS.outPath, FLAGS.sim_name, FLAGS.snapNum, FLAGS.M1, FLAGS.MC, FLAGS.eta, FLAGS.beta)
    address = params['outPath']
    params['BoxSize'], params['Nmesh'] = [int(i) for i in re.findall(r'[a-zA-Z](\d+)', params['sim_name'])]
    
    if rank==0:
        if not os.path.isdir(address):
            os.makedirs(address)
            print(f'address generated at {address}\n')
        else:
            if not FLAGS.overwrite:
                raise ValueError('Run with this exact setup already exists')
        outfile = open(address + 'INIT_FILE', 'wb')
        pickle.dump(params, outfile)
        outfile.close()
    if rank ==0:
        print(f'param file writen at {address}INIT_FILE')

    bcm_start = time.time()
    full_path = params['basePath'] + params['sim_name'] + '/output'

    filename_x = '%s_BCM/snapdir_%s/BCMpos_x.txt'%(full_path, str(params['snapNum']).zfill(3))
    filename_y = '%s_BCM/snapdir_%s/BCMpos_y.txt'%(full_path, str(params['snapNum']).zfill(3))
    filename_z = '%s_BCM/snapdir_%s/BCMpos_z.txt'%(full_path, str(params['snapNum']).zfill(3))


    with h5py.File(il.snapshot.snapPath(full_path, params['snapNum']), 'r') as f:
        header = dict(f['Header'].attrs.items())
        M = header['MassTable']
        redshift = header['Redshift']
        particle_mass = M[1]

    # Illustris Data
    subgroupFields = ['SubhaloPos']
    groupFields    = ['GroupFirstSub', 'GroupLen',  'Group_M_Crit200','Group_R_Crit200', 'GroupPos', 'GroupMass']

    subgroupPos = il.groupcat.loadSubhalos(full_path, params['snapNum'], fields=subgroupFields)
    groups    = il.groupcat.loadHalos(  full_path, params['snapNum'], fields=groupFields)

    groupPos = groups.pop('GroupPos')
    group_df = pd.DataFrame(groups)

    num_halos = int(np.sum(group_df['GroupMass'] > params['Mass_thresh']/1e10))
    batches = params['batches']

    n_per_batch, extras = divmod(num_halos, batches) # Split the calculations into batch number of n_per_batch halos


    files = sorted(glob.glob('%s_BCM/snapdir_%s/*.hdf5'%(full_path, str(params['snapNum']).zfill(3))))
    sort_nicely(files)
    len_groups = []
    for chunk in range(len(files)):
        f = h5py.File(il.groupcat.gcPath(full_path, params['snapNum'], chunk), 'r')
        len_groups.append(f['Header'].attrs['Ngroups_ThisFile'])
        f.close()
    len_groups = np.concatenate(np.array([[gi] if gi<1000 else np.repeat(int(gi/comm.size/4), comm.size*4) for gi in len_groups]))
    if rank ==0:
        print(len_groups)
    """ --------------------------------------------CORRECTION----------------------------"""
    num_computed=0
    for n_groups in len_groups:
        
        bcm_pos = np.ones((1, 3))
        counter = 0
        if n_groups < num_halos - num_computed:
            perrank, res = divmod(n_groups, comm.size)
        else:
            perrank, res = divmod(num_halos - num_computed, comm.size)
        if rank ==0:
            print(f'Perrank = {perrank}')
        # Do the BCM on each of halos, parallelized
        for halo_num in range(rank * perrank + num_computed, (rank + 1) * perrank + num_computed):
            print(f'computing BCM for {halo_num}/{num_halos}')
            h_coords= BCM_POS(group_df, halo_num,groupPos, subgroupPos, 
                    full_path, constraint=params['constraint'], 
                    M1=params['M1'], MC=params['MC'], eta=params['eta'], beta=params['beta'], 
                    particle_mass=particle_mass, z=redshift, snapNum=params['snapNum'])
            bcm_pos = np.concatenate((bcm_pos, h_coords))
            counter += len(h_coords)
        num_computed += comm.size * perrank

        # Account for unequal splitting of halos into ranks
        if rank < res:
            perrank = 1
            for halo_num in range(rank * perrank + num_computed, (rank + 1) * perrank+num_computed):
                print(f'computing BCM for {halo_num}/{num_halos}')
                h_coords= BCM_POS(group_df, halo_num,groupPos, subgroupPos, 
                        full_path, constraint=params['constraint'], 
                        M1=params['M1'], MC=params['MC'], eta=params['eta'], beta=params['beta'], 
                        particle_mass=particle_mass, z=redshift, snapNum=params['snapNum'])
                bcm_pos = np.concatenate((bcm_pos, h_coords))
                counter += len(h_coords)
        num_computed += res
        # Now replace the snapshots halos with BCM corrected halos
        if len(bcm_pos) == 1:
            bcm_pos = np.array([]).astype(np.float64)
        else:
            bcm_pos  = bcm_pos[1:].astype(np.float64)

        comm.Barrier()
        out_arr = comm.allgather(len(bcm_pos))
        sendcounts = np.array(out_arr).astype('int') * 3
        displacements = np.insert(np.cumsum(sendcounts),0,0)[0:-1].astype(int)
        if rank ==0:
            recvbuf = np.empty((int(sum(sendcounts)/3), 3 )).astype(np.float64)
        else:
            recvbuf=None
        comm.Barrier()
        comm.Gatherv(bcm_pos, [recvbuf, sendcounts, displacements, MPI.DOUBLE], root=0)
        counter = comm.reduce(counter, root=0, op=MPI.SUM)
        del bcm_pos
        if rank==0:
            bcm_pos = recvbuf
            del recvbuf
            bcm_pos[np.isnan(bcm_pos)] = 0
            x, y, z = np.array(bcm_pos).T
            del bcm_pos
            # Check if position file exists, if not write to
            if os.path.exists(filename_x):
                append_write= 'a'
            else:
                append_write= 'w'
            # open position file, write to it. 
            print('writing to pos files')
            x_output = open(filename_x, append_write)
            x.astype(np.float32).tofile(x_output)
            x_output.close()
            #x_output.write(x.astype(str))
            y_output = open(filename_y, append_write)
            y.astype(np.float32).tofile(y_output)
            y_output.close()
            #y_output.write(y.astype(str))
            z_output = open(filename_z, append_write)
            z.astype(np.float32).tofile(z_output)
            z_output.close()
            #z_output.write(z.astype(str))
        if num_computed >= num_halos:
            break
    if rank ==0:
        filename_x = '%s_BCM/snapdir_%s/BCMpos_x.txt'%(full_path, str(params['snapNum']).zfill(3))
        filename_y = '%s_BCM/snapdir_%s/BCMpos_y.txt'%(full_path, str(params['snapNum']).zfill(3))
        filename_z = '%s_BCM/snapdir_%s/BCMpos_z.txt'%(full_path, str(params['snapNum']).zfill(3))
        
        num_lines = int(os.path.getsize(filename_x) / 4)
        start = 0
        stop = 0
        computed =0
        breakage=False
        print(f'Number of positions to fix = {num_lines}')
        for f_name in files:
            # Open a file and check that the length is shorter than the length of BCM
            print(f'Writing to {f_name}')
            f = h5py.File(f_name, 'r+')
            d = f['PartType1']['Coordinates']
            l_d = len(d)
            f.close()
            splits = 50
            l_ds, res = divmod(l_d, splits)
            print(f'res={res}')
            for split in range(splits):
                start = l_ds * split +computed
                stop = l_ds * (split + 1) + computed
                if stop >num_lines: 
                    stop = num_lines
                    breakage = True
        
                pos = []
                print(f'Picking files from {start} to {stop} on data thats {l_d} long')
                for f in [filename_x, filename_y, filename_z]:
                    pos.append(read_file(f, start, stop, num_lines))
                
                pos = np.array(pos).T
                with h5py.File(f_name, 'r+') as d:
                    d['PartType1']['Coordinates'][start-computed:stop-computed] = pos
                if breakage:
                    print('Should break here')
                    break
            computed += l_d
            if breakage:
                break
    comm.Barrier()
        
    if params['power']:
        power_start = time.time()
        mdi=None
        offset_path = params['basePath'] + params['sim_name'] +'/postprocessing/offsets/offsets_%s.hdf5'%(str(params['snapNum']).zfill(3))        
        dat =loadSubset(full_path + '_BCM', params['snapNum'], 'dm', fields='Coordinates', mdi=mdi, chunkNum=comm.rank, totNumChunks=comm.size, offset_path = offset_path) 
        dat /= 1000
        dat %= params['BoxSize']
        dat[np.isnan(dat)] =0
        dat = dat.astype(np.float32)

        # Put into a particle mesh object for mapping and power spectrum
        pm = ParticleMesh(Nmesh=[params['Nmesh']]*3, BoxSize=[params['BoxSize']]*3, resampler='cic')
        mass1 = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(dat), op=MPI.SUM) * particle_mass
        layout = pm.decompose(dat)
        pos1 = layout.exchange(dat)

        IllustrisMap = pm.create(type="real")
        IllustrisMap.paint(pos1, mass=mass1, layout=None, hold=False)
        del mass1, dat, pos1

        address = params['outPath']
        #FieldMesh(IllustrisMap).save(address_BCM)
        if not os.path.exists(address + 'power/'):
            os.makedirs(address + 'power/')
        address_BCM = address + 'power/' + params['constraint'] + 'map_Nmesh' + str(pm.Nmesh[0])

        P = FFTPower(IllustrisMap, mode='1d').save(address_BCM +'_p.json')
        if rank ==0:
            print(f'\npower calculation time: {time.time()-power_start}')
    """-------------------------------------------DMO-----------------------------"""
    comm.Barrier()
    pm = ParticleMesh(Nmesh=[params['Nmesh']]*3, BoxSize=[params['BoxSize']]*3, resampler='cic')
    mdi=None
    if params['DMO']:
        address = params['outPath']
        #FieldMesh(IllustrisMap).save(address_BCM)
        if not os.path.exists(address + 'power/'):
            os.makedirs(address + 'power/')
        address = params['outPath']
        address_DMO = address + 'power/' +'DMOmap_Nmesh' + str(pm.Nmesh[0])

        offset_path = params['basePath'] + params['sim_name'] +'/postprocessing/offsets/offsets_%s.hdf5'%(str(params['snapNum']).zfill(3))        
        pm = ParticleMesh(Nmesh=[params['Nmesh']]*3, BoxSize=[params['BoxSize']]*3, resampler='cic')
        try:
            IllustrisMap_DMO = BigFileMesh(address_DMO, dataset='Field').to_real_field()
        except:
            dat =loadSubset(full_path , params['snapNum'], 'dm', fields='Coordinates', mdi=mdi, chunkNum=comm.rank, totNumChunks=comm.size, offset_path = offset_path) 
            dat /= 1000
            dat %= params['BoxSize']
            dat = dat.astype(np.float32)
            layout = pm.decompose(dat)
            pos1 = layout.exchange(dat)

            mass1 = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(dat), op=MPI.SUM) * particle_mass
            del dat

            IllustrisMap_DMO = pm.create(type="real")
            IllustrisMap_DMO.paint(pos1, mass=mass1, layout=None, hold=False)
            del pos1, mass1

            #FieldMesh(IllustrisMap_DMO).save(address_DMO)
            P_DMO = FFTPower(IllustrisMap_DMO, mode='1d').save(address_DMO +'_p.json')

        # Cross correlation coefficient incase this is a useful statistic
    #    r_cc = FFTPower(IllustrisMap, second=IllustrisMap_DMO, mode='1d').save(address_BCM +'_rcc.json')


if __name__ == "__main__":
    app.run(main)
