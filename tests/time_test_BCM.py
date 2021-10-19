import numpy as np
from BCM_lensing.bcm_pos import BCM_pos
import warnings; warnings.simplefilter('ignore')
from absl import app, flags
import illustris_python as il
import pandas as pd
import time

FLAGS = flags.FLAGS
flags.DEFINE_integer('halo_num', 3, 'halo number')

def main(argv):
    del argv  # Unused.
    tt =[]
    print(FLAGS.halo_num)
    basePath = '../Illustris-3-Dark/output'
    particle_mass=4.8e-2

    subgroupFields = [ 'SubhaloIDMostbound', 'SubhaloLen', 'SubhaloMass','SubhaloParent', 'SubhaloPos' ]
    groupFields = [ 'GroupFirstSub', 'GroupLen', 'GroupMass','GroupNsubs', 'Group_M_Crit200','Group_R_Crit200', 'GroupPos', 'Group_R_Mean200']

    subgroups = il.groupcat.loadSubhalos(basePath,135, fields=subgroupFields)
    groups = il.groupcat.loadHalos(basePath,135, fields=groupFields)

    groupPos = groups.pop('GroupPos')
    subgroupPos = subgroups.pop('SubhaloPos')

    group_df = pd.DataFrame(groups)
    subgroup_df = pd.DataFrame(subgroups)

    group_df['GroupMass']  = group_df['GroupMass']
    group_df['Group_M_Crit200']  = group_df['Group_M_Crit200']
    group_df['Group_R_Crit200']  = group_df['Group_R_Crit200']

    for RUN in range(100):
        start = time.time()
        _ = BCM_pos(FLAGS.halo_num, group_df, groupPos, subgroupPos, resolution=20)
        end = time.time()
        tt.append(end - start)
        print(f'Total time {RUN}: {end-start}')
    print(f'Avg time: {np.mean(tt)} plus or minus {np.std(tt)}')


if __name__ == "__main__":
    app.run(main)

