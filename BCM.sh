#!/bin/sh
#SBATCH --account=astro        # Replace ACCOUNT with your group account name
#SBATCH --job-name=BCM_z_0    # The job name
#SBATCH -c 32                     # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=0-0:10            # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=5G         # The memory the job will use per cpu core
 
module load anaconda
source ~/.bashrc
conda activate BCMLensing_env

python run_BCM.py --constraint=CG&
python run_BCM.py --constraint=BG&
python run_BCM.py --constraint=EG&
python run_BCM.py --constraint=RDM&
wait
# End of script
