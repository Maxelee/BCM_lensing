#!/bin/bash

#SBATCH -J BCM   # Job name
#SBATCH -o BCM.o%j # Name of stdout output file
#SBATCH -e BCM.e%j # Name of stderr error file
#SBATCH -p normal    # Queue (partition) name
#SBATCH -N 8       # Total # of nodes (must be 1 for serial)
#SBATCH -n 136      # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 12:00:00   # Run time (hh:mm:ss)


module restore
module list
export PYTHONPATH="/home1/08434/tg877334/.local/lib/python3.8/site-packages"


ibrun python3 -u run_BCM2.py --snapNum=96
ibrun python3 -u run_BCM2.py --snapNum=90
ibrun python3 -u run_BCM2.py --snapNum=85
ibrun python3 -u run_BCM2.py --snapNum=80
ibrun python3 -u run_BCM2.py --snapNum=76
ibrun python3 -u run_BCM2.py --snapNum=71
ibrun python3 -u run_BCM2.py --snapNum=67
ibrun python3 -u run_BCM2.py --snapNum=63
ibrun python3 -u run_BCM2.py --snapNum=59
ibrun python3 -u run_BCM2.py --snapNum=56
ibrun python3 -u run_BCM2.py --snapNum=52
ibrun python3 -u run_BCM2.py --snapNum=49
ibrun python3 -u run_BCM2.py --snapNum=46
ibrun python3 -u run_BCM2.py --snapNum=43
ibrun python3 -u run_BCM2.py --snapNum=41
ibrun python3 -u run_BCM2.py --snapNum=38
ibrun python3 -u run_BCM2.py --snapNum=35
ibrun python3 -u run_BCM2.py --snapNum=33
ibrun python3 -u run_BCM2.py --snapNum=31
ibrun python3 -u run_BCM2.py --snapNum=29


