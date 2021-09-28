# Baryon Correction Model

Here I have implemented the BCM as described by Schndeider, Teyssier, Arico etc.

I will use this model with Lens tools to perform weak lensing calculations and analyze the BCM parameters


## Getting the Environment set
I built a yml file with the basic dependencies. To get started run: 

`conda env create -f environment.yml`

Then 

`conda activate BCMLensing_env`

the Attatched jupyter notebook has some examples of how I am implementing the BCM, and run_BCM.py performs the BCM on a halo and can be used for profiling. 

## Data

I put one halos subhalo particles in the directory Data. though if you change directories, make sure you change the arguments to the run_BCM.py file. 


